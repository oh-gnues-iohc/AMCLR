import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import numpy as np
import torch

import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    get_scheduler
)
from tqdm import tqdm
import torch
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
from torch_xla import runtime as xr
import torch_xla
# Enable SPMD mode execution
xr.use_spmd()

logger = logging.getLogger(__name__)

def get_global_rank():
    return xm.get_ordinal()

MODEL_SIZES = ["small", "base", "large"]
MODEL_TYPES = ["AMCLR", "ELECTRA", "AMOS"]


@dataclass
class ModelArguments:
    model_size: Optional[str] = field(
        default=None,
        metadata={"help": f"Choose from: {', '.join(MODEL_SIZES)}"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": f"Choose from: {', '.join(MODEL_TYPES)}"},
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
@dataclass
class TrainingArgumentsExtended(TrainingArguments):
    pass

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArgumentsExtended)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    xm.set_rng_state(training_args.seed)
    
    from transformers.models.electra import ElectraConfig
    from datasets import load_dataset, load_from_disk
    
    datasets = load_from_disk(data_args.dataset_name)["train"]

    # Load tokenizer and model configurations
    disc_config_path = f"google/electra-{model_args.model_size}-discriminator"
    gen_config_path = f"google/electra-{model_args.model_size}-generator"

    if model_args.model_type == "AMCLR":
        from src.amclr.model import AMCLR, AMCLRMLM
        disc_model = AMCLR
        gen_model = AMCLRMLM
    else:
        if model_args.model_type == "AMOS":
            from src.amclr.model_amos import SimsElectra, SimsMLM
        else:
            from src.amclr.model_electra import SimsElectra, SimsMLM
        disc_model = SimsElectra
        gen_model = SimsMLM

    tokenizer = AutoTokenizer.from_pretrained(disc_config_path)

    # Initialize models
    gen = gen_model(ElectraConfig.from_pretrained(gen_config_path), tokenizer.all_special_ids)
    disc = disc_model(ElectraConfig.from_pretrained(disc_config_path), tokenizer.all_special_ids, gen)

    # Log model parameters
    n_params = sum(p.numel() for p in disc.parameters())
    logger.info(f"Training new model from scratch - Total size={n_params / 2**20:.2f}M params")


    num_devices = xr.global_runtime_device_count()
    device_ids = np.arange(num_devices)
    mesh_shape = (num_devices,)
    mesh = xs.Mesh(device_ids, mesh_shape, ('data',))
    batch_size = num_devices * training_args.per_device_train_batch_size
    
    # logger.info(f"SPMD Mesh: {mesh}, Batch size: {batch_size}")

    # Dataset preparation
    def collate_fn(batch):
        return {
            "input_ids": torch.tensor([x["input_ids"] for x in batch]),
            "attention_mask": torch.tensor([x["attention_mask"] for x in batch]),
            "labels": torch.tensor([x["labels"] for x in batch]),
        }


    train_dataset = datasets.with_format("torch")

    # DataLoader
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=None,
    )

    # Sharded DataLoader
    import torch_xla.distributed.parallel_loader as pl
    train_device_loader = pl.MpDeviceLoader(
        train_loader,
        xm.xla_device(),
        input_sharding=xs.ShardingSpec(mesh, ('data', None)),
    )
    args = training_args
    model = disc.to(xm.xla_device())
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)


    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_device_loader) / args.gradient_accumulation_steps)
    if args.max_steps is None:
        args.max_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps
    )
    
    num_update_steps_per_epoch = math.ceil(len(train_device_loader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.save_steps

    progress_bar = tqdm(range(training_args.max_steps), disable=not training_args.local_rank == 0)
    completed_steps = 0
    starting_epoch = 0
    
    
    def step_fn(batch):
        optimizer.zero_grad()
        loss = model(**batch)
        loss.backward()
        optimizer.step()
        return loss
    import os


    compiled_step_fn = torch_xla.compile(
        step_fn, full_graph=True, name="step_fn")

    progress_bar.update(completed_steps)
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        active_dataloader = train_device_loader
        for step, batch in enumerate(active_dataloader):
            loss = compiled_step_fn(batch)
            xm.mark_step()
            
            progress_bar.update(1)
            completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    xm.wait_device_ops()
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    model.electra.save_pretrained(output_dir)
                    model.generator.save_pretrained(os.path.join(output_dir, "gens"))
                    xm.wait_device_ops()
                    
            if completed_steps >= args.max_steps:
                xm.wait_device_ops()
                break


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()