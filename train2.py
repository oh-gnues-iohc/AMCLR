import logging
import os
import sys
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from datasets import Dataset
from dataclasses import dataclass, field
import torch.distributed as dist
from typing import Optional
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu

import torch_xla.distributed.parallel_loader as pl
import torch
import wandb
import torch_xla
import torch_xla.runtime as xr


from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
    get_scheduler,
    DefaultDataCollator
)
from transformers.models.electra import ElectraConfig

logger = logging.getLogger(__name__)

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


def unwrap_model(model):
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model

# xr.use_spmd()
def main(rank):
    
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    
    new_rank = xr.global_ordinal()
    assert new_rank == rank
    world_size = xr.world_size()

    dist.init_process_group('xla', init_method='xla://')
    logger.info(f"Running basic DDP example on rank {rank}.")

    if rank==0:
        wandb.init(project="my_project", name=f"my_run_{world_size}")
    set_seed(training_args.seed)
    datasets = load_from_disk(data_args.dataset_name)["train"].shuffle(8324)

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
    device = xm.xla_device()
    gen = gen_model(ElectraConfig.from_pretrained(gen_config_path), tokenizer.all_special_ids).to(device)
    model = disc_model(ElectraConfig.from_pretrained(disc_config_path), tokenizer.all_special_ids, gen).to(device)

    xm.broadcast_master_param(model)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate, eps=training_args.adam_epsilon)
    
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        datasets,
        num_replicas=xr.world_size(),
        rank=xr.global_ordinal(),
        shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        datasets,
        batch_size=8,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=DefaultDataCollator(),
        shuffle=False if train_sampler else True,
        num_workers=4)
    import math
    num_update_steps_per_epoch = math.ceil(len(train_loader))
    num_train_epochs = math.ceil(training_args.max_steps / num_update_steps_per_epoch)
    
    train_device_loader = pl.MpDeviceLoader(train_loader, device, loader_prefetch_size=32, device_prefetch_size=16, host_to_device_transfer_threads=4)
    
    from tqdm import tqdm
    gloabl_batch = 0
    loss = 0.0
    progress_bar = tqdm(range(training_args.max_steps), disable=not xm.is_master_ordinal(local=False))
    for epoch in range(0, num_train_epochs):
        model.train()
        for step, batch in enumerate(train_device_loader):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs[0]
            gloabl_batch = outputs[1]
            loss.backward()
            xm.optimizer_step(optimizer, barrier=True)
            lr_scheduler.step()
            
            global_step = epoch * num_update_steps_per_epoch + step
            progress_bar.update(1)
            
            if global_step > 0 and global_step % training_args.save_steps == 0:
                xm.rendezvous("before_saving_checkpoint")
                if xm.is_master_ordinal(local=False):
                    unwrapped_model = unwrap_model(model)
                    gen = unwrapped_model.gen
                    disc = unwrapped_model.electra
                    save_path = os.path.join(training_args.output_dir, f"disc-checkpoint-{global_step}")
                    xm.master_print(f"Saving model checkpoint to {save_path}")
                    disc.save_pretrained(
                    save_path,
                    state_dict=xm._maybe_convert_to_cpu(disc.state_dict()),
                    save_function=xm.save,
                    )
                    save_path = os.path.join(training_args.output_dir, f"gen-checkpoint-{global_step}")
                    xm.master_print(f"Saving model checkpoint to {save_path}")
                    gen.save_pretrained(
                    save_path,
                    state_dict=xm._maybe_convert_to_cpu(gen.state_dict()),
                    save_function=xm.save,
                    )
                    
                xm.rendezvous("after_saving_checkpoint")

            # 특정 스텝마다 wandb에 로깅
            if global_step % training_args.logging_steps == 0:
                if xm.is_master_ordinal(local=False):
                    current_lr = optimizer.param_groups[0]["lr"]
                    loss = loss.detach().to("cpu").item()
                    wandb.log({"loss": loss, "lr": current_lr, "gloabl_batch": gloabl_batch}, step=global_step)
                    progress_bar.set_postfix({"loss": loss, "gloabl_batch": gloabl_batch, "global_step": global_step})
            
            if global_step >= training_args.max_steps:
                break
    

def _mp_fn(index):
    main(index)


if __name__ == "__main__":
    torch_xla.launch(_mp_fn, args=())
    # main()