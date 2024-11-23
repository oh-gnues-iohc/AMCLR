import os
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.experimental.pjrt_backend  # Required for `xla://` init_method
import torch_xla.utils.utils as xu
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import os
import sys
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from datasets import Dataset
from dataclasses import dataclass, field
from typing import Optional
import torch_xla.distributed.xla_multiprocessing as xmp

import torch_xla
import torch_xla.runtime as xr


from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
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


def train_tpu(rank):
    # Parse arguments
    device = xm.xla_device()
    torch.distributed.init_process_group('xla', init_method='xla://')
    torch.manual_seed(42)
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # Log training parameters
    logger.info(f"Training/evaluation parameters {training_args}")

    # Load dataset
    datasets = load_from_disk(data_args.dataset_name)

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

    # Initialize the process group

    datasets.set_format("torch")

    # Prepare dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        datasets["train"], batch_size=training_args.per_device_train_batch_size, shuffle=True
    )
    # Load model
    model = disc.to(device)

    # Broadcast parameters for consistency across replicas
    xm.broadcast_master_param(model)

    # Wrap model in DDP
    model = DDP(model, gradient_as_bucket_view=True)

    # Define optimizer, loss, and scheduler
    num_training_steps = training_args.max_steps  # Max steps
        
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.max_steps
    )
    # Training loop
    model.train()
    para_loader = pl.MpDeviceLoader(train_dataloader, device)
    step = 0

    # Infinite loop with itertools.cycle
    for batch in itertools.cycle(para_loader):
        if step >= num_training_steps:
            break

        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        loss = model(input_ids, attention_mask=attention_mask)
        loss.backward()

        xm.optimizer_step(optimizer)
        scheduler.step()
        xm.mark_step()

        # Log training progress
        if step % 10 == 0 and xm.is_master_ordinal():
            tqdm.write(f"Step {step}: Loss = {loss.item()}")
        if step % training_args.save_steps == 0 and xm.is_master_ordinal():
            model.module.electra.save_pretrained(f"./output/step-{step}")
            model.module.generator.electra.save_pretrained(f"./output/step-{step}/gens")
        step += 1

    # Save model (only on master replica)
    if xm.is_master_ordinal():
        os.makedirs("saved_model", exist_ok=True)
        model.save_pretrained("saved_model")
        tokenizer.save_pretrained("saved_model")


# Start training with TPU
if __name__ == "__main__":
    torch_xla.launch(train_tpu, args=())