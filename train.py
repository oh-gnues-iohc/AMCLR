import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from transformers.models.electra import ElectraConfig

# Enable SPMD mode execution and compile cache for TPU
xr.use_spmd()
xr.initialize_cache("/root/files/tpu_cache", readonly=False)

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


def main():
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

    # Log training parameters
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set random seed
    set_seed(training_args.seed)

    # Load dataset
    datasets = load_dataset(data_args.dataset_name)

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

    # Initialize Trainer
    trainer = Trainer(
        model=disc,
        args=training_args,
        train_dataset=datasets,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()


if __name__ == "__main__":
    main()
