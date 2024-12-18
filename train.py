import logging
import os
import sys
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from datasets import Dataset
from dataclasses import dataclass, field
from typing import Optional

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


# xr.use_spmd()
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
    datasets = load_from_disk(data_args.dataset_name)["train"].shuffle(8324)

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
        train_dataset=datasets
    )

    # Train the model
    trainer.train()

def _mp_fn(index):
    # For xla_spawn (TPUs)
    xr.initialize_cache(f'/tmp/xla_cache_{index}', readonly=False)
    main()


if __name__ == "__main__":
    torch_xla.launch(_mp_fn, args=())
    # main()