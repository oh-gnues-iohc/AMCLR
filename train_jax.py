#!/usr/bin/env python
# coding=utf-8

"""
Fine-tuning library models for masked language modeling with whole word masking on a
text file or a dataset using JAX/Flax on a TPU v4-64 cluster.
"""

import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset, load_from_disk
from flax import jax_utils, traverse_util
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
from tqdm import tqdm
from transformers import AutoTokenizer, ElectraConfig, HfArgumentParser, TrainingArguments, set_seed
from transformers.utils import send_example_telemetry
from jax.experimental import maps
from jax.sharding import Mesh
# User-defined model imports
from src.amclr.model_jax import AMCLRModule, AMCLRMLMModule

# Import wandb for logging
import wandb

# Model sizes and types
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
    # You can define additional training arguments here if needed.
    pass

def generate_batch_splits(samples_idx: np.ndarray, batch_size: int, drop_last=True) -> np.ndarray:
    """Generate batches of data for a specified batch size from sample indices."""
    num_samples = len(samples_idx)
    if drop_last:
        samples_to_remove = num_samples % batch_size
        if samples_to_remove != 0:
            samples_idx = samples_idx[:-samples_to_remove]
        sections_split = num_samples // batch_size
        samples_idx = samples_idx.reshape((sections_split, batch_size))
    else:
        sections_split = math.ceil(num_samples / batch_size)
        samples_idx = np.array_split(samples_idx, sections_split)
    return samples_idx


def initialize_generator(config: ElectraConfig, tokenizer: AutoTokenizer, rng: jax.random.PRNGKey) -> Dict[str, Any]:
    """
    생성자(Generator) 모듈의 파라미터를 초기화하는 함수입니다.
    """
    generator = AMCLRMLMModule(
        config=config,
        special_token_ids=tokenizer.all_special_ids,
        dtype=jnp.float32
    )
    
    # 예제 입력 데이터 생성
    batch_size = 1
    seq_length = 128
    input_ids = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    token_type_ids = jnp.zeros((batch_size, seq_length), dtype=jnp.int32)
    position_ids = jnp.arange(seq_length)[None, :]
    
    # RNG 딕셔너리 생성
    rngs_gen = {
        "gumbel": jax.random.split(rng, 2)
    }
    
    # 생성자 파라미터 초기화
    variables_gen = generator.init(
        rng,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        deterministic=True,
        rngs=rngs_gen
    )
    
    generator_params = variables_gen['params']
    return generator_params

def initialize_discriminator(config: ElectraConfig, tokenizer: AutoTokenizer, generator: AMCLRMLMModule, rng: jax.random.PRNGKey) -> Dict[str, Any]:
    """
    판별자(Discriminator) 모듈의 파라미터를 초기화하는 함수입니다.
    """
    discriminator = AMCLRModule(
        config=config,
        special_token_ids=tokenizer.all_special_ids,
        generator=generator,
        dtype=jnp.float32
    )
    
    # 예제 입력 데이터 생성
    batch_size = 1
    seq_length = 128
    input_ids = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    token_type_ids = jnp.zeros((batch_size, seq_length), dtype=jnp.int32)
    position_ids = jnp.arange(seq_length)[None, :]
    labels = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    
    # RNG 딕셔너리 생성
    rngs_disc = {
        "gumbel": jax.random.split(rng, 2)
    }
    
    # 판별자 파라미터 초기화
    variables_disc = discriminator.init(
        rng,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        labels=labels,
        deterministic=True,
        rngs=rngs_disc
    )
    
    discriminator_params = variables_disc['params']
    return discriminator_params


def main():
    # Initialize JAX distributed backend
    jax.distributed.initialize()
    
    devices = np.array(jax.devices()).reshape((8, 4))

    mesh = Mesh(devices, ('dp', 'mp'))
    with mesh:
        # Parse arguments
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArgumentsExtended))
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()


        # Check output directory on master node
        host_id = jax.process_index()
        num_hosts = jax.process_count()
        if (
            host_id == 0 and
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

        # Initialize wandb only on master node
        if host_id == 0:
            wandb.init(project=training_args.output_dir.split('/')[-1], config=vars(training_args))
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                level=logging.INFO,
                datefmt="[%X]",
            )
            logger = logging.getLogger(__name__)
            logger.info(f"Training parameters {training_args}")
            logger.info(f"Number of Hosts: {num_hosts}")
        else:
            logger = logging.getLogger(__name__)

        # Set seed
        set_seed(training_args.seed)

        # Load dataset
        datasets = load_from_disk(data_args.dataset_name)
        train_dataset = datasets['train']

        # Initialize RNGs
        rng = jax.random.PRNGKey(training_args.seed)
        rng, init_rng = jax.random.split(rng)
        dropout_rngs = jax.random.split(init_rng, jax.local_device_count())
        gumbel_rngs = jax.random.split(init_rng, jax.local_device_count())

        # Create rngs dictionary
        rngs = {'gumbel': gumbel_rngs}

        # Load tokenizer
        disc_config_path = f"google/electra-{model_args.model_size}-discriminator"
        gen_config_path = f"google/electra-{model_args.model_size}-generator"

        tokenizer = AutoTokenizer.from_pretrained(disc_config_path)

        # Initialize models based on model_type
        if model_args.model_type == "AMCLR":
            # gen = AMCLRMLMModule(, tokenizer.all_special_ids)
            model = AMCLRModule(ElectraConfig.from_pretrained(disc_config_path), ElectraConfig.from_pretrained(gen_config_path), tokenizer.all_special_ids)
        else:
            raise NotImplementedError(f"Model type {model_args.model_type} is not implemented.")

        # Adjust batch sizes based on total devices
        num_steps = int(training_args.max_steps)
        global_train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
        num_train_steps = num_steps
        if host_id == 0:
            logger.info(f"Global train batch size: {global_train_batch_size}")

        # Learning rate schedule
        warmup_fn = optax.linear_schedule(
            init_value=0.0, end_value=training_args.learning_rate, transition_steps=training_args.warmup_steps
        )
        decay_fn = optax.linear_schedule(
            init_value=training_args.learning_rate,
            end_value=0,
            transition_steps=num_train_steps - training_args.warmup_steps,
        )
        linear_decay_lr_schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, decay_fn], boundaries=[training_args.warmup_steps]
        )

        # Optimizer
        if training_args.adafactor:
            optimizer = optax.adafactor(
                learning_rate=linear_decay_lr_schedule_fn,
            )
        else:
            optimizer = optax.adamw(
                learning_rate=linear_decay_lr_schedule_fn,
                b1=training_args.adam_beta1,
                b2=training_args.adam_beta2,
                eps=training_args.adam_epsilon,
                weight_decay=training_args.weight_decay,
            )

        # Initialize model parameters
        rng, params_rng = jax.random.split(rng)
        params = model.init(params_rng, input_ids=jnp.ones((1, 1)), attention_mask=jnp.ones((1, 1)))

        # Train state
        state = train_state.TrainState.create(apply_fn=model.__call__, params=params, tx=optimizer)

        # Define training step
        def train_step(state, batch, dropout_rng, rngs):
            """
            Perform a single training step.

            Args:
                state: Current training state.
                batch: Training data batch.
                dropout_rng: RNG for dropout.
                rngs: RNGs for random operations in the model.

            Returns:
                New training state, metrics, updated dropout_rng.
            """
            # Split dropout RNG
            dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

            def loss_fn(params):
                """
                Defines the loss function by calling the model to compute the loss.
                """
                logits = state.apply_fn(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch.get('token_type_ids', None),
                    position_ids=batch.get('position_ids', None),
                    labels=batch['labels'],
                    deterministic=False,  # Enable dropout
                    rngs=rngs,            # RNGs for random operations
                    params=params,        # Current parameters
                )
                loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['labels']).mean()
                return loss

            # Compute loss and gradients
            loss, grad = jax.value_and_grad(loss_fn)(state.params)

            # Sum the loss across devices
            loss = jax.lax.pmean(loss, axis_name='dp')

            # Average the gradients
            grad = jax.lax.pmean(grad, axis_name='dp')

            # Update parameters using optimizer
            new_state = state.apply_gradients(grads=grad)

            # Define metrics
            metrics = {
                "loss": loss,
                "learning_rate": linear_decay_lr_schedule_fn(state.step)
            }

            return new_state, metrics, new_dropout_rng

        # Parallelize train_step
        p_train_step = jax.pmap(train_step, axis_name='dp', donate_argnums=(0, 1, 2))

        # Replicate state across devices
        state = jax_utils.replicate(state)

        # Replicate RNGs across devices
        rngs = jax_utils.replicate(rngs)

        # Initialize dropout_rngs
        dropout_rngs = jax_utils.replicate(dropout_rngs)

        # Initialize step counter
        current_step = 0

        # Define save_checkpoint function
        def save_checkpoint(train_state, milestone=False):
            step = int(jax.device_get(train_state.step[0]))
            metadata = dict(
                step=step,
                variant={},  # Replace with your variant configuration
                flags={},    # Replace with your flags configuration
                gemma_config={},  # Replace with your gemma_config
            )
            # Implement your own checkpointer or adjust as per your setup
            # For demonstration, we'll simply save the state dict
            output_dir = os.path.join(training_args.output_dir, f"checkpoint-{step}")
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "train_state.msgpack"), "wb") as f:
                f.write(flax.serialization.to_bytes(train_state))
            with open(os.path.join(output_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f)
            logger.info(f"Saved checkpoint at step {step}")

        # Training loop
        with tqdm(total=num_steps, desc=f"Training Steps") as pbar:
            while current_step < num_steps:
                # Calculate remaining steps
                remaining_steps = num_steps - current_step
                # Determine steps in this epoch
                steps_in_epoch = min(remaining_steps, len(train_dataset) // global_train_batch_size)
                
                # Shuffle and split data
                train_samples_idx = np.random.permutation(np.arange(len(train_dataset)))
                train_batch_idx = generate_batch_splits(train_samples_idx, global_train_batch_size)

                for batch_idx in tqdm(train_batch_idx, desc="Training...", position=1, leave=False, disable=(host_id != 0)):
                    # Prepare batch
                    samples = [train_dataset[int(idx)] for idx in batch_idx]
                    # Assuming that each sample is a dictionary with necessary keys
                    # Convert list of samples to batch dictionary
                    batch = {k: np.stack([sample[k] for sample in samples]) for k in samples[0].keys()}
    
                    # Shard model inputs across devices
                    model_inputs = shard(batch)
    
                    # Call p_train_step with RNGs
                    state, train_metric, new_dropout_rng = p_train_step(
                        state,
                        model_inputs,
                        dropout_rngs,
                        rngs
                    )
    
                    # Update RNGs
                    dropout_rngs = new_dropout_rng
                    rngs = {'gumbel': jax.random.split(jax_utils.unreplicate(rngs['gumbel']), jax.local_device_count())}
                    rngs = jax_utils.replicate(rngs)
    
                    # Update step counter
                    current_step += 1
                    pbar.update(1)
    
                    # Logging and checkpointing
                    if current_step % training_args.logging_steps == 0 and host_id == 0:
                        # Collect metrics across devices
                        train_metric = jax_utils.unreplicate(train_metric)
                        loss = float(train_metric['loss'])
                        learning_rate = float(train_metric['learning_rate'])
    
                        # Log to wandb
                        wandb.log({
                            "train_loss": loss,
                            "learning_rate": learning_rate,
                            "step": current_step
                        }, step=current_step)
    
                        logger.info(
                            f"Step {current_step}: Loss = {loss}, Learning Rate = {learning_rate}"
                        )
    
                    if current_step % training_args.save_steps == 0 and host_id == 0:
                        # Save checkpoint
                        save_checkpoint(state, milestone=True)
    
                    if current_step >= num_steps:
                        break

        # Final checkpoint after training
        if host_id == 0:
            save_checkpoint(state, milestone=True)
            wandb.finish()

if __name__ == "__main__":
    main()
