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
from jax.sharding import Mesh, PartitionSpec
from jax.experimental.pjit import pjit
from functools import partial

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

def get_sharded_batches(dataset, batch_size):
    """
    효율적인 데이터 로딩을 위해 사전 샤딩된 배치를 생성하는 제너레이터 함수입니다.
    """
    for batch in dataset:
        # 각 배치의 데이터를 NumPy 배열로 변환하고 샤딩
        yield shard({k: np.array(v) for k, v in batch.items()})

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

from jax.lax import with_sharding_constraint as _with_sharding_constraint
from jax.interpreters import pxla

def names_in_current_mesh(*names):
    """ Check if current mesh axes contain these names. """
    mesh_axis_names = pxla.thread_resources.env.physical_mesh.axis_names
    return set(names) <= set(mesh_axis_names)


def get_names_from_parition_spec(partition_specs):
    """ Return axis names from partition specs. """
    names = set()
    if isinstance(partition_specs, dict):
        partition_specs = partition_specs.values()
    for item in partition_specs:
        if item is None:
            continue
        elif isinstance(item, str):
            names.add(item)
        else:
            names.update(get_names_from_parition_spec(item))

    return list(names)
def with_sharding_constraint(x, partition_specs):
    """ A smarter version of with_sharding_constraint that only applies the
        constraint if the current mesh contains the axes in the partition specs.
    """
    axis_names = get_names_from_parition_spec(partition_specs)
    if names_in_current_mesh(*axis_names):
        x = _with_sharding_constraint(x, partition_specs)
    return x

def manual_shard(xs):
    """Helper for pjit to shard a pytree of arrays by global_device_count.

    Args:
        xs: a pytree of arrays.
    Returns:
        A matching pytree with arrays' leading dimensions sharded by the
        global device count.
    """
    global_device_count = jax.device_count()
    return jax.tree_util.tree_map(
        lambda x: x.reshape((global_device_count, -1) + x.shape[1:]), xs
    )

def shard_rngs(rngs, global_device_count):
    """
    Split RNGs across all devices globally.

    Args:
        rngs: A dictionary of RNGs to split.
        global_device_count: Total number of devices.

    Returns:
        A dictionary of sharded RNGs.
    """
    return {k: jax.random.split(v, global_device_count) for k, v in rngs.items()}

def prepare_rngs(rng, global_device_count):
    """
    Prepare RNGs to match the global device count.

    Args:
        rng: The base RNG key.
        global_device_count: Total number of devices.

    Returns:
        Sharded RNGs array with shape (global_device_count, ...).
    """
    return jax.random.split(rng, global_device_count)


def main():
    # Initialize JAX distributed backend
    jax.distributed.initialize()
    
    devices = np.array(jax.devices()).reshape((32,))
    mesh = Mesh(devices, ('dp',))
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
        rng, gumbel_rngs, dropout_rngs = jax.random.split(rng, 3)

        # Create rngs dictionary
        rngs = {'gumbel': gumbel_rngs, 'dropout': dropout_rngs}

        # Load tokenizer
        disc_config_path = f"google/electra-{model_args.model_size}-discriminator"
        gen_config_path = f"google/electra-{model_args.model_size}-generator"

        tokenizer = AutoTokenizer.from_pretrained(disc_config_path)

        # Initialize models based on model_type
        if model_args.model_type == "AMCLR":
            discriminator = AMCLRModule(ElectraConfig.from_pretrained(disc_config_path), ElectraConfig.from_pretrained(gen_config_path), tokenizer.all_special_ids)
            model = discriminator  # Assuming discriminator is the main model for training
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
        rngs["params"] = params_rng
        # params = model.init(rngs, input_ids=jnp.ones((1, 1)), attention_mask=jnp.ones((1, 1)), is_training=False)

        @jax.pmap
        def init_step(key):
            input_ids = jnp.ones([32, 512])
            attention_mask = jnp.ones([32, 512])
            variables = model.init(key, input_ids=input_ids, attention_mask=attention_mask, is_training=False)
            opt_state = optimizer.init(variables['params'])
            return variables, opt_state

        # Replicate parameters across all devices (data parallel replicas)
        # replicated_params = jax_utils.replicate(params)

        # def create_trainstate_from_params(params):
        #     return train_state.TrainState.create(params=params, tx=optimizer, apply_fn=None)
        
        # sharded_create_trainstate_from_params = pjit(
        #     create_trainstate_from_params,
        #     in_shardings=(replicated_params,),
        #     out_shardings=train_state_partition,
        #     donate_argnums=(0,),
        # )


        # Define TrainState with sharded parameters
        state = train_state.TrainState.create(apply_fn=None, params=replicated_params, tx=optimizer)

        # Define sharding specifications
        # 파라미터는 모든 데이터 병렬 축('dp')에 복제되어야 하므로 PartitionSpec() 사용
        # 입력 데이터는 'dp' 축을 따라 샤딩됨
        # RNGs도 'dp' 축을 따라 샤딩됨
        input_sharding = PartitionSpec()
        params_sharding = PartitionSpec()  # Replicated
        rng_sharding = PartitionSpec()

        # Define pjit training step
        @partial(jax.pmap, axis_name="dp")
        def train_step(batch, variables, opt_state, rngs, steps):
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
            params = variables['params']
            # Split dropout RNG
            dropout_rng = rngs['dropout']
            gumbel_rngs = rngs['gumbel']
            dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
            gumbel_rngs, new_gumbel_rngs = jax.random.split(gumbel_rngs)
            
            rngs={
                'gumbel': new_gumbel_rngs,
                'dropout': new_dropout_rng,
            }

            def loss_fn(params, variables):
                """
                Defines the loss function by calling the model to compute the loss.
                """
                variables = variables.copy({'params': params})
                loss, updates = model.apply(
                    variables,
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch.get('token_type_ids', None),
                    position_ids=batch.get('position_ids', None),
                    # labels=batch['labels'],
                    is_training=True,  # Enable training-specific operations
                    deterministic=False,  # Enable dropout
                    rngs=rngs
                )
                variables = variables.copy(updates)
                return (loss, variables)

            # Compute loss and gradients
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, variables), grad = grad_fn(params, variables)

            # Sum the loss across devices
            loss = jax.lax.pmean(loss, axis_name='dp')

            # Average the gradients
            grad = jax.lax.pmean(grad, axis_name='dp')
            
            updates, opt_state = optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            variables = variables.copy({'params': params})
            
            batch_stats = jax.lax.pmean(variables['batch_stats'], "device")
            variables = variables.copy({'batch_stats': batch_stats})


            # Define metrics
            metrics = {
                "loss": loss,
                "learning_rate": linear_decay_lr_schedule_fn(steps)
            }

            return variables, opt_state, metrics, rngs

        replicated_rngs = rngs

        # Define save_checkpoint function
        def save_checkpoint(train_state, milestone=False):
            step = int(jax.device_get(train_state.step)[0])
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

        train_samples_idx = np.random.permutation(np.arange(len(train_dataset)))
        train_batch_idx = generate_batch_splits(train_samples_idx, global_train_batch_size)
        # Training loop
        current_step = 0
        with tqdm(total=num_steps, desc=f"Training Steps") as pbar:
            for batch_idx in tqdm(train_batch_idx, desc="Training...", position=1, leave=False, disable=(host_id != 0)):
                # Shard model inputs across devices
                samples = [train_dataset[int(idx)] for idx in batch_idx]
                # Assuming that each sample is a dictionary with necessary keys
                # Convert list of samples to batch dictionary
                batch = {k: np.stack([sample[k] for sample in samples]) for k in samples[0].keys()}
                
                # Shard model inputs across devices
                model_inputs = manual_shard(batch)

                # Call p_train_step with RNGs
                state, train_metric, replicated_rngs = train_step(
                    state,
                    model_inputs,
                    replicated_rngs
                )

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
