import os
import sys
import logging as lll
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import flax
import flax.linen as nn
import optax
from flax import jax_utils
from flax.training import train_state

from datasets import load_dataset, load_from_disk

from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    set_seed,
    ElectraConfig,
    TrainingArguments
)

import jax
import jax.numpy as jnp
from jax import random
from src.amclr.model_jax import *

# Import your model classes
# Ensure that these are correctly imported from your model code
# from your_model_file import AMCLRModule, AMCLRMLMModule

logger = lll.getLogger(__name__)

@dataclass
class ModelArguments:
    model_size: Optional[str] = field(
        default=None,
        metadata={"help": "Model size: small, base, large"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "Model type: AMCLR, ELECTRA, AMOS"},
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the dataset to use (via the datasets library)."},
    )


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set random seed
    set_seed(training_args.seed)
    jax.distributed.initialize()

    # Get the devices
    devices = jax.devices()
    num_devices = len(devices)
    logger.info(f"Using {num_devices} devices: {devices}")

    # Load dataset
    datasets = load_from_disk(data_args.dataset_name)["train"]
    # Assume 'text' is the field in your dataset
    column_names = datasets.column_names

    # Initialize tokenizer
    disc_config_path = f"google/electra-{model_args.model_size}-discriminator"
    tokenizer = AutoTokenizer.from_pretrained(disc_config_path)


    # Convert to numpy arrays
    train_dataset = datasets

    # Compute batch size
    total_train_batch_size = training_args.per_device_train_batch_size * num_devices

    # Initialize model
    if model_args.model_type == "AMCLR":
        # Initialize generator and discriminator configurations
        gen_config_path = f"google/electra-{model_args.model_size}-generator"
        gen_config = ElectraConfig.from_pretrained(gen_config_path)
        disc_config = ElectraConfig.from_pretrained(disc_config_path)

        # Initialize models
        gen = AMCLRMLMModule(gen_config, tokenizer.all_special_ids)
        model = AMCLRModule(disc_config, tokenizer.all_special_ids, gen)
    else:
        # Handle other model types
        raise ValueError(f"Unsupported model type: {model_args.model_type}")

    # Create dummy inputs for initialization
    dummy_input = tokenizer("Hello, this is a dummy input", return_tensors='np')

    # Initialize model parameters
    rng = jax.random.PRNGKey(training_args.seed)
    rng, init_rng = jax.random.split(rng)
    params = model.init(
        init_rng,
        input_ids=dummy_input["input_ids"],
        attention_mask=dummy_input["attention_mask"],
        token_type_ids=dummy_input.get("token_type_ids", None),
        position_ids=None,
        labels=jnp.array([0]),
        deterministic=True,
        rngs={"gumbel": rng},
    )

    total_steps = training_args.max_steps

    # 워밍업 스텝 수 가져오기
    if hasattr(training_args, 'warmup_steps') and training_args.warmup_steps > 0:
        warmup_steps = training_args.warmup_steps
    elif hasattr(training_args, 'warmup_ratio') and training_args.warmup_ratio > 0:
        warmup_steps = int(training_args.warmup_ratio * total_steps)
    else:
        warmup_steps = 0
        
    learning_rate_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=training_args.learning_rate,
        transition_steps=warmup_steps,
    )
    decay_schedule_fn = optax.linear_schedule(
        init_value=training_args.learning_rate,
        end_value=0.0,
        transition_steps=total_steps - warmup_steps,
    )
    schedule_fn = optax.join_schedules(
        schedules=[learning_rate_fn, decay_schedule_fn],
        boundaries=[warmup_steps],
    )

    # 옵티마이저 생성
    tx = optax.adamw(
        learning_rate=schedule_fn,
        weight_decay=training_args.weight_decay,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
    )
    # Create train state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    # Replicate the state across devices
    state = jax_utils.replicate(state)

    # Prepare the training dataset
    def data_generator():
        num_examples = len(train_dataset)
        indices = np.arange(num_examples)
        rng_data = np.random.default_rng(training_args.seed)
        rng_data.shuffle(indices)

        while True:
            for start_idx in range(0, num_examples, total_train_batch_size):
                end_idx = start_idx + total_train_batch_size
                batch_indices = indices[start_idx:end_idx]
                batch = train_dataset.select(batch_indices)
                batch = {k: np.array(v) for k, v in batch.items()}
                # 배치를 샤딩
                batch = shard(batch)
                yield batch
            rng_data.shuffle(indices)

    def shard(xs):
        """Reshape the batch to have leading dimension equal to the number of devices."""
        return {k: xs[k].reshape((num_devices, -1) + xs[k].shape[1:]) for k in xs}

    # Define the training step function
    def train_step(state, batch, rng):
        def loss_fn(params):
            rngs = {'gumbel': rng}
            loss = state.apply_fn(
                **batch,
                params=params,
                deterministic=False,
                rngs=rngs,
            )
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        # 그래디언트를 복제 간에 집계
        grads = jax.lax.pmean(grads, axis_name='batch')
        new_state = state.apply_gradients(grads=grads)
        # 손실을 복제 간에 집계
        loss = jax.lax.pmean(loss, axis_name='batch')
        return new_state, loss

    # pmap the train_step function
    p_train_step = jax.pmap(train_step, axis_name='batch')

    # Training loop
    rngs = jax.random.split(rng, num_devices)
    global_step = 0
    batch_iter = data_generator()

    while global_step < total_steps:
        batch = next(batch_iter)
        # rngs 분할
        rngs = jax.random.split(rngs.reshape(-1), num_devices)
        state, loss = p_train_step(state, batch, rngs)
        global_step += 1
        # 손실 로깅
        if global_step % training_args.logging_steps == 0:
            # 디바이스 0에서 손실 가져오기
            loss_value = jax_utils.unreplicate(loss)
            logger.info(f"step {global_step} loss {loss_value}")
        # 체크포인트 저장
        if global_step % training_args.save_steps == 0:
            # 상태 저장
            # 상태를 unreplicate
            state_to_save = jax_utils.unreplicate(state)
            with open(os.path.join(training_args.output_dir, f"checkpoint_{global_step}.ckpt"), "wb") as f:
                f.write(flax.serialization.to_bytes(state_to_save))

    # 최종 모델 저장
    state_to_save = jax_utils.unreplicate(state)
    with open(os.path.join(training_args.output_dir, f"final_checkpoint.ckpt"), "wb") as f:
        f.write(flax.serialization.to_bytes(state_to_save))

if __name__ == "__main__":
    main()
