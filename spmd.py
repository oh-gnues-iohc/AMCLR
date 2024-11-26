#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import numpy as np
import contextlib

import datasets
import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from accelerate import init_empty_weights

import torch_xla
import torch_xla.debug.profiler as xp
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as spmd

# Enable SPMD mode execution
xr.use_spmd()

# Enable compile cache
xr.initialize_cache("/root/files/tpu_cache", readonly=False)


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.


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
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
            )
        model_args.token = model_args.use_auth_token

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
    datasets.utils.logging.set_verbosity(log_level)
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

    server = xp.start_server(9012)
    logger.info(f"Profiling server started: {str(server)}")

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    
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

    
    

    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs


    # Place DCN on an independent axis in the mesh. Model parameters should be
    # replicated along the DCN axis, and inputs and activations should have
    # the batch dimension sharded along the combined DCN and data axes.
    num_devices = xr.global_runtime_device_count()
    model_axis = max(model_args.spmd_2d_sharding, 1)
    assert (
        xr.device_type() == "TPU" or xr.device_type() == "CUDA"
    ), f"Supported hardware are TPU and CUDA. Detected hardware: {xr.device_type()}"
    if xr.device_type() == "TPU":
        dcn_axis = model_args.spmd_dcn_parallelism
        data_axis = num_devices // model_axis // dcn_axis
        ici_mesh_shape = (1, data_axis, model_axis)
        dcn_mesh_shape = (dcn_axis, 1, 1)
        spmd_mesh = xs.HybridMesh(
            ici_mesh_shape=ici_mesh_shape,
            dcn_mesh_shape=dcn_mesh_shape,
            axis_names=("dcn", "data", "model"),
        )
        print("spmd_mesh:", spmd_mesh)
        xs.set_global_mesh(spmd_mesh)
    elif xr.device_type() == "CUDA":
        data_axis = num_devices // model_axis
        mesh_shape = (1, data_axis, model_axis)
        spmd_mesh = xs.Mesh(
            np.arange(num_devices), mesh_shape, ("dcn", "data", "model")
        )

    # Update training args with relevant SPMD config
    training_args.spmd_mesh = spmd_mesh
    training_args.spmd_fsdp_sharding = model_args.spmd_fsdp_sharding

    # Replace the linear layer
    from torch_xla.distributed.fsdp.utils import apply_xla_patch_to_nn_linear

    model = apply_xla_patch_to_nn_linear(model, xs.xla_patched_nn_linear_forward)

    # Set the dtype, and move to the XLA device when parameters are already initialized
    if model_args.spmd_defer_init:
        model = model.to(dtype=getattr(torch, model_args.torch_dtype))
    else:
        model = model.to(xm.xla_device(), dtype=getattr(torch, model_args.torch_dtype))

    # Shard each parameter in the model based on the sharding strategy provided.
    for name, param in model.named_parameters():
        if model_args.spmd_defer_init:
            with torch.no_grad():
                param = torch.empty_like(param, device="cpu")
                # TODO(jonbolin): Currently, deferred initialization ignores any custom
                # weight initialization in the model.
                torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                param = torch.nn.Parameter(param.to(xm.xla_device()))
                # Find the corresponding module
                path = name.split(".")
                module = model
                for module_name in path[:-1]:
                    module = dict(module.named_children())[module_name]
                # Replace the meta tensor parameter with the initialized XLA tensor
                module.register_parameter(path[-1], param)

        if model_args.spmd_fsdp_sharding:
            print("> [FSDP] Sharding tensor", name, param.shape, param.dtype)
            # We don't care about layernorm's weights, and
            # LLaMA doesn't use biases.
            if len(param.shape) == 1:
                continue
            assert len(param.shape) == 2

            # Shard the largest dimension
            if param.shape[0] > param.shape[1]:
                partition_spec = ("data", None)
            else:
                partition_spec = (None, "data")
            xs.mark_sharding(param, spmd_mesh, partition_spec)
        elif model_args.spmd_2d_sharding > 0:
            # Apply 2D sharding:
            print("> [2D] Sharding tensor", name, param.shape)

            # We don't care about layernorm's weights, and
            # LLaMA doesn't use biases.
            if len(param.shape) == 1:
                continue
            
            if "embeddings" in name or "word_embeddings" in name:
                xs.mark_sharding(param, spmd_mesh, ("model", "data"))

            # Self-Attention layers
            elif "attention.self.query" in name or "attention.self.key" in name or "attention.self.value" in name:
                xs.mark_sharding(param, spmd_mesh, ("data", "model"))
            elif "attention.self.output.dense" in name:
                xs.mark_sharding(param, spmd_mesh, ("model", "data"))

            # Feed-Forward Network (FFN)
            elif "intermediate.dense" in name:  # FFN's intermediate layer
                xs.mark_sharding(param, spmd_mesh, ("model", "data"))
            elif "output.dense" in name:  # FFN's output layer
                xs.mark_sharding(param, spmd_mesh, ("data", "model"))

            # Discriminator predictions
            elif "discriminator_predictions.dense" in name or "dense_prediction" in name or "cls_representation" in name:
                xs.mark_sharding(param, spmd_mesh, ("model", "data"))

            # Generator predictions
            elif "generator_predictions.dense" in name or "lm_head" in name or "generator_score_head" in name:
                xs.mark_sharding(param, spmd_mesh, ("model", "data"))

        print(f"{name} {torch_xla._XLAC._get_xla_sharding_spec(param)}")


    for i, layer in enumerate(model.electra.encoder.layer):
        spmd.xla_sharding.apply_backward_optimization_barrier(layer)
    for i, layer in enumerate(model.generator.electra.encoder.layer):
        spmd.xla_sharding.apply_backward_optimization_barrier(layer)

    # for i, block in enumerate(model.model.layers):
    #     # LLaMA-specific
    #     # xs.apply_backward_optimization_barrier(model.model.layers[i])
    #     spmd.xla_sharding.apply_backward_optimization_barrier(model.model.layers[i])

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets,
        tokenizer=tokenizer,
    )

    # Test model saving
    # if training_args.do_train:
    #     print("Saving model before training")
    #     trainer.save_model()
    #     print("Model saved")

    trainer.train()
        # TODO(jonbolin): For our benchmarks, we don't need to persist the final training result.
        # This should be re-enabled if the final result is needed in a non-checkpoint form.
        # trainer.save_model()  # Saves the tokenizer too for easy upload



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()