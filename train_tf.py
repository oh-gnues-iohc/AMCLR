import tensorflow as tf, tf_keras
import numpy as np
from transformers import AdamWeightDecay
from transformers.optimization_tf import create_optimizer
from src.amclr.model_tf import AMCLR_TF, AMCLRConfig
from transformers import AutoTokenizer
import os
import math
import wandb

# 커스텀 AMCLRConfig, AMCLR_TF 클래스 정의
# 이미 정의되어 있다고 가정합니다.
# 만약 별도의 파일에 있다면, import 문을 사용하세요.
# 예: from my_model import AMCLRConfig, AMCLR_TF

# AMCLRConfig 및 AMCLR_TF 클래스가 이미 정의되어 있다고 가정합니다.


import sys
from typing import Any, Dict, Optional, Union


import wandb
from wandb.integration.keras.keras import patch_tf_keras
from wandb.sdk.lib import telemetry

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


LogStrategy = Literal["epoch", "batch"]


patch_tf_keras()


class WandbMetricsLogger(tf_keras.callbacks.Callback):
    """Logger that sends system metrics to W&B.

    `WandbMetricsLogger` automatically logs the `logs` dictionary that callback methods
    take as argument to wandb.

    This callback automatically logs the following to a W&B run page:
    * system (CPU/GPU/TPU) metrics,
    * train and validation metrics defined in `model.compile`,
    * learning rate (both for a fixed value or a learning rate scheduler)

    Notes:
    If you resume training by passing `initial_epoch` to `model.fit` and you are using a
    learning rate scheduler, make sure to pass `initial_global_step` to
    `WandbMetricsLogger`. The `initial_global_step` is `step_size * initial_step`, where
    `step_size` is number of training steps per epoch. `step_size` can be calculated as
    the product of the cardinality of the training dataset and the batch size.

    Arguments:
        log_freq: ("epoch", "batch", or int) if "epoch", logs metrics
            at the end of each epoch. If "batch", logs metrics at the end
            of each batch. If an integer, logs metrics at the end of that
            many batches. Defaults to "epoch".
        initial_global_step: (int) Use this argument to correctly log the
            learning rate when you resume training from some `initial_epoch`,
            and a learning rate scheduler is used. This can be computed as
            `step_size * initial_step`. Defaults to 0.
    """

    def __init__(
        self,
        log_freq: Union[LogStrategy, int] = "epoch",
        initial_global_step: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if wandb.run is None:
            raise wandb.Error(
                "You must call `wandb.init()` before WandbMetricsLogger()"
            )

        with telemetry.context(run=wandb.run) as tel:
            tel.feature.keras_metrics_logger = True

        if log_freq == "batch":
            log_freq = 1

        self.logging_batch_wise = isinstance(log_freq, int)
        self.log_freq: Any = log_freq if self.logging_batch_wise else None
        self.global_batch = 0
        self.global_step = initial_global_step

        if self.logging_batch_wise:
            # define custom x-axis for batch logging.
            wandb.define_metric("batch/batch_step")
            # set all batch metrics to be logged against batch_step.
            wandb.define_metric("batch/*", step_metric="batch/batch_step")
        else:
            # define custom x-axis for epoch-wise logging.
            wandb.define_metric("epoch/epoch")
            # set all epoch-wise metrics to be logged against epoch.
            wandb.define_metric("epoch/*", step_metric="epoch/epoch")

    def _get_lr(self) -> Union[float, None]:
        if isinstance(self.model.optimizer.learning_rate, tf.Variable):
            return float(self.model.optimizer.learning_rate.numpy().item())
        try:
            return float(
                self.model.optimizer.learning_rate(step=self.global_step).numpy().item()
            )
        except Exception:
            wandb.termerror("Unable to log learning rate.", repeat=False)
            return None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of an epoch."""
        logs = dict() if logs is None else {f"epoch/{k}": v for k, v in logs.items()}

        logs["epoch/epoch"] = epoch

        lr = self._get_lr()
        if lr is not None:
            logs["epoch/learning_rate"] = lr

        wandb.log(logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self.global_step += 1
        """An alias for `on_train_batch_end` for backwards compatibility."""
        if self.logging_batch_wise and batch % self.log_freq == 0:
            logs = {f"batch/{k}": v for k, v in logs.items()} if logs else {}
            logs["batch/batch_step"] = self.global_batch

            lr = self._get_lr()
            if lr is not None:
                logs["batch/learning_rate"] = lr

            wandb.log(logs)

            self.global_batch += self.log_freq

    def on_train_batch_end(
        self, batch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the end of a training batch in `fit` methods."""
        self.on_batch_end(batch, logs if logs else {})



class WarmUpLinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_steps, total_steps):
        super(WarmUpLinearDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        learning_rate = tf.cond(
            step < warmup_steps,
            lambda: self.initial_learning_rate * (step / warmup_steps),
            lambda: self.initial_learning_rate * (1 - (step - warmup_steps) / (total_steps - warmup_steps))
        )
        return learning_rate

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps
        }
        
def main():
    # 4개의 GPU를 사용하기 위해 MirroredStrategy 설정
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver("node-1")
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    # 데이터셋 생성
    
    GLOBAL_BATCH_SIZE = 256  # 총 배치 사이즈 (각 GPU당 64 배치)
    
    TRAIN_STEPS = 766_000  # Base 모델의 Train Steps (ELECTRA 기준)
    WARMUP_STEPS = 10000
    
    def decode_fn(sample):
        features = {
            "input_ids": tf.io.FixedLenFeature((512,), dtype=tf.int64),
            "attention_mask": tf.io.FixedLenFeature((512,), dtype=tf.int64),
            "token_type_ids": tf.io.FixedLenFeature((512,), dtype=tf.int64),
        }
        parsed_features = tf.io.parse_single_example(sample, features)
        return parsed_features
    
    NUM_EPOCHS = math.ceil(TRAIN_STEPS / (34_258_796 / GLOBAL_BATCH_SIZE))  # 예: 100,000 샘플을 256 배치로 => ~390 에포크
    TRAIN_STEPS = 802_938
    tf_dataset = tf.data.TFRecordDataset(["gs://tempbb/dataset.tfrecords"])
    tf_dataset = tf_dataset.map(decode_fn)
    tf_dataset = tf_dataset.shuffle(10_000_000).batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
    tf_dataset = tf_dataset.apply(
        tf.data.experimental.assert_cardinality(34_258_796 // GLOBAL_BATCH_SIZE)
    )
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    # 옵션을 데이터셋에 적용
    tf_dataset = tf_dataset.with_options(options)
    
    with strategy.scope():
        # 모델 설정
        config = AMCLRConfig.from_pretrained("google/electra-base-discriminator")
        tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
        special_token_ids = tokenizer.all_special_ids
        
        # 모델 인스턴스 생성
        model = AMCLR_TF(config, special_token_ids)
        
        
        optimizer, lr_schedule = create_optimizer(
            init_lr=2e-4,
            num_train_steps=TRAIN_STEPS,
            num_warmup_steps=WARMUP_STEPS,
            adam_epsilon=1e-6,
            weight_decay_rate=0.01
        )
        
        # 모델 컴파일
        model.compile(optimizer=optimizer)
        
    
    # 체크포인트 및 TensorBoard 콜백 정의
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    wandb.init(
        project="your_project_name",  # 프로젝트 이름
        name="experiment_name",  # 실험 이름
        config={  # 하이퍼파라미터 로깅
            "learning_rate": 2e-4,
            "batch_size": GLOBAL_BATCH_SIZE,
            "train_steps": TRAIN_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "epochs": NUM_EPOCHS,
        },
    )
    
    checkpoint_callback = tf_keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'model.{epoch:02d}-{loss:.2f}.keras'),
        monitor='loss',
        mode='min',
        save_freq="epoch"
    )
    
    wandb_callback = WandbMetricsLogger(log_freq="batch")
    # 모델 학습
    model.fit(
        tf_dataset,
        epochs=NUM_EPOCHS,
        callbacks=[checkpoint_callback, wandb_callback]
    )

if __name__ == "__main__":
    main()
