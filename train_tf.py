import tensorflow as tf
import numpy as np
from wandb.integration.keras import WandbMetricsLogger
from wandb.integration.keras import WandbModelCheckpoint
from transformers import AdamWeightDecay
from transformers.models.electra.modeling_tf_electra import *
from transformers.models.electra import ElectraConfig
from transformers.modeling_tf_utils import *
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
    
    TRAIN_STEPS = 766000  # Base 모델의 Train Steps (ELECTRA 기준)
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
    
    tf_dataset = tf.data.TFRecordDataset(["gs://tempbb/dataset.tfrecords"])
    tf_dataset = tf_dataset.map(decode_fn)
    tf_dataset = tf_dataset.shuffle(10_000_000).batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
    
    with strategy.scope():
        # 모델 설정
        config = AMCLRConfig.from_pretrained("google/electra-base-discriminator")
        tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
        special_token_ids = tokenizer.all_special_ids
        
        # 모델 인스턴스 생성
        model = AMCLR_TF(config, special_token_ids)
        
        learning_rate_schedule = WarmUpLinearDecay(
            initial_learning_rate=2e-4,
            warmup_steps=WARMUP_STEPS,
            total_steps=TRAIN_STEPS
        )
        
        # AdamW 옵티마이저 정의
        optimizer = AdamWeightDecay(
            learning_rate=learning_rate_schedule,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6
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
    
    checkpoint_callback = WandbModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'model.{epoch:02d}-{loss:.2f}.keras'),
        monitor='loss',
        mode='min',
        save_freq="epoch"
    )
    
    wandb_callback = WandbMetricsLogger()
    # 모델 학습
    model.fit(
        tf_dataset,
        epochs=NUM_EPOCHS,
        callbacks=[checkpoint_callback, wandb_callback]
    )

if __name__ == "__main__":
    main()
