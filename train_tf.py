import tensorflow as tf, tf_keras
from transformers.optimization_tf import create_optimizer
from src.amclr.model_tf import AMCLR_TF, AMCLRConfig
from transformers import AutoTokenizer
import os

        
def main():
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
    
    tf_dataset = tf.data.TFRecordDataset(["gs://tempbb/dataset.tfrecords"])
    tf_dataset = tf_dataset.map(decode_fn)
    tf_dataset = tf_dataset.shuffle(10_000_000).batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
    tf_dataset = tf_dataset.repeat()  # 무한 반복
    
    tf_dataset = tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # Prefetch 추가

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
        
    
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    
    checkpoint_callback = tf_keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'model.step-{batch:06d}.keras'),
        monitor='loss',
        mode='min',
        save_freq=766000
    )
    
    model.fit(
        tf_dataset,
        epochs=1,
        steps_per_epoch=TRAIN_STEPS,
        callbacks=[checkpoint_callback]
    )

if __name__ == "__main__":
    main()
        