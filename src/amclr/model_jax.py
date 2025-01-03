import jax
import jax.numpy as jnp
from flax import linen as nn
from transformers import (
    ElectraConfig,
    FlaxElectraForMaskedLM,
    FlaxElectraForPreTraining,
)
from transformers.models.electra.modeling_flax_electra import *
from typing import Any, Dict, Optional, Tuple
from flax.training.common_utils import onehot
import optax


def sample_gumbel(shape, dtype=jnp.float32, rng=None):
    if rng is None:
        rng = jax.random.PRNGKey(0)
    U = jax.random.uniform(rng, shape=shape, minval=1e-6, maxval=1.0)
    return -jnp.log(-jnp.log(U))


def gumbel_softmax(logits, tau=1.0, hard=False, axis=-1, rng=None):
    gumbel_noise = sample_gumbel(logits.shape, dtype=logits.dtype, rng=rng)
    y = logits + gumbel_noise
    y = nn.softmax(y / tau, axis=axis)
    if hard:
        y_hard = onehot(jnp.argmax(y, axis=axis), logits.shape[-1])
        y = y + jax.lax.stop_gradient(y_hard - y)
    return y


def grad_multiply(x, lambd=-1.0):
    @jax.custom_vjp
    def _grad_mul(x):
        return x

    def fwd(x):
        return x, None

    def bwd(_, g):
        return (g * lambd,)

    _grad_mul.defvjp(fwd, bwd)
    return _grad_mul(x)


class MixtureEncoder(nn.Module):
    config: ElectraConfig
    grad_detach_layer: Tuple[int] = (4, 6, 8)
    dtype: Any = jnp.float32

    def setup(self):
        self.layers = [
            FlaxElectraLayer(self.config, name=f"layer_{i}")
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = [] if output_hidden_states else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if i in self.grad_detach_layer:
                hidden_states = jax.lax.stop_gradient(hidden_states)

            hidden_states = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                deterministic=deterministic,
            )

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        if not return_dict:
            return hidden_states

        return {"last_hidden_state": hidden_states, "hidden_states": all_hidden_states}


class FlaxMixtureElectraModel(nn.Module):
    config: ElectraConfig
    grad_detach_layer: Tuple[int] = (4, 6, 8)
    dtype: Any = jnp.float32

    def setup(self):
        self.embeddings = FlaxElectraEmbeddings(self.config, dtype=self.dtype)
        self.encoder = MixtureEncoder(
            self.config, grad_detach_layer=self.grad_detach_layer, dtype=self.dtype
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        hidden_states = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return encoder_outputs

        return encoder_outputs


class AMCLRMLMModule(nn.Module):
    config: ElectraConfig
    special_token_ids: Any
    dtype: Any = jnp.float32

    def setup(self):
        # self.electra = FlaxMixtureElectraModel(
        #     self.config, grad_detach_layer=(4, 6, 8), dtype=self.dtype
        # )
        self.electra = FlaxElectraModule(
            config=self.config, dtype=self.dtype
        )
        self.generator_predictions = FlaxElectraGeneratorPredictions(
            self.config, dtype=self.dtype
        )
        self.generator_lm_head = nn.Dense(
            self.config.vocab_size, use_bias=False, dtype=self.dtype, name="generator_lm_head"
        )
        self.generator_score_head = nn.Dense(
            1, dtype=self.dtype, name="generator_score_head"
        )
        self.masking_ratio = 0.15
        self.temperature = 0.3

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids=None,
        deterministic: bool = True,
        rngs: Dict[str, Any] = None,
    ):
        # Electra 모델의 출력을 얻습니다.
        generator_outputs = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_hidden_states=True,
            return_dict=True,
        )
        generator_sequence_output = generator_outputs["last_hidden_state"]

        # 예측 점수를 계산합니다.
        prediction_scores = self.generator_predictions(generator_sequence_output)
        similarity = self.generator_lm_head(prediction_scores)
        scores = self.generator_score_head(prediction_scores).squeeze(-1)

        batch_size, seq_len = input_ids.shape

        # 특수 토큰 마스크를 생성합니다.
        special_token_ids = jnp.array(self.special_token_ids)
        is_special = (input_ids[..., None] == special_token_ids).any(-1)
        non_special_mask = ~is_special

        # 유효한 토큰을 선택합니다.
        valid_tokens = attention_mask * non_special_mask
        invalid_tokens = ~valid_tokens.astype(bool)
        masked_scores = jnp.where(invalid_tokens, -jnp.inf, scores)

        # Gumbel Softmax를 사용하여 토큰을 선택합니다.
        if rngs is not None:
            rng_scores, rng_tokens = jax.random.split(rngs["gumbel"], 2)
        else:
            rng_scores, rng_tokens = None, None
        y_soft = gumbel_softmax(
            masked_scores, tau=1.0, hard=False, axis=-1, rng=rng_scores
        )
        num_maskings = max(int(seq_len * self.masking_ratio), 1)
        topk_indices = jnp.argsort(y_soft, axis=-1)[:, -num_maskings:]

        # 선택된 토큰의 마스크를 생성합니다.
        topk_mask = jnp.zeros_like(y_soft)
        batch_indices = jnp.arange(batch_size)[:, None]
        topk_mask = topk_mask.at[batch_indices, topk_indices].set(1.0)
        top_k_scores = jax.lax.stop_gradient(topk_mask - y_soft) + y_soft

        # Gumbel Softmax를 사용하여 대체 토큰을 선택합니다.
        token_probs = gumbel_softmax(
            similarity,
            tau=self.temperature,
            hard=True,
            axis=-1,
            rng=rng_tokens,
        )

        # 확률과 레이블을 계산합니다.
        probs = token_probs * top_k_scores[..., None]
        labels = top_k_scores.astype(jnp.int32)

        return probs, generator_sequence_output, labels



class MyModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False

    def setup(self):
        self.embeddings = FlaxElectraEmbeddings(self.config, dtype=self.dtype)
        if self.config.embedding_size != self.config.hidden_size:
            self.embeddings_project = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        self.encoder = FlaxElectraEncoder(
            self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        embeddings = None,
        head_mask: Optional[np.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        if embeddings is None:
            embeddings = self.embeddings(
                input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic
            )
        else:
            position_embeds = self.embeddings.position_embeddings(position_ids.astype("i4"))
            token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids.astype("i4"))
            embeddings = embeddings + token_type_embeddings + position_embeds

            # Layer Norm
            embeddings = self.embeddings.LayerNorm(embeddings)
            embeddings = self.embeddings.dropout(embeddings, deterministic=deterministic)
            
        if hasattr(self, "embeddings_project"):
            embeddings = self.embeddings_project(embeddings)

        return self.encoder(
            embeddings,
            attention_mask,
            head_mask=head_mask,
            deterministic=deterministic,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class AMCLRModule(nn.Module):
    config: ElectraConfig
    config2: ElectraConfig
    special_token_ids: Any
    dtype: Any = jnp.float32

    def setup(self):
        self.electra = MyModule(self.config, dtype=self.dtype)
        self.generator = AMCLRMLMModule(self.config2, self.special_token_ids)
        self.cls_representation = nn.Dense(
            self.generator.config.hidden_size,
            dtype=self.dtype,
            name="cls_representation",
        )
        self.discriminator_predictions = FlaxElectraDiscriminatorPredictions(
            self.config, dtype=self.dtype
        )
        self.l1 = 50
        self.l2 = 1

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        position_ids=None,
        deterministic: bool = True,
        rngs: Dict[str, Any] = None,
        is_training:bool = True,
    ):
        # Generator를 통해 probs, generator_sequence_output, labels를 얻습니다.
        if position_ids is None:
            # input_ids의 shape 가져오기: [batch_size, seq_length]
            print(input_ids.shape, attention_mask.shape)
            batch_size, seq_length = input_ids.shape
            # [0, 1, 2, ..., seq_length - 1] 범위를 가진 position_ids 생성
            position_ids = jnp.broadcast_to(
                jnp.arange(seq_length, dtype=jnp.int32), (batch_size, seq_length)
            )
            
        if token_type_ids is None:
            # 일반적으로 single sequence의 경우, 모든 token_type_ids는 0
            token_type_ids = jnp.zeros_like(input_ids)
        probs, generator_sequence_output, labels = self.generator(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            rngs=rngs,
        )

        # Gradient reversal을 적용합니다.
        probs = grad_multiply(probs, -1.0)

        # 입력 임베딩을 얻습니다.
        inputs_embeds = self.electra.embeddings.word_embeddings(input_ids.astype("i4"))

        # 대체된 임베딩을 계산합니다.
        generator_embeddings = self.electra.embeddings.word_embeddings.embedding
        replaced_embeds = jnp.einsum("...vh,...v->...h", generator_embeddings, probs)

        # 마스크된 위치에 임베딩을 대체합니다.
        mask_indices = labels == 1
        inputs_embeds = jnp.where(
            mask_indices[..., None], replaced_embeds, inputs_embeds
        )

        # Discriminator를 통해 출력을 얻습니다.
        discriminator_outputs = self.electra(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            embeddings = inputs_embeds,
            deterministic=deterministic,
            output_hidden_states=True,
            return_dict=True,
        )
        discriminator_sequence_output = discriminator_outputs["last_hidden_state"]

        logits = self.discriminator_predictions(discriminator_sequence_output)

        # Loss 계산을 위한 준비
        disc_cls_hidden_state = self.cls_representation(
            discriminator_sequence_output[:, 0, :]
        )
        gen_cls_hidden_state = generator_sequence_output[:, 0, :]
        
        if is_training:

            disc_cls_hidden_state = jax.lax.all_gather(disc_cls_hidden_state, 'dp')
            gen_cls_hidden_state = jax.lax.all_gather(gen_cls_hidden_state, 'dp')

            print(disc_cls_hidden_state)

            # Stop gradients from flowing back to representations from other devices
            def stop_gradient_except_own(x):
                own_device_index = jax.lax.axis_index('dp')
                num_devices = x.shape[0]
                device_indices = jnp.arange(num_devices)
                mask = device_indices[:, None] != own_device_index  # [num_devices, 1]
                mask = jnp.expand_dims(mask, axis=-1)  # [num_devices, 1, 1]
                return jnp.where(mask, jax.lax.stop_gradient(x), x)

            disc_cls_hidden_state = stop_gradient_except_own(disc_cls_hidden_state)
            gen_cls_hidden_state = stop_gradient_except_own(gen_cls_hidden_state)

        # Contrastive loss computation
        batch_size = disc_cls_hidden_state.shape[1]
        global_batch_size = disc_cls_hidden_state.shape[0] * batch_size
        disc_cls_hidden_state = disc_cls_hidden_state.reshape(global_batch_size, -1)
        gen_cls_hidden_state = gen_cls_hidden_state.reshape(global_batch_size, -1)
        scores = jnp.matmul(
            disc_cls_hidden_state, gen_cls_hidden_state.T
        )  # [batch_size, batch_size]
        positive_idx = jnp.arange(scores.shape[0])

        # 손실 함수 계산
        loss_fct = optax.sigmoid_binary_cross_entropy
        
        disc_labels = jnp.zeros_like(input_ids, dtype=self.dtype)
        disc_labels = jnp.where(
            mask_indices, 1.0, disc_labels
        )
        disc_loss = loss_fct(logits, disc_labels.astype(self.dtype)) * self.l1
        
        disc_loss = disc_loss * attention_mask
        disc_loss = jnp.sum(disc_loss) / jnp.sum(attention_mask)
        
        # softmax_scores = nn.log_softmax(scores, axis=1)
        # sims_loss = -softmax_scores[
        #     jnp.arange(scores.shape[0]), positive_idx
        # ].mean() * self.l2
        sims_labels = jax.nn.one_hot(positive_idx, num_classes=scores.shape[1])
        sims_loss = optax.softmax_cross_entropy(
            logits=scores, labels=sims_labels
        ).mean() * self.l2

        total_loss = disc_loss + sims_loss

        return total_loss