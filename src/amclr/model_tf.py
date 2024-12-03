from transformers.models.electra.modeling_tf_electra import *
from transformers.models.electra import ElectraConfig
from transformers.modeling_tf_utils import *
import tensorflow as tf

class AMCLRConfig(ElectraConfig):
    def __init__(
        self,
        vocab_size=30522,
        embedding_size=128,
        hidden_size=256,
        num_hidden_layers=12,
        num_attention_heads=4,
        intermediate_size=1024,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        summary_type="first",
        summary_use_proj=True,
        summary_activation="gelu",
        summary_last_dropout=0.1,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.g_hidden_size = hidden_size // 3
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.g_num_attention_heads = num_attention_heads // 3
        self.intermediate_size = intermediate_size
        self.g_intermediate_size = intermediate_size // 3
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_last_dropout = summary_last_dropout
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
    


@tf.custom_gradient
def grad_multiply(x, lambd=-1.0):
    def grad(dy):
        return dy * lambd
    return x, grad

def gumbel_softmax(logits, temperature, hard=False):
    """Sample from Gumbel Softmax distribution."""
    gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1) + 1e-20) + 1e-20)
    y = logits + gumbel_noise
    y = tf.nn.softmax(y / temperature)
    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, axis=-1, keepdims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y

@keras_serializable
class TFElectraMainLayerNonEmbeddings(TFElectraMainLayer):
    config_class = ElectraConfig

    def __init__(self, config, embeddings, **kwargs):
        super().__init__(config, **kwargs)

        self.config = config
        self.is_decoder = config.is_decoder

        self.embeddings = embeddings

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = keras.layers.Dense(config.hidden_size, name="embeddings_project")

        self.encoder = TFElectraEncoder(config, name="encoder")

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, "embeddings_project", None) is not None:
            with tf.name_scope(self.embeddings_project.name):
                self.embeddings_project.build([None, None, self.config.embedding_size])
        
        

class AMCLRProject(TFElectraGeneratorPredictions):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.dense = keras.layers.Dense(config.g_hidden_size, name="dense")
        self.config = config

    def call(self, generator_hidden_states, training=False):
        hidden_states = self.dense(generator_hidden_states)

        return hidden_states
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # Build the dense layer with appropriate input shape
                self.dense.build([None, None, self.config.hidden_size])

@dataclass
class Myoutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (`tf.Tensor` of shape `(n,)`, *optional*, where n is the number of non-masked labels, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: tf.Tensor

class AMCLR_TF(TFElectraForPreTraining):
    def __init__(self, config, special_token_ids, **kwargs):
        super().__init__(config, **kwargs)
        self.special_token_ids = special_token_ids
        self.min_value = tf.float32.min

        self.electra = TFElectraMainLayer(config, name="electra")
        self.discriminator_predictions = TFElectraDiscriminatorPredictions(config, name="discriminator_predictions")
        self.discriminator_project = AMCLRProject(config, name="disc_projection")
        from copy import deepcopy
        g_config = deepcopy(config)
        g_config.hidden_size = config.g_hidden_size
        g_config.num_attention_heads = config.g_num_attention_heads
        g_config.intermediate_size = config.g_intermediate_size
        
        self.electr_for_generator = TFElectraMainLayerNonEmbeddings(g_config, self.electra.embeddings, name="generator_electra")
        self.generator_predictions = TFElectraGeneratorPredictions(g_config, name="generator_predictions")
        self.score_predictions = TFElectraDiscriminatorPredictions(g_config, name="score_predictions")
        
        self.l1 = 50
        self.l2 = 1

        if isinstance(config.hidden_act, str):
            self.activation = get_tf_activation(config.hidden_act)
        else:
            self.activation = config.hidden_act

        self.generator_lm_head = TFElectraMaskedLMHead(g_config, self.electra.embeddings, name="generator_lm_head")

    def get_lm_head(self):
        return self.generator_lm_head

    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.generator_lm_head.name
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "electra", None) is not None:
            with tf.name_scope(self.electra.name):
                self.electra.build(None)
        if getattr(self, "discriminator_predictions", None) is not None:
            with tf.name_scope(self.discriminator_predictions.name):
                self.discriminator_predictions.build(None)
        if getattr(self, "discriminator_project", None) is not None:
            with tf.name_scope(self.discriminator_project.name):
                self.discriminator_project.build(None)
        if getattr(self, "electr_for_generator", None) is not None:
            with tf.name_scope(self.electr_for_generator.name):
                self.electr_for_generator.build(None)
        if getattr(self, "generator_predictions", None) is not None:
            with tf.name_scope(self.generator_predictions.name):
                self.generator_predictions.build(None)
        if getattr(self, "score_predictions", None) is not None:
            with tf.name_scope(self.score_predictions.name):
                self.score_predictions.build(None)
        if getattr(self, "generator_lm_head", None) is not None:
            with tf.name_scope(self.generator_lm_head.name):
                self.generator_lm_head.build(None)

    @unpack_inputs
    def call(
        self,
        input_ids = None,
        attention_mask= None,
        token_type_ids= None,
        position_ids= None,
        head_mask= None,
        inputs_embeds= None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ):
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """ 
        generator_hidden_states = self.electr_for_generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        special_token_ids_tensor = tf.constant(self.special_token_ids, dtype=tf.int32)
        
        generator_sequence_output = generator_hidden_states[0]
        prediction_scores = self.generator_predictions(generator_sequence_output, training=training)
        prediction_scores = self.generator_lm_head(prediction_scores, training=training)
        

        mask_special_tokens = tf.scatter_nd(
            indices=tf.expand_dims(special_token_ids_tensor, axis=1),
            updates=tf.ones_like(special_token_ids_tensor, dtype=tf.bool),
            shape=[self.config.vocab_size]
        )
        mask_special_tokens_expanded = tf.reshape(mask_special_tokens, [1, 1, -1])
        mask_input_ids = tf.one_hot(input_ids, depth=self.config.vocab_size, dtype=tf.float32)
        mask_input_ids = tf.cast(mask_input_ids, tf.bool)

        total_mask = mask_special_tokens_expanded | mask_input_ids

        prediction_scores = tf.where(total_mask, self.min_value, prediction_scores)
        
        prediction_scores_hard = gumbel_softmax(prediction_scores, 0.3, hard=True)
        
        masking_scores = self.score_predictions(generator_sequence_output, training=training)
        special_tokens_mask = tf.reduce_any(
            tf.equal(
                tf.expand_dims(input_ids, axis=-1),
                special_token_ids_tensor
            ),
            axis=-1
        )
        
        # special_tokens_mask_expanded = tf.expand_dims(special_tokens_mask, axis=-1)

        masking_scores = tf.where(special_tokens_mask, self.min_value, masking_scores) # batch, seq_len, 1
        # masking_scores = tf.squeeze(masking_scores, axis=-1) # batch, seq_len
        
        num_maskings = 77  # Number of tokens to mask
        _, topk_indices = tf.math.top_k(masking_scores, k=num_maskings)
        # Create topk_hard tensor
        
        masking_scores_soft = gumbel_softmax(masking_scores, 1, hard=False)
        masking_scores_hard = tf.reduce_sum(
            tf.one_hot(topk_indices, depth=tf.shape(masking_scores)[1], dtype=masking_scores.dtype),
            axis=1
        )
        masking_scores_hard = tf.stop_gradient(masking_scores_hard - masking_scores_soft) + masking_scores_soft
        disc_labels = tf.cast(masking_scores_hard, tf.float32)
        
        masking_scores_hard = tf.expand_dims(masking_scores_hard, axis=-1)
        probs = masking_scores_hard * prediction_scores_hard
        probs = grad_multiply(probs, -1.0)
        
        embedding_weight = self.get_input_embeddings().weight
        replaced_embeds = tf.matmul(probs, embedding_weight)
        
        inputs_embeds = tf.gather(params=embedding_weight, indices=input_ids)
        mask_indices = tf.equal(disc_labels, 1)
        mask_indices_expanded = tf.expand_dims(mask_indices, axis=-1)
        inputs_embeds = tf.where(mask_indices_expanded, replaced_embeds, inputs_embeds)
        
        
        discriminator_hidden_states = self.electra(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]
        logits = self.discriminator_predictions(discriminator_sequence_output)
        
         
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    
        unmasked_loss = loss_fn(disc_labels, logits)
        weights = tf.cast(attention_mask, tf.float32)
        masked_loss = unmasked_loss * tf.expand_dims(weights, axis=-1)
        disc_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(weights)
            
        
        disc_cls = discriminator_sequence_output[:, 0]
        disc_cls = self.discriminator_project(disc_cls)
        gen_cls = generator_sequence_output[:, 0]
        
        replica_ctx = tf.distribute.get_replica_context()
        if replica_ctx is not None:
            disc_cls_all = replica_ctx.all_gather(disc_cls, axis=0)
            gen_cls_all = replica_ctx.all_gather(gen_cls, axis=0)
            
            local_batch_size = tf.shape(disc_cls)[0]
            global_batch_size = tf.shape(disc_cls_all)[0]
            
            replica_id = replica_ctx.replica_id_in_sync_group
            num_replicas = replica_ctx.num_replicas_in_sync
            
            start_idx = replica_id * local_batch_size
            end_idx = start_idx + local_batch_size
            
            indices = tf.range(global_batch_size)
            is_local = (indices >= start_idx) & (indices < end_idx)
            
            disc_cls_all = tf.where(
                tf.expand_dims(is_local, -1),
                disc_cls_all,  # Keep local entries as is
                tf.stop_gradient(disc_cls_all)  # Stop gradients on non-local entries
            )
            gen_cls_all = tf.where(
                tf.expand_dims(is_local, -1),
                gen_cls_all,  # Keep local entries as is
                tf.stop_gradient(gen_cls_all)  # Stop gradients on non-local entries
            )
        else:
            disc_cls_all = disc_cls
            gen_cls_all = gen_cls
            global_batch_size = tf.shape(disc_cls_all)[0]
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        similarity = tf.matmul(disc_cls_all, gen_cls_all, transpose_b=True)
        labels = tf.range(global_batch_size)
        sims_loss = loss_fn(labels, similarity)
        sims_loss = tf.reduce_mean(sims_loss)
        
        loss = disc_loss * self.l1 + sims_loss * self.l2
        return TFMaskedLMOutput(
            loss=tf.reshape(loss, (1,)),
            logits=prediction_scores,
            hidden_states=generator_hidden_states.hidden_states,
            attentions=generator_hidden_states.attentions,
        )
