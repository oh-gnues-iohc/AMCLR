import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.models.electra.modeling_electra import ElectraForMaskedLM, ElectraForPreTraining, ElectraDiscriminatorPredictions
from typing import Optional
import torch.nn.functional as F
from torch.autograd import Function
import torch_xla.core.xla_model as xm
import pickle

from transformers.models.electra.modeling_electra import *
from transformers.modeling_outputs import *


def get_rank():
    """Returns the rank of the current TPU replica."""
    return xm.get_ordinal()

def get_world_size():
    """Returns the total number of TPU replicas."""
    return xm.xrt_world_size()

def all_reduce(tensor, reduce_type="sum"):
    """
    Reduces the tensor across all TPU replicas.
    Args:
        tensor (torch.Tensor): The tensor to be reduced.
        reduce_type (str): Type of reduction. Can be 'sum', 'mean', etc.
    """
    return xm.all_reduce(reduce_type, [tensor])

def all_gather_list(data, max_size=16384):
    """
    Gathers arbitrary data from all TPU replicas into a list.
    Similar to `xm.all_gather` but for arbitrary Python data.
    Args:
        data (Any): Data from the local worker to be gathered on other workers.
        max_size (int): Maximum size of serialized data in bytes.
    """
    SIZE_STORAGE_BYTES = 4  # int32 to encode the payload size

    enc = pickle.dumps(data)
    enc_size = len(enc)

    if enc_size + SIZE_STORAGE_BYTES > max_size:
        raise ValueError(
            'encoded data exceeds max_size, this can be fixed by increasing buffer size: {}'.format(enc_size))

    rank = get_rank()
    world_size = get_world_size()

    # Create a buffer for this replica
    buffer = torch.zeros(max_size, dtype=torch.uint8, device="cpu")
    size_bytes = enc_size.to_bytes(SIZE_STORAGE_BYTES, byteorder="big")
    buffer[:SIZE_STORAGE_BYTES] = torch.tensor(list(size_bytes), dtype=torch.uint8)
    buffer[SIZE_STORAGE_BYTES: enc_size + SIZE_STORAGE_BYTES] = torch.tensor(list(enc), dtype=torch.uint8)

    # Perform all_gather across TPU replicas
    gathered_buffers = xm.all_gather(buffer)

    try:
        result = []
        for i in range(world_size):
            out_buffer = gathered_buffers[i * max_size: (i + 1) * max_size]
            size = int.from_bytes(out_buffer[:SIZE_STORAGE_BYTES].tolist(), byteorder="big")
            if size > 0:
                result.append(pickle.loads(bytes(out_buffer[SIZE_STORAGE_BYTES: size + SIZE_STORAGE_BYTES].tolist())))
        return result
    except pickle.UnpicklingError:
        raise Exception(
            'Unable to unpickle data from other replicas. all_gather_list requires all '
            'replicas to enter the function together, so this error usually indicates '
            'that the replicas have fallen out of sync somehow. Replicas can fall out of '
            'sync if one of them runs out of memory, or if there are other conditions '
            'in your training script that can cause one replica to finish an epoch '
            'while other replicas are still iterating over their portions of the data.'
        )




class MixtureEncoder(ElectraEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList([ElectraLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self.grad_detach_layer = [4, 6, 8]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if i in self.grad_detach_layer:
                hidden_states = hidden_states.detach()

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class AMCLRMLM(ElectraForMaskedLM):
    def __init__(self, config, special_token_ids):
        super().__init__(config)
        self.special_token_ids = special_token_ids
        self.masking_ratio = 0.15
        self.temperature = 0.3
        self.generator_score_head = nn.Linear(config.embedding_size, 1)
        
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        generator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        generator_sequence_output = generator_hidden_states[0]  # [batch_size, seq_len, hidden_dim]


        prediction_scores = self.generator_predictions(generator_sequence_output)
        similarity = self.generator_lm_head(prediction_scores) # batch_size, seq_len, vocab_size
        scores = self.generator_score_head(prediction_scores).squeeze(-1) # [batch_size, seq_len]
                
        mask = torch.zeros_like(similarity)
        
        batch_indices = torch.arange(similarity.size(0)).unsqueeze(1)
        seq_indices = torch.arange(similarity.size(1)).unsqueeze(0)
        mask[batch_indices, seq_indices, input_ids] = torch.finfo(self.dtype).min
        mask[:, :, :100] = torch.finfo(self.dtype).min
        mask[:, :, 104:999] = torch.finfo(self.dtype).min
        masked_similarities = similarity + mask #[batch_size, seq_len, vocab_size]
        
        batch_size, seq_len, hidden_dim = generator_sequence_output.shape

        # Create special token mask
        special_tokens = torch.tensor(self.special_token_ids, device=self.device)
        is_special = (input_ids.unsqueeze(-1) == special_tokens).any(dim=-1)  # [batch_size, seq_len]
        non_special_mask = (~is_special).float()

        # Calculate number of tokens to swap
        num_maskings = max(int(seq_len * self.masking_ratio), 1)
        
        score_mask = torch.zeros_like(scores)
        
        valid_tokens = attention_mask * non_special_mask  # [batch_size, seq_len]
        invalid_tokens = ~(valid_tokens).bool() # [batch_size, seq_len]
        score_mask[invalid_tokens] = torch.finfo(self.dtype).min
        
        masked_scores = scores + score_mask # [batch_size, seq_len]
        
        y_soft = F.gumbel_softmax(masked_scores, hard=False, dim=-1) # [batch_size, seq_len]
        _, topk_indices = y_soft.topk(num_maskings, dim=1)
        
        topk_hard = torch.zeros_like(masked_scores).scatter_(-1, topk_indices, 1.0)

        top_k_socres = topk_hard - y_soft.detach() + y_soft
        
        
        token_probs = F.gumbel_softmax(masked_similarities, tau=self.temperature, hard=True, dim=-1) # [batch_size, seq_len, vocab_size]
        
        
        probs = token_probs * top_k_socres.unsqueeze(-1)
        labels = top_k_socres.detach().bool().long()
        
        return probs, generator_sequence_output, labels

class GradMultiply(Function):
    """Gradient backpropagation multiplication."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * ctx.lambd), None

def grad_multiply(x, lambd=-1):
    return GradMultiply.apply(x, lambd)


class AMCLR(ElectraForPreTraining):

    def __init__(self, config, special_token_ids, generator):
        super().__init__(config)
        self.config = config
        self.special_token_ids = special_token_ids
        self.generator = generator
        self.set_input_embeddings(self.generator.get_input_embeddings())
        
        self.electra.embeddings.position_embeddings = self.generator.electra.embeddings.position_embeddings
        self.electra.embeddings.token_type_embeddings = self.generator.electra.embeddings.token_type_embeddings
        self.electra.embeddings.LayerNorm = self.generator.electra.embeddings.LayerNorm
        
        self.cls_representation = nn.Linear(config.hidden_size, self.generator.config.hidden_size)
        self.l1 = 50
        self.l2 = 1

        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        probs, generator_sequence_output, labels = self.generator(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        probs = grad_multiply(probs, -1) # batch, seq_len, vocab_size
        
        inputs_embeds = self.get_input_embeddings()(input_ids) # batch, seq_len, dim
        
        
        replaced_embeds = torch.matmul(probs, self.get_input_embeddings().weight)
        
        mask_indices = labels == 1
        inputs_embeds[mask_indices] = replaced_embeds[mask_indices]
        
        discriminator_hidden_states = self.electra(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        logits = self.discriminator_predictions(discriminator_sequence_output)
        
        
        
        distributed_world_size = get_world_size()
        
        local_rank = xm.get_local_ordinal()
        print(local_rank, distributed_world_size)
        disc_cls_hidden_state = self.cls_representation(discriminator_sequence_output[:, 0, :])
        gen_cls_hidden_state = generator_sequence_output[:, 0, :]
        
        local_positive_idxs = list(range(disc_cls_hidden_state.size(0)))
        
        if distributed_world_size > 1:
            q_vector_to_send = torch.empty_like(disc_cls_hidden_state).cpu().copy_(disc_cls_hidden_state).detach_()
            ctx_vector_to_send = torch.empty_like(gen_cls_hidden_state).cpu().copy_(gen_cls_hidden_state).detach_()

            global_question_ctx_vectors = all_gather_list(
                [
                    q_vector_to_send,
                    ctx_vector_to_send,
                    local_positive_idxs,
                ],
                max_size=200000
            )

            global_disc_cls_hidden_state = []
            global_gen_cls_hidden_state = []

            positive_idx_per_question = []
            total_ctxs = 0

            for i, item in enumerate(global_question_ctx_vectors):
                q_vector, ctx_vectors, positive_idx = item

                if i != local_rank:
                    global_disc_cls_hidden_state.append(q_vector.to(disc_cls_hidden_state.device))
                    global_gen_cls_hidden_state.append(ctx_vectors.to(disc_cls_hidden_state.device))
                    positive_idx_per_question.extend([v + total_ctxs for v in positive_idx])
                else:
                    global_disc_cls_hidden_state.append(disc_cls_hidden_state)
                    global_gen_cls_hidden_state.append(gen_cls_hidden_state)
                    positive_idx_per_question.extend([v + total_ctxs for v in local_positive_idxs])
                total_ctxs += ctx_vectors.size(0)
            global_disc_cls_hidden_state = torch.cat(global_disc_cls_hidden_state, dim=0)
            global_gen_cls_hidden_state = torch.cat(global_gen_cls_hidden_state, dim=0)

        else:
            global_disc_cls_hidden_state = disc_cls_hidden_state
            global_gen_cls_hidden_state = gen_cls_hidden_state
            positive_idx_per_question = local_positive_idxs
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = labels[active_loss]
                disc_loss = loss_fct(active_logits, active_labels.float()) * self.l1
            else:
                disc_loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels.float()) * self.l1
            
                                    
            scores = torch.matmul(global_disc_cls_hidden_state, torch.transpose(global_gen_cls_hidden_state, 0, 1)) * self.l2

            softmax_scores = F.log_softmax(scores, dim=1)

            sims_loss = F.nll_loss(
                softmax_scores,
                torch.tensor(positive_idx_per_question).to(softmax_scores.device),
                reduction="mean",
            )
            
            with torch.no_grad():
                replaced_logits = logits[mask_indices]
                replaced_labels = labels[mask_indices].float()
                rtd_logits = (replaced_logits > 0.5).long()
                rtd_correct = (rtd_logits == replaced_labels).float()
                self.acc_rtd = rtd_correct.mean().item()
            
            
        loss = disc_loss + sims_loss
        
        
        output = (None,)
        return ((loss,) + output) if loss is not None else output

    def save_pretrained(self, dirs, state_dict, safe_serialization):
        import os
        self.electra.save_pretrained(dirs)
        self.generator.save_pretrained(os.path.join(dirs, "generator"))