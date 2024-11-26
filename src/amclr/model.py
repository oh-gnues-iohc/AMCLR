import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.models.electra.modeling_electra import ElectraForMaskedLM, ElectraForPreTraining, ElectraDiscriminatorPredictions
from typing import *
import torch.nn.functional as F
from torch.autograd import Function
import torch_xla.core.xla_model as xm
import pickle
import collections
import struct

from transformers.models.electra.modeling_electra import *
from transformers.modeling_outputs import *


def new_groups(grouped_ranks: List[List[int]]):
    return ("tpu", grouped_ranks)

def get_global_world_size():
    return xm.xrt_world_size()

def get_global_rank():
    return xm.get_ordinal()
    
def _find_my_group_index(grouped_ranks):
    my_rank = get_global_rank()
    for i, group in enumerate(grouped_ranks):
        if my_rank in group:
            return i
    raise RuntimeError

def _find_my_group(grouped_ranks):
    index = _find_my_group_index(grouped_ranks)
    return grouped_ranks[index]

def get_world_size(group):
    assert group[0] == "tpu"
    my_group = _find_my_group(group[1])
    return len(my_group)

def get_global_group():
    return new_groups([list(range(get_global_world_size()))])

def _get_rank():
    """Returns the rank of the current TPU replica."""
    return xm.get_ordinal()

def _get_world_size():
    """Returns the total number of TPU replicas."""
    return xm.xrt_world_size()


def get_rank(group):
    assert group[0] == "tpu"
    my_group = _find_my_group(group[1])
    return my_group.index(get_global_rank())

def all_reduce(tensor, group, op="sum"):
    assert isinstance(group, tuple) and group[0] == "tpu"
    tensor = [tensor]  # wrap in a list to make xm.all_reduce in-place
    return xm.all_reduce(op, tensor, groups=group[1])[0]


def all_gather(tensor, group=None, return_tensor=False):
    if group is None:
        group = get_global_group()
    """Perform an all-gather operation."""
    result = xm.all_gather(tensor, groups=group[1])
    world_size = get_world_size(group=group)
    result = result.view(world_size, *tensor.size())
    if return_tensor:
        return result
    else:
        return [result[i] for i in range(world_size)]



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
        score_mask = torch.where(invalid_tokens, torch.tensor(torch.finfo(self.dtype).min, device=score_mask.device), score_mask)
        # score_mask[invalid_tokens] = torch.finfo(self.dtype).min
        
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
        inputs_embeds = torch.where(mask_indices.unsqueeze(-1), replaced_embeds, inputs_embeds)
        # inputs_embeds[mask_indices] = replaced_embeds[mask_indices]
        
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
        
        group = get_global_group()
        distributed_world_size = get_world_size(group)
        local_rank = get_rank(group)
        disc_cls_hidden_state = self.cls_representation(discriminator_sequence_output[:, 0, :])
        gen_cls_hidden_state = generator_sequence_output[:, 0, :]
        
        # xm.mark_step()
        
        if distributed_world_size > 1:

            global_disc_cls_hidden_state = []
            global_gen_cls_hidden_state = []
            
            
            
            all_q_vectors = all_gather(disc_cls_hidden_state.detach())
            all_c_vectors = all_gather(gen_cls_hidden_state.detach())

            for i, item in enumerate(zip(all_q_vectors, all_c_vectors)):
                q_vector, ctx_vectors = item
                if i != local_rank:
                    global_disc_cls_hidden_state.append(q_vector.to(disc_cls_hidden_state.device))
                    global_gen_cls_hidden_state.append(ctx_vectors.to(disc_cls_hidden_state.device))
                else:
                    global_disc_cls_hidden_state.append(disc_cls_hidden_state)
                    global_gen_cls_hidden_state.append(gen_cls_hidden_state)
                
            global_disc_cls_hidden_state = torch.cat(global_disc_cls_hidden_state, dim=0)
            global_gen_cls_hidden_state = torch.cat(global_gen_cls_hidden_state, dim=0)
            
        else:
            global_disc_cls_hidden_state = disc_cls_hidden_state
            global_gen_cls_hidden_state = gen_cls_hidden_state
            
            
        print(global_gen_cls_hidden_state.shape, logits.shape, local_rank, distributed_world_size)
            
        positive_idx_per_question = torch.arange(
        global_disc_cls_hidden_state.size(0), device=disc_cls_hidden_state.device
        )

        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            disc_label = torch.where(
                mask_indices,
                torch.full_like(mask_indices, 1.0, dtype=torch.float, device=mask_indices.device),
                torch.full_like(mask_indices, 0.0, dtype=torch.float, device=mask_indices.device),
            )
            disc_loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), disc_label.float())
            
            masked_loss = disc_loss * attention_mask
            disc_loss = masked_loss.sum() / attention_mask.sum() * self.l1
            
                                    
            scores = torch.matmul(global_disc_cls_hidden_state, torch.transpose(global_gen_cls_hidden_state, 0, 1))

            softmax_scores = F.log_softmax(scores, dim=1)

            sims_loss = F.nll_loss(
                softmax_scores,
                positive_idx_per_question,
                reduction="mean",
            )  * self.l2
            
        loss = disc_loss + sims_loss
        output = (None,)
        return loss

    def save_pretrained(self, dirs, state_dict, safe_serialization):
        import os
        self.electra.save_pretrained(dirs)
        self.generator.save_pretrained(os.path.join(dirs, "generator"))