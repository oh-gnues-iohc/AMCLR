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
        masking_ratio = 0.15
        self.temperature = 0.3
        self.generator_score_head = nn.Linear(config.embedding_size, 1)
        self.num_maskings = max(int(512 * masking_ratio), 1)
        self.min_value = torch.finfo(self.dtype).min
        
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
        
        special_token_ids_tensor = torch.tensor(
            self.special_token_ids, dtype=torch.long, device=input_ids.device
        )
        

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
        masking_scores = self.generator_score_head(prediction_scores)
                
        mask_special_tokens = torch.zeros(
            self.config.vocab_size, dtype=torch.bool, device=input_ids.device
        )
        mask_special_tokens[special_token_ids_tensor] = True
        mask_special_tokens_expanded = mask_special_tokens.view(1, 1, -1)  # Shape: [1, 1, vocab_size]
        
        mask_input_ids = F.one_hot(input_ids, num_classes=self.config.vocab_size).bool()  # Shape: [batch, seq_len, vocab_size]
        
        # Combine masks
        total_mask = mask_special_tokens_expanded | mask_input_ids  # Broadcasting to [batch, seq_len, vocab_size]
        
        # Apply the combined mask to prediction_scores
        similarity = torch.where(total_mask, self.min_value, similarity)
        
        # Apply Gumbel Softmax
        prediction_scores_hard = F.gumbel_softmax(similarity, tau=0.3, hard=True)
        
        # Compute masking scores
        
        special_tokens_mask = (input_ids.unsqueeze(-1) == special_token_ids_tensor).any(dim=-1)  # Shape: [batch, seq_len]
        
        # Apply the special tokens mask to masking_scores
        masking_scores = torch.where(
            special_tokens_mask.unsqueeze(-1),
            torch.tensor(self.min_value, dtype=masking_scores.dtype, device=masking_scores.device),
            masking_scores
        )  # Shape: [batch, seq_len, 1]
        
        # Define the number of tokens to mask
        num_maskings = 77
        
        # Squeeze the last dimension for topk operation
        masking_scores_squeezed = masking_scores.squeeze(-1)  # Shape: [batch, seq_len]
        
        # Select top-k masking scores
        _, topk_indices = torch.topk(masking_scores_squeezed, k=num_maskings, dim=1)  # Shape: [batch, num_maskings]
        
        # Apply Gumbel Softmax to masking_scores
        masking_scores_soft = F.gumbel_softmax(masking_scores, tau=1.0, hard=False, dim=1)  # Shape: [batch, seq_len, 1]
        
        # Create hard masking scores based on topk_indices
        one_hot_topk = F.one_hot(topk_indices, num_classes=masking_scores.size(1)).type_as(masking_scores)  # Shape: [batch, num_maskings, seq_len]
        masking_scores_hard = one_hot_topk.sum(dim=1)  # Shape: [batch, seq_len]
        
        # Detach the gradient for the hard masking scores and add the soft scores
        masking_scores_hard = (masking_scores_hard - masking_scores_soft.squeeze(-1)).detach() + masking_scores_soft.squeeze(-1)  # Shape: [batch, seq_len]
        
        # Convert to float for discriminator labels
        disc_labels = masking_scores_hard.detach().float()  # Shape: [batch, seq_len]
        
        # Expand dimensions to match prediction_scores_hard
        masking_scores_hard = masking_scores_hard.unsqueeze(-1)  # Shape: [batch, seq_len, 1]
        
        # Compute the final probabilities
        probs = masking_scores_hard * prediction_scores_hard  # Shape: [batch, seq_len, vocab_size]
        
        return probs, generator_sequence_output, disc_labels

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


def unwrap_model(model):
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model

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
            
            all_q_vectors = all_gather(disc_cls_hidden_state.detach(), return_tensor=True) # word_size, batch_size, dim
            all_c_vectors = all_gather(gen_cls_hidden_state.detach(), return_tensor=True) # word_size, batch_size, dim
            
            all_q_vectors = all_q_vectors.to(disc_cls_hidden_state.device)
            all_c_vectors = all_c_vectors.to(gen_cls_hidden_state.device)

            # Create a tensor index for the local rank
            local_rank_tensor = torch.tensor(local_rank, device=all_q_vectors.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            temp_mask = torch.arange(all_q_vectors.size(0), device=all_q_vectors.device).view(-1, 1, 1, 1) == local_rank_tensor
            # Replace the local_rank slice with the original tensors
            all_q_vectors = torch.where(
                temp_mask,
                disc_cls_hidden_state.unsqueeze(0),
                all_q_vectors
            )

            all_c_vectors = torch.where(
                temp_mask,
                gen_cls_hidden_state.unsqueeze(0),
                all_c_vectors
            )

            # Concatenate along the first dimension (world_size)
            global_disc_cls_hidden_state = all_q_vectors.view(-1, disc_cls_hidden_state.size(-1))  # Shape: [world_size * word_size * batch_size, dim]
            global_gen_cls_hidden_state = all_c_vectors.view(-1, gen_cls_hidden_state.size(-1))    # Shape: [world_size * word_size * batch_size, dim]
            
        else:
            global_disc_cls_hidden_state = disc_cls_hidden_state
            global_gen_cls_hidden_state = gen_cls_hidden_state
            
            
        # logger.info(global_gen_cls_hidden_state.shape, logits.shape, local_rank, distributed_world_size)
        # print(global_gen_cls_hidden_state.shape, logits.shape, local_rank, distributed_world_size)
            
        positive_idx_per_question = torch.arange(
        global_disc_cls_hidden_state.size(0), device=disc_cls_hidden_state.device
        )

        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            disc_loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels)
            
            masked_loss = disc_loss * attention_mask
            disc_loss = (masked_loss.sum() / attention_mask.sum()) * self.l1
                                    
            scores = torch.matmul(global_disc_cls_hidden_state, torch.transpose(global_gen_cls_hidden_state, 0, 1))

            softmax_scores = F.log_softmax(scores, dim=1)

            sims_loss = F.nll_loss(
                softmax_scores,
                positive_idx_per_question,
                reduction="mean",
            )  * self.l2
            
        loss = disc_loss + sims_loss
        output = (None,)
        return ((loss,) + output) if loss is not None else output
        # return loss
    # def save_pretrained(
    #     self,
    #     save_directory: Union[str, os.PathLike],
    #     is_main_process: bool = True,
    #     state_dict: Optional[dict] = None,
    #     save_function: Callable = torch.save,
    #     push_to_hub: bool = False,
    #     max_shard_size: Union[int, str] = "5GB",
    #     safe_serialization: bool = True,
    #     variant: Optional[str] = None,
    #     token: Optional[Union[str, bool]] = None,
    #     save_peft_format: bool = True,
    #     **kwargs,
    # ):
    #     import os
    #     unwrap_model(self.electra).save_pretrained(
    #         save_directory,
    #         is_main_process=is_main_process,
    #         state_dict=state_dict,
    #         save_function=save_function,
    #         push_to_hub=push_to_hub,
    #         max_shard_size=max_shard_size,
    #         safe_serialization=safe_serialization,
    #         variant=variant,
    #         token=token,
    #         save_peft_format=save_peft_format,
    #         **kwargs,  # 추가 인자 전달
    #     )
    #     unwrap_model(self.generator).save_pretrained(
    #         os.path.join(save_directory, "generator"),
    #         is_main_process=is_main_process,
    #         state_dict=state_dict,
    #         save_function=save_function,
    #         push_to_hub=push_to_hub,
    #         max_shard_size=max_shard_size,
    #         safe_serialization=safe_serialization,
    #         variant=variant,
    #         token=token,
    #         save_peft_format=save_peft_format,
    #         **kwargs,  # 추가 인자 전달
    #     )
