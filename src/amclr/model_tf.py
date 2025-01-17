import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.models.electra.modeling_electra import ElectraForMaskedLM, ElectraForPreTraining, ElectraDiscriminatorPredictions
from typing import *
import torch.nn.functional as F
from torch.autograd import Function
import torch_xla.core.xla_model as xm
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

    def __init__(self, config, special_token_ids):
        super().__init__(config)
        self.config = config
        self.special_token_ids = special_token_ids
        
        self.head = nn.Linear(config.hidden_size, config.vocab_size)
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
        
        original_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        logits = self.head(original_hidden_states.detach())
        
        mask = torch.zeros_like(logits)  # [batch_size, seq_len, vocab_size]

        one_hot_ids = F.one_hot(input_ids, num_classes=logits.size(-1)).bool()

        mask = torch.where(
            one_hot_ids, 
            torch.tensor(torch.finfo(self.dtype).min, device=mask.device), 
            mask
        )

        vocab_range = torch.arange(logits.size(-1), device=mask.device)
        vocab_range = vocab_range.view(1, 1, -1)  # [1, 1, vocab_size] for broadcasting

        range_mask_1 = (vocab_range < 100)
        range_mask_2 = (vocab_range >= 104) & (vocab_range < 999)
        total_range_mask = range_mask_1 | range_mask_2

        mask = torch.where(
            total_range_mask,
            torch.tensor(torch.finfo(self.dtype).min, device=mask.device),
            mask
        )

        logits = logits + mask  # [batch_size, seq_len, vocab_size]
        probs = F.softmax(logits, dim=-1)

        
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
        positive_idx = list(range(disc_cls_hidden_state.size(0)))
        if distributed_world_size > 1:
            q_vector_to_send = torch.empty_like(disc_cls_hidden_state).copy_(disc_cls_hidden_state).detach_()
            ctx_vector_to_send = torch.empty_like(gen_cls_hidden_state).copy_(gen_cls_hidden_state).detach_()

            global_disc_cls_hidden_state = []
            global_gen_cls_hidden_state = []
            positive_idx_per_question = []
            
            all_q_vectors = all_gather(q_vector_to_send, return_tensor=False) # word_size, batch_size, dim
            all_c_vectors = all_gather(ctx_vector_to_send, return_tensor=False) # word_size, batch_size, dim
            
            total_ctxs = 0
            
            # all_q_vectors = all_q_vectors.to(disc_cls_hidden_state.device)
            # all_c_vectors = all_c_vectors.to(gen_cls_hidden_state.device)

            # Create a tensor index for the local rank
            for i in range(distributed_world_size):
                if i == local_rank:
                    global_disc_cls_hidden_state.append(disc_cls_hidden_state)
                    global_gen_cls_hidden_state.append(gen_cls_hidden_state)
                    positive_idx_per_question.extend([v + total_ctxs for v in positive_idx])
                else:
                    global_disc_cls_hidden_state.append(all_q_vectors[i].to(disc_cls_hidden_state.device))
                    global_gen_cls_hidden_state.append(all_c_vectors[i].to(disc_cls_hidden_state.device))
                    positive_idx_per_question.extend([v + total_ctxs for v in positive_idx])
                    
                total_ctxs += all_c_vectors[i].size(0)
                
            global_disc_cls_hidden_state = torch.cat(global_disc_cls_hidden_state, dim=0)
            global_gen_cls_hidden_state = torch.cat(global_gen_cls_hidden_state, dim=0)
            
        else:
            global_disc_cls_hidden_state = disc_cls_hidden_state
            global_gen_cls_hidden_state = gen_cls_hidden_state
            positive_idx_per_question = position_ids
            
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            disc_loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels.float())
            
            masked_loss = disc_loss * attention_mask
            disc_loss = (masked_loss.sum() / attention_mask.sum()) * self.l1
                                    
            scores = torch.matmul(global_disc_cls_hidden_state, torch.transpose(global_gen_cls_hidden_state, 0, 1))
            softmax_scores = F.log_softmax(scores, dim=1)

            sims_loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
            ) * self.l2
            
            
        loss = disc_loss + sims_loss
        output = (global_disc_cls_hidden_state.size(0),)
        # return loss, disc_loss, sims_loss
        return (loss, ) + output
