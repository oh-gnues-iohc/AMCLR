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
        
        
        # Apply Gumbel Softmax to masking_scores
        masking_scores_soft = F.gumbel_softmax(masking_scores, tau=1.0, hard=False, dim=1)  # Shape: [batch, seq_len, 1]
        
        _, topk_indices = torch.topk(masking_scores_soft.squeeze(-1), k=num_maskings, dim=1)  # Shape: [batch, num_maskings]
        # Create hard masking scores based on topk_indices
        
        one_hot_topk = F.one_hot(topk_indices, num_classes=masking_scores.size(1)).type_as(masking_scores)
        masking_scores_hard = one_hot_topk.sum(dim=1).unsqueeze(-1) # batch, seq_len, 1
        
        # masking_scores_hard = torch.zeros_like(masking_scores).scatter_(-1, topk_indices, 1.0)
        # 원-핫 인코딩
        masking_scores_hard = (masking_scores_hard - masking_scores_soft.detach()) + masking_scores_soft
        
        # Convert to float for discriminator labels
        
        # Expand dimensions to match prediction_scores_hard
        # masking_scores_hard = masking_scores_hard.unsqueeze(-1)  # Shape: [batch, seq_len, 1]
        
        # Compute the final probabilities
        probs = masking_scores_hard * prediction_scores_hard  # Shape: [batch, seq_len, vocab_size]
        disc_labels = masking_scores_hard.detach().long().squeeze(-1)  # Shape: [batch, seq_len]
        
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
        # inputs_embeds = torch.where(mask_indices.unsqueeze(-1), replaced_embeds, inputs_embeds)
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
            for i in range(distributed_world_size):
                if i == local_rank:
                    all_q_vectors[i] = disc_cls_hidden_state
                    all_c_vectors[i] = gen_cls_hidden_state
                else:
                    all_q_vectors[i] = all_q_vectors[i]
                    all_c_vectors[i] = all_c_vectors[i]

            global_disc_cls_hidden_state = all_q_vectors.view(-1, disc_cls_hidden_state.size(-1))  # Shape: [world_size * word_size * batch_size, dim]
            global_gen_cls_hidden_state = all_c_vectors.view(-1, gen_cls_hidden_state.size(-1))    # Shape: [world_size * word_size * batch_size, dim]
            
        else:
            global_disc_cls_hidden_state = disc_cls_hidden_state
            global_gen_cls_hidden_state = gen_cls_hidden_state
            
        
            
        positive_idx_per_question = torch.arange(
        global_disc_cls_hidden_state.size(0), device=disc_cls_hidden_state.device
        )

        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
            active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
            active_labels = labels[active_loss]
            disc_loss = loss_fct(active_logits, active_labels.float()) * self.l1
                                    
            scores = torch.matmul(global_disc_cls_hidden_state, torch.transpose(global_gen_cls_hidden_state, 0, 1))

            sims_loss = F.cross_entropy(scores, positive_idx_per_question) * self.l2
            
            
        loss = disc_loss + sims_loss
        output = (global_disc_cls_hidden_state.size(0),)
        return ((loss,) + output) if loss is not None else output
