import os
import random
from datasets.arrow_dataset import embed_table_storage
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import BertTokenizer, BertModel, BertConfig, ElectraConfig, Trainer, TrainingArguments
from transformers.models.electra.modeling_electra import ElectraForMaskedLM, ElectraForPreTraining, ElectraDiscriminatorPredictions
from datasets import load_dataset, load_from_disk
from typing import Optional
import socket
import time
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.autograd import Function

def pad_and_create_attention_mask(x, tokenizer):
    max_length = 128
    x = tokenizer(
        x["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length)
    x['labels'] = x['input_ids']
    return x

def prepare_dataset(tokenizer):
    dataset = load_dataset('gsgoncalves/roberta_pretrain', cache_dir="./hf_temp",split='train').shuffle(83245280).select(range(10_000_000))
    dataset = dataset.map(lambda x: pad_and_create_attention_mask(x, tokenizer), num_proc=10)
    return dataset

def get_mask_with_probability(input_ids, special_token_ids, prob=0.25):
    device = input_ids.device

    special_mask = ~input_ids.unsqueeze(-1).eq(torch.tensor(special_token_ids, device=device)).any(-1)

    random_tensor = torch.rand_like(input_ids, dtype=torch.float)
    mask = (random_tensor < prob) & special_mask

    masked_count = mask.sum(dim=1)

    unmasked_batches = (masked_count == 0)
    if unmasked_batches.any():
        valid_positions = special_mask[unmasked_batches].float()
        random_selection = torch.rand_like(valid_positions)
        random_selection = random_selection * valid_positions
        random_mask = random_selection == random_selection.max(dim=1, keepdim=True)[0]
        mask[unmasked_batches] |= random_mask
    
    return mask



def get_swip_positions(input_ids, special_token_ids=[0], masking_ratio=0.002, min_swips=1):
    """
    Select swip_positions from input_ids, excluding special tokens.
    
    Args:
        input_ids (torch.Tensor): 1D Tensor of shape [seq_length], e.g., input_ids[batch].
        special_token_ids (list): List of token IDs to exclude from selection (default: [0]).
        masking_ratio (float): Ratio of tokens to select for swapping (default: 0.2%).
        min_swips (int): Minimum number of positions to select (default: 1).
    
    Returns:
        list: List of selected positions (indices) for swapping.
    """
    # Ensure input_ids is a 1D tensor
    if input_ids.dim() != 1:
        raise ValueError("input_ids should be a 1D tensor")
    
    seq_length = input_ids.size(0)
    
    # Create a mask for non-special tokens
    # Expand special_token_ids to match input_ids dimensions for comparison
    special_tokens_tensor = torch.tensor(special_token_ids, device=input_ids.device).unsqueeze(0)  # Shape: [1, num_special_tokens]
    input_ids_expanded = input_ids.unsqueeze(1)  # Shape: [seq_length, 1]
    
    # Compare input_ids with special_token_ids and create a mask
    is_special = (input_ids_expanded == special_tokens_tensor).any(dim=1)  # Shape: [seq_length]
    non_special_mask = ~is_special  # Invert mask to get non-special tokens
    
    # Get indices of non-special tokens
    non_special_indices = torch.nonzero(non_special_mask).squeeze()
    
    # Handle cases where only one non-special token exists
    if non_special_indices.numel() == 0:
        return []  # No valid positions to swap
    elif non_special_indices.dim() == 0:
        non_special_indices = non_special_indices.unsqueeze(0)
    
    num_non_special = non_special_indices.size(0)
    
    # Calculate number of swip_positions to select
    num_swips = max(int(seq_length * masking_ratio), min_swips)
    num_swips = min(num_swips, num_non_special)  # Ensure we don't exceed available positions
    
    # Randomly select indices without replacement
    selected_indices = torch.randperm(num_non_special)[:num_swips]
    swip_positions = non_special_indices[selected_indices].tolist()
    
    return swip_positions


def get_shuffle_indices(input_ids, shuffle_input_ids):
    """
    shuffle_input_ids의 각 요소가 input_ids의 어디에 위치하는지를 나타내는 인덱스 텐서를 반환합니다.

    Args:
        input_ids (torch.Tensor): 원본 토큰 ID 텐서, 크기 (batch, seq_len)
        shuffle_input_ids (torch.Tensor): 셔플된 토큰 ID 텐서, 크기 (batch, seq_len)

    Returns:
        torch.Tensor: 각 요소의 원본 위치 인덱스, 크기 (batch, seq_len)
    """
    batch_size, seq_len = input_ids.size()

    # input_ids를 (batch, 1, seq_len)로 확장
    input_ids_exp = input_ids.unsqueeze(1).expand(-1, seq_len, -1)

    # shuffle_input_ids를 (batch, seq_len, 1)로 확장
    shuffle_input_ids_exp = shuffle_input_ids.unsqueeze(2).expand(-1, -1, seq_len)

    # 두 텐서 비교하여 일치하는 위치 찾기
    mask = (shuffle_input_ids_exp == input_ids_exp)  # 크기: (batch, seq_len, seq_len)

    # 각 위치에서 True인 인덱스 찾기
    indices = torch.argmax(mask.int(), dim=2)  # 크기: (batch, seq_len)

    return indices

class SimsMLM(ElectraForMaskedLM):
    def __init__(self, config, special_token_ids):
        super().__init__(config)
        self.special_token_ids = special_token_ids
        self.masking_ratio = 0.15
        self.temperature = 0.3
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

        # ELECTRA Encoding
        generator_hidden_states = self.electra(
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
        generator_sequence_output = generator_hidden_states[0]  # [batch_size, seq_len, hidden_dim]


        prediction_scores = self.generator_predictions(generator_sequence_output)
        similarity = self.generator_lm_head(prediction_scores) # batch_size, seq_len, vocab_size
        
        mask = torch.zeros_like(similarity)
        
        batch_indices = torch.arange(similarity.size(0)).unsqueeze(1)
        seq_indices = torch.arange(similarity.size(1)).unsqueeze(0)
        mask[batch_indices, seq_indices, input_ids] = torch.finfo(self.dtype).min
        mask[:, :, :100] = torch.finfo(self.dtype).min
        mask[:, :, 104:999] = torch.finfo(self.dtype).min
        masked_similarities = similarity + mask #[batch_size, seq_len, vocab_size]
        
        
        return masked_similarities, generator_sequence_output, labels

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


class SimsElectra(ElectraForPreTraining):

    def __init__(self, config, special_token_ids, generator):
        super().__init__(config)
        self.config = config
        self.special_token_ids = special_token_ids
        self.generator = generator
        self.set_input_embeddings(self.generator.get_input_embeddings())
        
        self.electra.embeddings.position_embeddings = self.generator.electra.embeddings.position_embeddings
        self.electra.embeddings.token_type_embeddings = self.generator.electra.embeddings.token_type_embeddings


        self.post_init()
        self.acc_rtd=0
        self.acc_mlm=0
        
        # Generator와 Discriminator의 임베딩이 동일한지 확인
        if self.get_input_embeddings() is self.generator.get_input_embeddings():
            print("Generator와 Discriminator가 동일한 임베딩을 공유하고 있습니다.")
        else:
            print("Generator와 Discriminator가 다른 임베딩을 사용하고 있습니다.")

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

        mask_positions = get_mask_with_probability(input_ids, self.special_token_ids, 0.15)
        
        labels[labels == 0] = -100
        mask_ids = input_ids.clone()
        mask_ids[mask_positions] = 103  # [MASK] 토큰 ID

        # 생성자 전방 전달
        probs, generator_sequence_output, labels = self.generator(
            mask_ids,
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
        
        s_probs = F.softmax(probs[mask_positions], dim=-1)
        mask_ids = torch.argmax(s_probs, dim=-1)
        new_ids = input_ids.clone()
        new_ids[mask_positions] = mask_ids
        discriminator_hidden_states = self.electra(
            new_ids,
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
        
        
        
        labels = input_ids.clone()
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                disc_labels = torch.zeros_like(labels)
                disc_labels[mask_positions] = 1
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = disc_labels[active_loss]
                disc_loss = loss_fct(active_logits, active_labels.float()) * 50
            else:
                disc_loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels.float()) * 50
            
            
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            gen_loss = loss_fct(probs[mask_positions].view(-1, self.config.vocab_size), labels[mask_positions].view(-1))
            
            
            with torch.no_grad():
                replaced_logits = logits[disc_labels == 1]
                replaced_labels = disc_labels[disc_labels == 1].float()
                rtd_logits = (replaced_logits > 0.5).long()
                rtd_correct = (rtd_logits == replaced_labels).float()
                self.acc_rtd = rtd_correct.mean().item()
            
            
        loss = disc_loss + gen_loss
        # print(disc_loss, gen_loss, sims_loss)
        
        
        output = (None,)
        return ((loss,) + output) if loss is not None else output

    def save_pretrained(self, dirs, state_dict, safe_serialization):
        import os
        self.electra.save_pretrained(dirs)
        self.generator.save_pretrained(os.path.join(dirs, "/generator/"))
