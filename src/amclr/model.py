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



def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, collections.OrderedDict):
            # OrderedDict has attributes that needs to be preserved
            od = collections.OrderedDict(
                (key, _apply(value)) for key, value in x.items()
            )
            od.__dict__ = x.__dict__
            return od
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


def move_to_cpu(sample):
    def _move_to_cpu(tensor):
        # PyTorch has poor support for half tensors (float16) on CPU.
        # Move any such tensors to float32.
        if tensor.dtype in {torch.bfloat16, torch.float16}:
            tensor = tensor.to(dtype=torch.float32)
        return tensor.cpu()

    return apply_to_sample(_move_to_cpu, sample)


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

def all_gather_list(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.

    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable and any CUDA tensors will be moved
    to CPU and returned on CPU as well.

    Args:
        data (Any): data from the local worker to be gathered on other workers
        group: group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """

    if group is None:
        group = get_global_group()
    rank = get_rank(group=group)
    world_size = get_world_size(group=group)

    buffer_size = max_size * world_size
    if (
        not hasattr(all_gather_list, "_buffer")
        or all_gather_list._buffer.numel() < buffer_size
    ):
        all_gather_list._buffer = torch.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size)
    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    data = move_to_cpu(data)
    enc = pickle.dumps(data)
    enc_size = len(enc)
    header_size = 4  # size of header that contains the length of the encoded data
    size = header_size + enc_size
    if size > max_size:
        raise ValueError(
            "encoded data size ({}) exceeds max_size ({})".format(size, max_size)
        )

    header = struct.pack(">I", enc_size)
    cpu_buffer[:size] = torch.ByteTensor(list(header + enc))
    start = rank * max_size
    buffer[start : start + size].copy_(cpu_buffer[:size])

    all_reduce(buffer, group=group)

    buffer = buffer.cpu()
    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size : (i + 1) * max_size]
            (enc_size,) = struct.unpack(">I", bytes(out_buffer[:header_size].tolist()))
            if enc_size > 0:
                result.append(
                    pickle.loads(
                        bytes(out_buffer[header_size : header_size + enc_size].tolist())
                    )
                )
        return result
    except pickle.UnpicklingError:
        raise Exception(
            "Unable to unpickle data from other workers. all_gather_list requires all "
            "workers to enter the function together, so this error usually indicates "
            "that the workers have fallen out of sync somehow. Workers can fall out of "
            "sync if one of them runs out of memory, or if there are other conditions "
            "in your training script that can cause one worker to finish an epoch "
            "while other workers are still iterating over their portions of the data. "
            "Try rerunning with --ddp-backend=legacy_ddp and see if that helps."
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
        
        
        
        distributed_world_size = _get_world_size()
        
        local_rank = xm.get_ordinal()
        disc_cls_hidden_state = self.cls_representation(discriminator_sequence_output[:, 0, :])
        gen_cls_hidden_state = generator_sequence_output[:, 0, :]
        
        positive_idx_per_question = torch.tensor(list(range(distributed_world_size))).to(disc_cls_hidden_state.device)
        
        if distributed_world_size > 1:

            global_disc_cls_hidden_state = []
            global_gen_cls_hidden_state = []

            total_ctxs = 0
            
            all_q_vectors = all_gather(disc_cls_hidden_state.detach())
            all_c_vectors = all_gather(gen_cls_hidden_state.detach())
            # all_idxs = all_gather(local_positive_idxs.detach())

            for i, item in enumerate(zip(all_q_vectors, all_c_vectors)):
                q_vector, ctx_vectors = item
                if i != local_rank:
                    global_disc_cls_hidden_state.append(q_vector.to(disc_cls_hidden_state.device))
                    global_gen_cls_hidden_state.append(ctx_vectors.to(disc_cls_hidden_state.device))
                
                else:
                    global_disc_cls_hidden_state.append(disc_cls_hidden_state)
                    global_gen_cls_hidden_state.append(gen_cls_hidden_state)
                
                total_ctxs += ctx_vectors.size(0)
            global_disc_cls_hidden_state = torch.cat(global_disc_cls_hidden_state, dim=0)
            global_gen_cls_hidden_state = torch.cat(global_gen_cls_hidden_state, dim=0)
            

        else:
            global_disc_cls_hidden_state = disc_cls_hidden_state
            global_gen_cls_hidden_state = gen_cls_hidden_state
            
        print(positive_idx_per_question.shape, local_rank)
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
            
            print(disc_loss, local_rank)
                                    
            scores = torch.matmul(global_disc_cls_hidden_state, torch.transpose(global_gen_cls_hidden_state, 0, 1))

            softmax_scores = F.log_softmax(scores, dim=1)

            sims_loss = F.nll_loss(
                softmax_scores,
                positive_idx_per_question,
                reduction="mean",
            )  * self.l2
            print(sims_loss, local_rank)
            
        loss = disc_loss + sims_loss
        print(loss, local_rank)
        
        output = (None,)
        return ((loss,) + output) if loss is not None else output

    def save_pretrained(self, dirs, state_dict, safe_serialization):
        import os
        self.electra.save_pretrained(dirs)
        self.generator.save_pretrained(os.path.join(dirs, "generator"))