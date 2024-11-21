import torch
import torch.nn as nn
from transformers.models.electra.modeling_electra import ElectraForMaskedLM, ElectraForPreTraining, ElectraDiscriminatorPredictions
from typing import Optional
import torch.nn.functional as F
from torch.autograd import Function


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


from transformers.models.electra.modeling_electra import *
from transformers.modeling_outputs import *


class CustomEncoder(ElectraEncoder):
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

class SimsMLM(ElectraForMaskedLM):
    def __init__(self, config, special_token_ids):
        super().__init__(config)
        self.special_token_ids = special_token_ids
        self.masking_ratio = 0.15
        self.temperature = 0.3
        self.electra.encoder = CustomEncoder(config)
        self.mw = nn.Linear(config.embedding_size, 1)
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
            output_hidden_states=True,
            return_dict=return_dict,
        )
        generator_sequence_output = generator_hidden_states[0]  # [batch_size, seq_len, hidden_dim]
        all_hidden_states = generator_hidden_states.hidden_states
        mask_positions = input_ids == 103
        all_mixture_logits = []
        all_vocab = []
        for layer, h in enumerate(all_hidden_states):
            if layer in [4,6,8]:
                h= self.generator_predictions(h[mask_positions])
                all_vocab.append(self.generator_lm_head(h).unsqueeze(-1))#batch, seq, dim -> batch, seq, dim
                all_mixture_logits.append(self.mw(h)) # batch, 1


                
        small_gen_x_masks = torch.cat(all_vocab, dim=-1)
        layer_mix_logits = torch.cat(all_mixture_logits, dim=-1)
        layer_mix_probs = F.softmax(layer_mix_logits.float(), dim=-1).to(layer_mix_logits)
        similarity = torch.matmul(small_gen_x_masks.detach(), layer_mix_probs.unsqueeze(-1)).squeeze(-1)
        # similarity = self.generator_lm_head(prediction_scores) # batch_size, seq_len, vocab_size
        
        mask = torch.zeros_like(similarity)
        
        mask[torch.arange(similarity.size(0)), input_ids[mask_positions]] = torch.finfo(self.dtype).min
        mask[:, :100] = torch.finfo(self.dtype).min
        mask[:, 104:999] = torch.finfo(self.dtype).min
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
        sample_probs = F.gumbel_softmax(probs.float(), 0.3, hard=True, dim=-1).to(probs)
        sampled_input = sample_probs.argmax(dim=-1)
        sample_probs = grad_multiply(sample_probs, lambd=-1)

        inputs_embeds = self.get_input_embeddings()(input_ids) # batch, seq_len, dim
        
        replaced_embeds = torch.matmul(sample_probs, self.get_input_embeddings().weight)

        
        inputs_embeds[mask_positions] = replaced_embeds
        # s_probs = F.softmax(probs[mask_positions], dim=-1)
        # mask_ids = torch.argmax(s_probs, dim=-1)
        # new_ids = input_ids.clone()
        # new_ids[mask_positions] = mask_ids
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
            gen_loss = loss_fct(probs.view(-1, self.config.vocab_size), labels[mask_positions].view(-1))
            
            
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