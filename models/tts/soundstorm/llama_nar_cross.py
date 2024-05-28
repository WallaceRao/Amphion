from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
import torch
import torch.nn.functional as F
import numpy as np
import os
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import math

from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaAttention,
    repeat_kv,
    apply_rotary_pos_emb,
    Cache,
)
from torchmetrics.classification import MulticlassAccuracy
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast


# sinusoidal positional encoding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] * 1.0
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LlamaAdaptiveRMSNorm(nn.Module):
    def __init__(self, hidden_size=1024, eps=1e-6, dim_cond=1024):
        super().__init__()
        self.to_weight = nn.Linear(dim_cond, hidden_size)
        nn.init.zeros_(self.to_weight.weight)
        nn.init.ones_(self.to_weight.bias)
        self.variance_epsilon = eps
        self._is_hf_initialized = True  # disable automatic init

    def forward(self, hidden_states, cond_embedding):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        weight = self.to_weight(cond_embedding)
        if len(weight.shape) == 2:
            weight = weight.unsqueeze(1)

        return (weight * hidden_states).to(input_dtype)


class LlamaCrossAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        _, kv_len, _ = cond_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(cond_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(cond_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(cond_states)
            value_states = self.v_proj(cond_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(
            1, 2
        )  # (B, n, T, d//n)
        key_states = key_states.view(
            bsz, kv_len, self.num_key_value_heads, self.head_dim
        ).transpose(
            1, 2
        )  # (B, n, T', d//n)
        value_states = value_states.view(
            bsz, kv_len, self.num_key_value_heads, self.head_dim
        ).transpose(
            1, 2
        )  # (B, n, T', d//n)

        # cos, sin = self.rotary_emb(value_states, position_ids)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # if past_key_value is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(
            self.head_dim
        )  # (B, n, T, T')

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaNARDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        """Override to adaptive layer norm"""
        super().__init__(config, layer_idx)  # init attention, mlp, etc.
        self.layer_idx = layer_idx
        self.input_layernorm = LlamaAdaptiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dim_cond=config.hidden_size
        )
        self.post_attention_layernorm = LlamaAdaptiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dim_cond=config.hidden_size
        )
        if (layer_idx + 1) % 3 == 0:  # 2, 5, 8, 11, 14
            self.cross_attention_layernorm = LlamaAdaptiveRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, dim_cond=config.hidden_size
            )
            self.cross_attn = LlamaCrossAttention(config, layer_idx)

    # add `cond` in forward function
    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_embedding: torch.Tensor,
        cond_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(
            hidden_states, cond_embedding=cond_embedding
        )

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Cross Attention
        if (self.layer_idx + 1) % 3 == 0:
            residual = hidden_states
            # prenorm
            hidden_states = self.cross_attention_layernorm(
                hidden_states, cond_embedding=cond_embedding
            )
            # cross attention
            hidden_states, _, _ = self.cross_attn(
                hidden_states=hidden_states,
                cond_states=cond_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            hidden_states, cond_embedding=cond_embedding
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class DiffLlamaCross(LlamaModel):
    def __init__(
        self,
        hidden_size=1024,
        num_heads=16,
        num_layers=16,
        dropout=0.1,
        ffn_dropout=0.1,
        attention_dropout=0.0,
        zero_init_cross_attn=True,
        config=LlamaConfig(vocab_size=0),
    ):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [
                LlamaNARDecoderLayer(
                    LlamaConfig(
                        hidden_size=hidden_size,
                        num_attention_heads=num_heads,
                        max_position_embeddings=4096,
                        intermediate_size=hidden_size * 4,
                    ),
                    layer_idx=i,
                )
                for i in range(num_layers)
            ]
        )

        self.norm = LlamaAdaptiveRMSNorm(hidden_size, dim_cond=hidden_size)

        self.diff_step_embedding = SinusoidalPosEmb(hidden_size)
        self.diff_step_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        for layer in self.layers:
            layer.input_layernorm = LlamaAdaptiveRMSNorm(
                hidden_size, dim_cond=hidden_size
            )
            layer.post_attention_layernorm = LlamaAdaptiveRMSNorm(
                hidden_size, dim_cond=hidden_size
            )

        self.post_init()

        if zero_init_cross_attn:
            for i in range(num_layers):
                layer_idx = i
                if (layer_idx + 1) % 3 == 0:
                    self.layers[
                        layer_idx
                    ].cross_attention_layernorm = LlamaAdaptiveRMSNorm(
                        hidden_size, dim_cond=hidden_size
                    )
                    self.layers[layer_idx].cross_attn.q_proj.weight.data.zero_()
                    self.layers[layer_idx].cross_attn.k_proj.weight.data.zero_()
                    self.layers[layer_idx].cross_attn.v_proj.weight.data.zero_()
                    self.layers[layer_idx].cross_attn.o_proj.weight.data.zero_()

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create noncausal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None

        def _expand_mask(
            mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
        ):
            """
            Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
            """
            bsz, src_len = mask.size()
            tgt_len = tgt_len if tgt_len is not None else src_len

            expanded_mask = (
                mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
            )

            inverted_mask = 1.0 - expanded_mask

            return inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(dtype).min
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        x,
        diffusion_step,
        cond,
        x_mask,
        phone_embedding: Optional[torch.LongTensor] = None,
        input_ids: torch.LongTensor = None,  # [num_quant, B, T]
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # retrieve some shape info
        batch_size, seq_length, _ = x.shape

        # condtion mlp
        cond_embedding = self.cond_mlp(cond)  # (B, T, C)

        # diffusion step embedding
        diffusion_step = self.diff_step_embedding(diffusion_step).to(x.device)
        diffusion_step = self.diff_step_mlp(diffusion_step)  # (B, C)

        x = x + cond_embedding

        inputs_embeds = x
        attention_mask = x_mask

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                raise NotImplementedError

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cond_embedding=diffusion_step,
                    cond_states=phone_embedding,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states, cond_embedding=diffusion_step)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        # if not return_dict:
        #     return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        # return BaseModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attns,
        # )
        return hidden_states


# if __name__ == "__main__":
#     llama_model = DiffLlamaCross()
#     print(llama_model)
