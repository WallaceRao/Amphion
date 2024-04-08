from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
import torch
import torch.nn.functional as F
import numpy as np
import os
import torch.nn as nn


class GPTTTS(nn.Module):
    def __init__(
        self,
        phone_vocab_size,
        target_vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        pad_token_id,
        bos_target_id,
        eos_target_id,
        bos_phone_id,
        eos_phone_id,
        use_input_embeds=False,
        emb_dim=None,
    ):
        super(GPTTTS, self).__init__()
        self.config = LlamaConfig(
            vocab_size=phone_vocab_size + target_vocab_size + 10,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            pad_token_id=pad_token_id,
            bos_token_id=bos_target_id,
            eos_token_id=eos_target_id,
        )
        self.phone_vocab_size = phone_vocab_size
        self.target_vocab_size = target_vocab_size
        self.pad_token_id = pad_token_id
        self.bos_target_id = bos_target_id
        self.eos_target_id = eos_target_id
        self.bos_phone_id = bos_phone_id
        self.eos_phone_id = eos_phone_id
        self.model = LlamaForCausalLM(self.config)

        self.use_input_embeds = use_input_embeds
        if self.use_input_embeds:
            self.emb_linear = nn.Linear(emb_dim, hidden_size)
            self.emb_linear.weight.data.normal_(mean=0.0, std=0.01)
            self.emb_linear.bias.data.zero_()

    def forward(
        self, phone_ids, phone_mask, target_ids, target_mask, input_embeds=None
    ):
        if input_embeds is not None:
            input_embeds = self.emb_linear(input_embeds)
        phone_ids, phone_mask, phone_label = self.add_phone_eos_bos_label(
            phone_ids,
            phone_mask,
            self.eos_phone_id,
            self.bos_phone_id,
            self.pad_token_id,
        )
        target_ids, target_mask, target_label = self.add_target_eos_bos_label(
            target_ids,
            target_mask,
            self.eos_target_id,
            self.bos_target_id,
            self.pad_token_id,
        )
        input_token_ids = torch.cat([phone_ids, target_ids], dim=-1)
        attention_mask = torch.cat([phone_mask, target_mask], dim=-1)
        if input_embeds is not None:
            attention_mask = torch.cat(
                [
                    torch.ones(
                        (input_embeds.shape[0], input_embeds.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                    attention_mask,
                ],
                dim=-1,
            )
        labels = torch.cat([phone_label, target_label], dim=-1)
        if input_embeds is not None:
            labels = torch.cat(
                [
                    -100
                    * torch.ones(
                        (input_embeds.shape[0], input_embeds.shape[1]),
                        dtype=labels.dtype,
                        device=labels.device,
                    ),
                    labels,
                ],
                dim=-1,
            )

        if input_embeds is not None:
            inputs_embeds = torch.cat(
                [input_embeds, self.model.model.embed_tokens(input_token_ids)], dim=1
            )
            out = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            return out

        out = self.model(
            input_token_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return out

    def add_phone_eos_bos_label(
        self, phone_ids, phone_mask, phone_eos_id, phone_bos_id, pad_token_id
    ):
        # phone_ids: [B, T]
        # phone_mask: [B, T]

        phone_ids = phone_ids + self.target_vocab_size * phone_mask

        phone_ids = phone_ids * phone_mask
        phone_ids = F.pad(phone_ids, (0, 1), value=0) + phone_eos_id * F.pad(
            1 - phone_mask, (0, 1), value=1
        )
        phone_mask = F.pad(phone_mask, (1, 0), value=1)
        phone_ids = phone_ids * phone_mask + pad_token_id * (1 - phone_mask)
        phone_ids = F.pad(phone_ids, (1, 0), value=phone_bos_id)
        phone_mask = F.pad(phone_mask, (1, 0), value=1)
        phone_label = -100 * torch.ones_like(phone_ids)
        return phone_ids, phone_mask, phone_label

    def add_target_eos_bos_label(
        self, target_ids, target_mask, target_eos_id, target_bos_id, pad_token_id
    ):
        # target_ids: [B, T]
        # target_mask: [B, T]
        target_ids = target_ids * target_mask
        target_ids = F.pad(target_ids, (0, 1), value=0) + target_eos_id * F.pad(
            1 - target_mask, (0, 1), value=1
        )
        target_mask = F.pad(target_mask, (1, 0), value=1)
        target_ids = target_ids * target_mask + pad_token_id * (1 - target_mask)
        target_ids = F.pad(target_ids, (1, 0), value=target_bos_id)
        target_mask = F.pad(target_mask, (1, 0), value=1)
        target_label = target_ids * target_mask + (-100) * (1 - target_mask)
        return target_ids, target_mask, target_label

    def sample_hf(
        self,
        phone_ids,
        prompt_ids,
        inputs_embeds=None,
        max_length=2000,
        temperature=1.0,
        top_k=100,
        top_p=0.9,
        repeat_penalty=1.0,
    ):
        if inputs_embeds is not None:
            inputs_embeds = self.emb_linear(inputs_embeds)
        phone_mask = torch.ones_like(phone_ids)
        prompt_mask = torch.ones_like(prompt_ids)
        phone_ids, _, _ = self.add_phone_eos_bos_label(
            phone_ids,
            phone_mask,
            self.eos_phone_id,
            self.bos_phone_id,
            self.pad_token_id,
        )
        prompt_ids, _, _ = self.add_target_eos_bos_label(
            prompt_ids,
            prompt_mask,
            self.eos_target_id,
            self.bos_target_id,
            self.pad_token_id,
        )
        prompt_ids = prompt_ids[:, :-1]

        input_token_ids = torch.cat([phone_ids, prompt_ids], dim=-1)

        if inputs_embeds is not None:
            inputs_embeds = torch.cat(
                [inputs_embeds, self.model.model.embed_tokens(input_token_ids)], dim=1
            )
            generated_ids = self.model.generate(
                inputs_embeds=inputs_embeds,
                do_sample=True,
                max_length=max_length,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_target_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repeat_penalty,
            )
            gen_tokens = generated_ids[:, :-1]
            return gen_tokens

        input_length = input_token_ids.shape[1]
        generated_ids = self.model.generate(
            input_token_ids,
            do_sample=True,
            max_length=max_length,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_target_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repeat_penalty,
        )

        gen_tokens = generated_ids[:, input_length:-1]

        return gen_tokens
