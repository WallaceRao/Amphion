from multiprocessing.sharedctypes import Value
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
import torch
import torch.nn.functional as F
import numpy as np
import os
import torch.nn as nn


class T2SLlama(nn.Module):
    def __init__(
        self,
        phone_vocab_size=1024,
        target_vocab_size=2048,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=12,
        num_attention_heads=16,
        pad_token_id=3072,
        bos_target_id=3073,
        eos_target_id=3074,
        bos_phone_id=3075,
        eos_phone_id=3076,
        use_lang_emb=False,
        cfg=None,
    ):
        super().__init__()

        phone_vocab_size = (
            cfg.phone_vocab_size
            if cfg is not None and hasattr(cfg, "phone_vocab_size")
            else phone_vocab_size
        )
        target_vocab_size = (
            cfg.target_vocab_size
            if cfg is not None and hasattr(cfg, "target_vocab_size")
            else target_vocab_size
        )
        hidden_size = (
            cfg.hidden_size
            if cfg is not None and hasattr(cfg, "hidden_size")
            else hidden_size
        )
        intermediate_size = (
            cfg.intermediate_size
            if cfg is not None and hasattr(cfg, "intermediate_size")
            else intermediate_size
        )
        num_hidden_layers = (
            cfg.num_hidden_layers
            if cfg is not None and hasattr(cfg, "num_hidden_layers")
            else num_hidden_layers
        )
        num_attention_heads = (
            cfg.num_attention_heads
            if cfg is not None and hasattr(cfg, "num_attention_heads")
            else num_attention_heads
        )
        pad_token_id = (
            cfg.pad_token_id
            if cfg is not None and hasattr(cfg, "pad_token_id")
            else pad_token_id
        )
        bos_target_id = (
            cfg.bos_target_id
            if cfg is not None and hasattr(cfg, "bos_target_id")
            else bos_target_id
        )
        eos_target_id = (
            cfg.eos_target_id
            if cfg is not None and hasattr(cfg, "eos_target_id")
            else eos_target_id
        )
        bos_phone_id = (
            cfg.bos_phone_id
            if cfg is not None and hasattr(cfg, "bos_phone_id")
            else bos_phone_id
        )
        eos_phone_id = (
            cfg.eos_phone_id
            if cfg is not None and hasattr(cfg, "eos_phone_id")
            else eos_phone_id
        )
        use_lang_emb = (
            cfg.use_lang_emb
            if cfg is not None and hasattr(cfg, "use_lang_emb")
            else use_lang_emb
        )

        self.config = LlamaConfig(
            vocab_size=phone_vocab_size + target_vocab_size + 20,
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
        self.use_lang_emb = use_lang_emb
        self.model = LlamaForCausalLM(self.config)

        if self.use_lang_emb:
            self.lang_emb = nn.Embedding(25, hidden_size, padding_idx=0)
            torch.nn.init.normal_(self.lang_emb, mean=0.0, std=0.02)

    def forward(
        self,
        phone_ids,
        phone_mask,
        target_ids,
        target_mask,
        lang_id=None,
    ):
        phone_ids, phone_mask, phone_label, lang_mask = self.add_phone_eos_bos_label(
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

        labels = torch.cat([phone_label, target_label], dim=-1)

        # lang_id: (B,); lang_mask: (B, T)
        if self.use_lang_emb:
            lang_embedding = self.lang_emb(lang_id).unsequeeze(1)  # (B, d) -> (B, 1, d)
            lang_embedding = lang_embedding * torch.cat(
                [lang_mask, torch.zeros_like(target_mask)], dim=-1
            ).unsequeeze(
                -1
            )  # (B, T, d)
            input_token_embedding = self.model.model.embed_tokens(
                input_token_ids
            )  # (B, T, d)
            inputs_embeds = input_token_embedding + lang_embedding

            out = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )

        else:
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

        # add 0 in the left
        lang_mask = F.pad(phone_mask, (1, 0), value=0)
        # add 0 in the right
        lang_mask = F.pad(lang_mask, (0, 1), value=0)

        phone_ids = phone_ids + self.target_vocab_size * phone_mask

        phone_ids = phone_ids * phone_mask
        phone_ids = F.pad(phone_ids, (0, 1), value=0) + phone_eos_id * F.pad(
            1 - phone_mask, (0, 1), value=1
        )  # make pad token eos token, add eos token at the end
        phone_mask = F.pad(phone_mask, (1, 0), value=1)  # add eos mask
        phone_ids = phone_ids * phone_mask + pad_token_id * (
            1 - phone_mask
        )  # restore pad token ids
        phone_ids = F.pad(phone_ids, (1, 0), value=phone_bos_id)  # add bos token
        phone_mask = F.pad(phone_mask, (1, 0), value=1)  # add bos mask
        phone_label = -100 * torch.ones_like(
            phone_ids
        )  # loss for entire phone is not computed (passed to llama)
        return phone_ids, phone_mask, phone_label, lang_mask

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
        target_label = target_ids * target_mask + (-100) * (
            1 - target_mask
        )  # loss for target is computed on unmasked tokens
        return target_ids, target_mask, target_label

    def sample_hf(
        self,
        phone_ids,  # the phones of prompt and target should be concatenated together
        prompt_ids,
        max_length=2000,
        temperature=1.0,
        top_k=100,
        top_p=0.9,
        repeat_penalty=1.0,
        lang_ids=None,
    ):
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
        prompt_ids = prompt_ids[:, :-1]  # remove end token. Make it continue mode

        input_token_ids = torch.cat([phone_ids, prompt_ids], dim=-1)
        input_length = input_token_ids.shape[1]

        if lang_ids != None and self.use_lang_emb:
            lang_ids = F.pad(lang_ids, (1, 0), value=0).pad(lang_ids, (0, 1), value=0)
            assert lang_ids.shape == phone_ids.shape
            input_token_embedding = self.model.model.embed_tokens(
                input_token_ids
            )  # (B, T, d)
            # lang_ids: [1,1,1,1,1,1,2,2,2,2] which means ['en','en','en','en','en','en','zh','zh','zh','zh']
            lang_mask = torch.ones_like(phone_ids)
            lang_mask[:, 0] = 0
            lang_mask[:, -1] = 0
            lang_embedding = self.lang_emb(lang_ids) * torch.cat(
                [lang_mask, torch.zeros_like(prompt_ids)], dim=-1
            ).unsqueeze(-1)
            inputs_embeds = input_token_embedding + lang_embedding

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

        else:
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


# if __name__ == "__main__":
#     model = T2SLlama()

#     phone_ids = torch.LongTensor([[1,2,3,4,5,0],
#                                   [1,2,3,4,5,6]])
#     phone_mask = torch.LongTensor([[1,1,1,0,0,0],
#                                    [1,1,1,0,0,0]])
#     target_ids = torch.LongTensor([765, 234, 123, 234, 123,599]).expand(2,-1)
#     target_mask = torch.LongTensor([1,1,1,1,0,0]).expand(2,-1)

#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

#     for i in range(15):
#         optimizer.zero_grad()
#         out = model(
#             phone_ids=phone_ids,
#             phone_mask=phone_mask,
#             target_ids=target_ids,
#             target_mask=target_mask,
#         )
#         loss = out.loss

#         loss.backward()

#         optimizer.step()

#         print(f"iter={i}, {loss}.")

#     phone_ids = torch.LongTensor([1,2,3]).reshape(1,-1)
#     target_ids = torch.LongTensor([765, 234]).reshape(1,-1)
#     sampled = model.sample_hf(phone_ids, target_ids)
#     print(sampled)
