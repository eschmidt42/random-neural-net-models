# -*- coding: utf-8 -*-
"""

references:
* tensordict example of transformer (encoder + decoder): https://pytorch.org/rl/tensordict/tutorials/tensordict_module.html#showcase-implementing-a-transformer-using-tensordictmodule
* sequence to sequence for translation: https://pytorch.org/tutorials/beginner/translation_transformer.html
* mingpt: https://github.com/karpathy/minGPT/tree/37baab71b9abea1b76ab957409a1cc2fbfba8a26
* cross-attention: https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
    * Q usually from decoder and K / V from encoder
"""

import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss as torch_loss
from einops import einsum, rearrange

import random_neural_net_models.text as rnnm_text

# from = enc(oder)
# to = dec(oder)


class TokensToQKV(nn.Module):
    def __init__(self, q_emb_dim: int, kv_emb_dim: int, latent_dim: int):
        super().__init__()
        self.q = nn.Linear(q_emb_dim, latent_dim)
        self.k = nn.Linear(kv_emb_dim, latent_dim)
        self.v = nn.Linear(kv_emb_dim, latent_dim)

    def forward(
        self, X_dec: torch.Tensor, X_enc: torch.Tensor
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = self.q(
            X_dec
        )  # (q0,q_emb_dim) x (q_emb_dim, latent_dim) -> (q0,latent_dim)
        K = self.k(
            X_enc
        )  # (k0,k_emb_dim) x (k_emb_dim, latent_dim) -> (k0,latent_dim)
        V = self.v(
            X_enc
        )  # (v0,v_emb_dim) x (v_emb_dim, latent_dim) -> (v0,latent_dim)
        return Q, K, V


class SplitHeads(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

    def _create_head_dims(self, X: torch.Tensor, num_part: int) -> torch.Tensor:
        # num_part is the depth of each head
        return rearrange(
            X,
            "B Nin (Nheads Npart) -> B Nheads Nin Npart",
            Nheads=self.num_heads,
            Npart=num_part,
        )

    def get_num_part(self, Q: torch.Tensor) -> int:
        latent_dim = Q.shape[-1]
        return latent_dim // self.num_heads

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_part = self.get_num_part(Q)
        Q = self._create_head_dims(Q, num_part)
        K = self._create_head_dims(K, num_part)
        V = self._create_head_dims(V, num_part)
        return Q, K, V


class Causal(nn.Module):
    # https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
    def __init__(self, n_tokens_q: int, n_tokens_kv: int):
        super().__init__()
        bias = torch.tril(
            torch.ones(n_tokens_q, n_tokens_kv, dtype=torch.float)
        )
        bias = rearrange(bias, "h w -> 1 1 h w")
        self.register_buffer("bias", bias)

    def forward(
        self, attn: torch.Tensor, Q: torch.Tensor, K: torch.Tensor
    ) -> torch.Tensor:
        # Q, K and V are of shape (B, Nheads, Nin, Npart), where Nin may be different for Q and K/V
        # attn is of the shape (B, Nheads, NinQ, NinK), where NinQ and NinK are the Nin for Q and K each
        _, _, NinQ, _ = Q.shape
        _, _, NinK, _ = K.shape
        mask = (
            self.bias[:, :, :NinQ, :NinK] == 0
        )  # TODO: why :NinQ and :NinK, or :S as originally by Karpathy?

        attn = attn.masked_fill(mask, float("-inf"))
        return attn


class NonCausal(nn.Module):
    def forward(self, attn: torch.Tensor, *args) -> torch.Tensor:
        return attn


class Attention(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        causal: bool,
        n_tokens_q: int,
        n_tokens_kv: int,
    ):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.lin = nn.Linear(latent_dim, out_dim)
        self.causal = causal

        if causal:
            self.make_causal = Causal(
                n_tokens_q=n_tokens_q, n_tokens_kv=n_tokens_kv
            )
        else:
            self.make_causal = NonCausal()

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        # Q, K and V are of shape (B, Nheads, Nin, Npart), where Nin may be different for Q and K/V
        _, _, _, num_part = Q.shape

        K_trans = rearrange(K, "B Nheads Nin Npart -> B Nheads Npart Nin")
        attn = einsum(
            Q,
            K_trans,
            "B Nheads NinQ Npart, B Nheads Npart NinK -> B Nheads NinQ NinK",
        )
        attn = attn / num_part

        attn = self.make_causal(attn, Q, K)

        attn = self.softmax(attn)

        # out = attn @ V
        out = einsum(
            attn,
            V,
            "B Nheads NinQ NinK, B Nheads NinK Npart -> B Nheads NinQ Npart",
        )
        out = rearrange(out, "B Nheads NinQ Npart -> B NinQ (Nheads Npart)")
        out = self.lin(out)
        return out


class SkipLayerNorm(nn.Module):
    # for tensors (B, Nin, Nemb) normalize across Nin and Nemb (Nemb = Nheads * Npart)
    def __init__(self, n_tokens: int, emb_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm((n_tokens, emb_dim))

    def forward(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x_0 + x_1)


def sanity_check_latent_dim_and_num_heads(latent_dim: int, num_heads: int):
    remainder = latent_dim % num_heads
    if remainder != 0:
        msg = f"{latent_dim=:_d} must be divisible by {num_heads=:_d}, but got {remainder=:_d}"
        raise ValueError(msg)


class AttentionBlock(nn.Module):
    # aka torch's MultiheadAttention module https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    def __init__(
        self,
        dec_emb_dim: int,
        dec_n_tokens: int,
        enc_emb_dim: int,
        enc_n_tokens: int,
        latent_dim: int,
        num_heads: int,
        causal: bool,
    ):
        super().__init__()
        sanity_check_latent_dim_and_num_heads(
            latent_dim=latent_dim, num_heads=num_heads
        )
        self.tokens_to_qkv = TokensToQKV(
            q_emb_dim=dec_emb_dim, kv_emb_dim=enc_emb_dim, latent_dim=latent_dim
        )
        self.split_heads = SplitHeads(num_heads=num_heads)
        self.attention = Attention(
            latent_dim=latent_dim,
            out_dim=dec_emb_dim,
            causal=causal,
            n_tokens_q=dec_n_tokens,
            n_tokens_kv=enc_n_tokens,
        )
        self.skip = SkipLayerNorm(n_tokens=dec_n_tokens, emb_dim=dec_emb_dim)

    def forward(self, X_dec: torch.Tensor, X_enc: torch.Tensor) -> torch.Tensor:
        Q, K, V = self.tokens_to_qkv(X_dec, X_enc)
        Q, K, V = self.split_heads(Q, K, V)
        attn = self.attention(Q, K, V)
        out = self.skip(X_dec, attn)
        return out


class FFN(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dropout_rate: float = 0.2):
        super().__init__()
        self.FFN = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_in),
            nn.Dropout(dropout_rate),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.FFN(X)


class EncoderTransformerBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        n_tokens: int,
        latent_dim: int,
        num_heads: int,
        causal: bool,
    ):
        super().__init__()
        self.attention_block = AttentionBlock(
            dec_emb_dim=emb_dim,
            dec_n_tokens=n_tokens,
            enc_emb_dim=emb_dim,
            enc_n_tokens=n_tokens,
            latent_dim=latent_dim,
            num_heads=num_heads,
            causal=causal,
        )
        self.FFN = FFN(dim_in=emb_dim, dim_hidden=4 * emb_dim)
        self.skip = SkipLayerNorm(n_tokens=n_tokens, emb_dim=emb_dim)

    def forward(self, X_enc: torch.Tensor) -> torch.Tensor:
        X_enc = self.attention_block(X_enc, X_enc)
        X_out = self.FFN(X_enc)
        return self.skip(X_out, X_enc)


class DecoderTransformerBlock(nn.Module):
    def __init__(
        self,
        dec_emb_dim: int,
        dec_n_tokens: int,
        enc_emb_dim: int,
        enc_n_tokens: int,
        latent_dim: int,
        num_heads: int,
        causal: bool,
    ):
        super().__init__()
        self.attention_block = AttentionBlock(
            dec_emb_dim=dec_emb_dim,
            dec_n_tokens=dec_n_tokens,
            enc_emb_dim=enc_emb_dim,
            enc_n_tokens=enc_n_tokens,
            latent_dim=latent_dim,
            num_heads=num_heads,
            causal=causal,
        )
        self.encoder_block = EncoderTransformerBlock(
            emb_dim=dec_emb_dim,
            n_tokens=dec_n_tokens,
            latent_dim=latent_dim,
            num_heads=num_heads,
            causal=causal,
        )

    def forward(self, X_dec: torch.Tensor, X_enc: torch.Tensor) -> torch.Tensor:
        X_dec = self.attention_block(X_dec, X_enc)
        X_dec = self.encoder_block(X_dec)
        return X_dec


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        enc_emb_dim: int,
        enc_n_tokens: int,
        latent_dim: int,
        num_heads: int,
        causal: bool,
        vocab_size: int,
    ):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, enc_emb_dim)
        self.wpe = nn.Embedding(enc_n_tokens, enc_emb_dim)
        self.encoder = nn.ModuleList(
            [
                EncoderTransformerBlock(
                    enc_emb_dim, enc_n_tokens, latent_dim, num_heads, causal
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, X_enc: torch.Tensor) -> torch.Tensor:
        t = X_enc.size()[-1]
        pos = torch.arange(0, t, dtype=torch.long, device=X_enc.device)
        pos = pos.unsqueeze(0)
        tok_emb = self.wte(X_enc)
        pos_emb = self.wpe(pos)

        X = tok_emb + pos_emb
        for i in range(len(self.encoder)):
            X = self.encoder[i](X)
        return X


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        dec_emb_dim: int,
        dec_n_tokens: int,
        enc_emb_dim: int,
        enc_n_tokens: int,
        latent_dim: int,
        num_heads: int,
        causal: bool,
        vocab_size: int,
    ):
        super().__init__()
        assert dec_emb_dim == enc_emb_dim
        self.wte = nn.Embedding(vocab_size, dec_emb_dim)
        self.wpe = nn.Embedding(dec_n_tokens, dec_emb_dim)
        self.decoder = nn.ModuleList(
            [
                DecoderTransformerBlock(
                    dec_emb_dim,
                    dec_n_tokens,
                    enc_emb_dim,
                    enc_n_tokens,
                    latent_dim,
                    num_heads,
                    causal,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, X_dec: torch.Tensor, X_enc: torch.Tensor) -> torch.Tensor:
        t = X_dec.size()[-1]
        pos = torch.arange(0, t, dtype=torch.long, device=X_dec.device)
        pos = pos.unsqueeze(0)
        tok_emb = self.wte(X_dec)
        pos_emb = self.wpe(pos)

        X = tok_emb + pos_emb

        for i in range(len(self.decoder)):
            X = self.decoder[i](X, X_enc)
        return X


class TransformerEncoderDecoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        enc_emb_dim: int,
        enc_n_tokens: int,
        dec_emb_dim: int,
        dec_n_tokens: int,
        latent_dim: int,
        num_heads: int,
        causal: bool,
        vocab_size: int,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_blocks,
            enc_emb_dim,
            enc_n_tokens,
            latent_dim,
            num_heads,
            causal,
            vocab_size,
        )
        self.decoder = TransformerDecoder(
            num_blocks,
            dec_emb_dim,
            dec_n_tokens,
            enc_emb_dim,
            enc_n_tokens,
            latent_dim,
            num_heads,
            causal,
            vocab_size,
        )

    def forward(self, X_dec: torch.Tensor, X_enc: torch.Tensor):
        X_enc = self.encoder(X_enc)
        X_out = self.decoder(X_dec, X_enc)
        return X_out


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        n_tokens: int,
        latent_dim: int,
        num_heads: int,
        num_blocks: int,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.transformer = TransformerEncoder(
            num_blocks=num_blocks,
            enc_emb_dim=emb_dim,
            enc_n_tokens=n_tokens,
            latent_dim=latent_dim,
            num_heads=num_heads,
            causal=True,
            vocab_size=vocab_size,
        )
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.language_model_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.transformer(X)
        X = self.layer_norm(X)
        logits = self.language_model_head(X)
        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> torch.LongTensor:
        for _ in range(max_new_tokens):
            if idx.size(1) > self.n_tokens:
                idx_cond = idx[:, -self.n_tokens :]
            else:
                idx_cond = idx

            logits = self(idx_cond)  # logits shape: (Batch, Token, Vocab)

            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)

            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class LanguageModelWithTensordict(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        n_tokens: int,
        latent_dim: int,
        num_heads: int,
        num_blocks: int,
    ):
        super().__init__()
        self.language_model = LanguageModel(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            n_tokens=n_tokens,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
        )

    def forward(
        self, input: T.Union[rnnm_text.TokenIDBlockXY, rnnm_text.TokenIDBlockX]
    ) -> torch.Tensor:
        return self.language_model(input.x)

    @torch.no_grad()
    def generate(
        self,
        input: T.Union[rnnm_text.TokenIDBlockXY, rnnm_text.TokenIDBlockX],
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> torch.LongTensor:
        return self.language_model.generate(
            input.x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )


class CrossEntropyLoss(torch_loss.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, inference: torch.Tensor, input: rnnm_text.TokenIDBlockXY
    ) -> torch.Tensor:
        return super().forward(
            inference.view(-1, inference.size(-1)), input.y.view(-1)
        )
