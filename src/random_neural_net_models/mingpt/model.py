# -*- coding: utf-8 -*-
"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from einops import einsum, rearrange
from torch.nn import functional as F

import random_neural_net_models.mingpt.configs as configs
import random_neural_net_models.utils as utils

logger = utils.get_logger("mingpt.model")


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    C = math.sqrt(2.0 / math.pi)

    def __init__(
        self,
        f0: float = 0.044715,
        f1: float = 3.0,
        f2: float = 0.5,
        f3: float = 1.0,
    ):
        super().__init__()
        self.f0 = f0
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x + self.f0 * torch.pow(x, self.f1)
        return self.f2 * x * (self.f3 + torch.tanh(self.C * a))


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config: configs.ModelConfig):
        super().__init__()

        if config.n_embd is None or config.n_head is None:
            raise ValueError(
                f"{config.n_embd=} and {config.n_head=} cannot be None here"
            )
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        if config.block_size is None:
            raise ValueError(f"{config.block_size=} cannot be None here")

        bias = torch.tril(torch.ones(config.block_size, config.block_size))
        bias = rearrange(bias, "h w -> 1 1 h w")
        self.register_buffer("bias", bias, persistent=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def _move_head_forward(self, x: torch.Tensor, emb_dims: int) -> torch.Tensor:
        return rearrange(
            x,
            "B S (Nh Ne) -> B Nh S Ne",
            Nh=self.n_head,
            Ne=emb_dims // self.n_head,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch size, sequence length, embedding dimensionality (n_embd)
        _, S, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = self._move_head_forward(k, C)
        q = self._move_head_forward(q, C)
        v = self._move_head_forward(v, C)

        # causal self-attention; Self-attend: (B, nh, S, hs) x (B, nh, hs, S) -> (B, nh, S, S)
        f = 1.0 / math.sqrt(k.size(-1))
        kT = rearrange(k, "B Nh S Ne -> B Nh Ne S")
        att = f * einsum(q, kT, "B Nh Sq Ne, B Nh Ne Sk -> B Nh Sq Sk")  # q @ kT * f

        if not isinstance(self.bias, torch.Tensor):
            raise ValueError(f"{type(self.bias)=} but exepcted torch.Tensor")

        mask = self.bias[:, :, :S, :S] == 0
        att = att.masked_fill(mask, float("-inf"))  # causal part

        att = F.softmax(att, dim=-1)

        att = self.attn_dropout(att)

        y = einsum(att, v, "B Nh S Sv, B Nh Sv Ne -> B Nh S Ne")  # self part
        y = rearrange(y, "B Nh S Ne -> B S (Nh Ne)")

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config: configs.ModelConfig):
        super().__init__()

        if config.n_embd is None:
            raise ValueError(f"{config.n_embd=} cannot be None here.")

        self.attn = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            CausalSelfAttention(config),
        )
        self.mlpf = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, 4 * config.n_embd),
            NewGELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlpf(x)
        return x


class Transformer(nn.Module):
    def __init__(self, config: configs.ModelConfig):
        super().__init__()

        if config.vocab_size is None:
            raise ValueError(f"{config.vocab_size=} cannot be None here.")
        if config.block_size is None:
            raise ValueError(f"{config.block_size=} cannot be None here.")
        if config.n_embd is None:
            raise ValueError(f"{config.n_embd=} cannot be None here.")
        if config.n_layer is None:
            raise ValueError(f"{config.n_layer=} cannot be None here.")

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb)  # (b, t, n_embd)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)  # (b, t, n_embd)
        logits = self.lm_head(x)  # (b, t, vocab_size)
        return logits  # (b, t, vocab_size)


MODEL_CONFIGS = {
    # names follow the huggingface naming conventions
    # GPT-1
    "openai-gpt": dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
    # GPT-2 configs
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
    # Gophers
    "gopher-44m": dict(n_layer=8, n_head=16, n_embd=512),
    # (there are a number more...)
    # I made these tiny models up
    "gpt-mini": dict(n_layer=6, n_head=6, n_embd=192),
    "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
    "gpt-nano": dict(n_layer=3, n_head=3, n_embd=48),
}


def init_weights_given_module_type(module: nn.Module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.ones_(module.weight)


class GPT(nn.Module):
    """GPT Language Model"""

    @staticmethod
    def get_config(**kwargs) -> configs.ModelConfig:
        return configs.ModelConfig(**kwargs)

    def handle_config(self, config: configs.ModelConfig) -> configs.ModelConfig:
        assert config.vocab_size is not None
        assert config.block_size is not None

        type_given = config.model_type is not None
        params_given = all(
            [
                config.n_layer is not None,
                config.n_head is not None,
                config.n_embd is not None,
            ]
        )
        assert type_given ^ params_given  # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            _config = MODEL_CONFIGS[config.model_type]
            config = configs.get_modified_model_config(config, verbose=False, **_config)
        return config

    def __init__(self, config: configs.ModelConfig):
        super().__init__()

        config = self.handle_config(config)

        if config.block_size is None:
            raise ValueError(f"{config.block_size=} cannot be None here.")

        self.block_size = config.block_size
        self.transformer = Transformer(config)

        if config.n_layer is None:
            raise ValueError(f"{config.n_layer=} cannot be None here.")

        self._init_weights(config.n_layer)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        logger.info(f"number of parameters: {n_params / 1e6:.2f} M")

    def _init_weights(self, n_layer: int):
        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(init_weights_given_module_type)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def configure_optimizers(
        self, train_config: configs.TrainerConfig
    ) -> torch.optim.Optimizer:
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, (
            "parameters %s made it into both decay/no_decay sets!"
            % (str(inter_params),)
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=train_config.learning_rate,
            betas=train_config.betas,
        )
        return optimizer

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device

        _, t = idx.size()

        assert t <= self.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        )

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)

        logits = self.transformer(idx, pos)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k=None,
    ) -> torch.Tensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            )

            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
