# -*- coding: utf-8 -*-
"""
Trains a GPT to add n-digit numbers.
"""
import typing as T

import torch
from pydantic.dataclasses import dataclass
from torch.utils.data import Dataset

import random_neural_net_models.mingpt.configs as configs
import random_neural_net_models.mingpt.data as data

# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DataConfig:
    ndigit: int


@dataclass(frozen=True)
class AdderConfig:
    system: configs.SystemConfig
    data: DataConfig
    model: configs.ModelConfig
    trainer: configs.TrainerConfig


def get_config(vocab_size: int, block_size: int, max_iters: int) -> AdderConfig:
    return AdderConfig(
        system=configs.SystemConfig(seed=3407, work_dir="./out/adder"),
        data=DataConfig(ndigit=2),
        model=configs.ModelConfig(
            model_type="gpt-nano",
            vocab_size=vocab_size,
            block_size=block_size,
        ),
        trainer=configs.TrainerConfig(
            max_iters=max_iters,
            learning_rate=5e-4,  # the model we're using is so small that we can go a bit faster
        ),
    )


# -----------------------------------------------------------------------------
def generate_list_of_random_integers(ndigit: int) -> torch.Tensor:
    # split up all addition problems into either training data or test data
    ndigit = ndigit
    assert (
        ndigit <= 3
    ), "the lines below would be very memory inefficient, in future maybe refactor to support"
    num = (
        10**ndigit
    ) ** 2  # total number of possible addition problems with ndigit numbers
    rng = torch.Generator()
    rng.manual_seed(1337)
    perm = torch.randperm(num, generator=rng)
    return perm


def get_abc(idx: int, ndigit: int) -> T.Tuple[int, int, int]:
    nd = 10**ndigit
    a = idx // nd
    b = idx % nd
    # calculate the "label" of the addition problem a + b
    c = a + b
    return a, b, c


def int2str(x: int, ndigit: int) -> str:
    return f"{x:0{ndigit}d}"


def encode_addition_problem(a: int, b: int, c: int, ndigit: int) -> T.List[int]:
    # if a = 1, b = 1, c = 2, ndigit = 2
    astr = int2str(a, ndigit)  # "01"
    bstr = int2str(b, ndigit)  # "01"
    cstr = int2str(c, ndigit + 1)[::-1]  # "200"
    render = astr + bstr + cstr  # "0101200"
    dix = [int(s) for s in render]  # convert each character to its token index
    return dix


class AdditionDataset(Dataset):
    """
    Creates n-digit addition problems. For example, if n=2, then an example
    addition problem would be to add 85 + 50 = 135. This problem would be
    represented as the following string for the GPT:

    "8550531"

    This is because:
    - we are discarding the + and =, which are not necessary. We just encode the digits
      of the input numbers concatenated together.
    - the result 135 is encoded backwards to make the addition easier to learn for the
      GPT model, because of how the addition algorithm works.

    As one more example, the problem 6 + 39 = 45 would be encoded as:

    "0639054"

    where you will notice that we are padding with zeros to make sure that we always
    produce strings of the exact same size: n + n + (n + 1). When n=2, this is 7.
    At test time, we will feed in an addition problem by giving the first 2n digits,
    and hoping that the GPT model completes the sequence with the next (n+1) digits
    correctly.
    """

    @staticmethod
    def get_config(ndigit: int = 2) -> DataConfig:
        return DataConfig(ndigit=ndigit)

    def __init__(self, config: DataConfig, split: data.SET_CHOICE):
        self.config = config
        self.split = split  # train/test

        perm = generate_list_of_random_integers(config.ndigit)

        num_test = min(int(len(perm) * 0.2), 500)
        self.ixes = (
            perm[:num_test]
            if split == data.SET_CHOICE.test
            else perm[num_test:]
        )

    def get_vocab_size(self):
        return 10  # digits 0..9

    def get_block_size(self):
        # a,b,a+b, and +1 due to potential carry overflow,
        # but then also -1 because very last digit doesn't ever plug back
        # as there is no explicit <EOS> token to predict, it is implied
        return 3 * self.config.ndigit + 1 - 1

    def __len__(self):
        return self.ixes.nelement()

    def __getitem__(self, idx):
        ndigit = self.config.ndigit  # 2

        a, b, c = get_abc(self.ixes[idx].item(), ndigit)  # 1, 1, 2

        dix = encode_addition_problem(a, b, c, ndigit)  # [0,1,0,1,2,0,0]

        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)  # [0,1,0,1,2,0]

        # predict the next token in the sequence
        y = torch.tensor(dix[1:], dtype=torch.long)  # [1,0,1,2,0,0]

        # we will only train in the output locations. -1 will mask loss to zero
        mask = ndigit * 2 - 1
        y[:mask] = -1  # [-1,-1,-1,2,0,0]

        return x, y
