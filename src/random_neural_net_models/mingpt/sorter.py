# -*- coding: utf-8 -*-
import pickle
import typing as T

import torch
from torch.utils.data import Dataset

import random_neural_net_models.mingpt.utils as gpt_utils
import random_neural_net_models.utils as utils

logger = utils.get_logger("mingpt.sorter")


def generate_list_of_random_integers(
    num_digits: int, length: int
) -> torch.Tensor:
    while True:
        # generate some random integers
        inp = torch.randint(num_digits, size=(length,), dtype=torch.long)
        # half of the time let's try to boost the number of examples that
        # have a large number of repeats, as this is what the model seems to struggle
        # with later in training, and they are kind of rate
        if torch.rand(1).item() < 0.5:
            if inp.unique().nelement() > length // 2:
                # too many unqiue digits, re-sample
                continue
        yield inp


def check_split(inp: torch.Tensor) -> gpt_utils.SET_CHOICE:
    # figure out if this generated example is train or test based on its hash
    h = hash(pickle.dumps(inp.tolist()))
    return (
        gpt_utils.SET_CHOICE.test if h % 4 == 0 else gpt_utils.SET_CHOICE.train
    )  # designate 25% of examples as test


class SortDataset(Dataset):
    """
    Dataset for the Sort problem. E.g. for problem length 6:
    Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
    Which will feed into the transformer concatenated as:
    input:  0 0 2 1 0 1 0 0 0 1 1
    output: I I I I I 0 0 0 1 1 2
    where I is "ignore", as the transformer is reading the input sequence
    """

    def __init__(
        self,
        split: gpt_utils.SET_CHOICE,
        length: int = 6,
        num_digits: int = 3,
        n_samples: int = 10_000,
    ):
        self.split = split
        self.length = length
        self.num_digits = num_digits
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def get_vocab_size(self):
        return self.num_digits

    def get_block_size(self):
        # the length of the sequence that will feed into transformer,
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length * 2 - 1

    def __getitem__(self, idx: int) -> T.Tuple[torch.Tensor, torch.Tensor]:
        # use rejection sampling to generate an input example from the desired split
        integer_generator = generate_list_of_random_integers(
            self.num_digits, self.length
        )
        for inp in integer_generator:
            inp_split = check_split(inp)

            if inp_split == self.split:
                break  # ok

        # solve the task: i.e. sort
        sol = torch.sort(inp)[0]

        # concatenate the problem specification and the solution
        cat = torch.cat((inp, sol), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input locations
        y[: self.length - 1] = -1
        return x, y
