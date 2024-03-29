# -*- coding: utf-8 -*-
from pathlib import Path
from einops import rearrange
import pytest
from unittest.mock import patch
import torch
import random_neural_net_models.text as rnnm_text
import random_neural_net_models.tokenization as rnnm_tok


def test_concat_files():
    file1 = Path("/path/to/file1.txt")
    file2 = Path("/path/to/file2.txt")
    file3 = Path("/path/to/file3.txt")
    files = [file1, file2, file3]
    join_str = "\n"

    expected_output = "Content of file1\nContent of file2\nContent of file3"

    with patch("pathlib.Path.read_text") as mock_read_text:
        mock_read_text.side_effect = expected_output.split("\n")

        assert rnnm_text.concat_files(files, join_str) == expected_output


@pytest.fixture(scope="module")
def dataset():
    path = Path("data/tom-lehrer")
    files = rnnm_text.find_files(path, "*.txt")
    assert len(files) > 0
    body_for_tokenizer = rnnm_text.concat_files(files, "\n")
    vocab_size = 200
    tokenizer = rnnm_tok.TokenizerRegex()
    tokenizer.fit(
        body_for_tokenizer,
        vocab_size=vocab_size,
        pattern=rnnm_tok.GPT4_SPLIT_PATTERN,
    )

    special_token2id_map = {
        "<|endoftext|>": 100257,
        "<|fim_prefix|>": 100258,
        "<|fim_middle|>": 100259,
        "<|fim_suffix|>": 100260,
        "<|endofprompt|>": 100276,
    }
    tokenizer.register_special_tokens(special_token2id_map)

    block_size = 128
    return rnnm_text.TextDataset(
        path=path,
        suffix="*.txt",
        tokenizer=tokenizer,
        block_size=block_size,
        end_of_text_token="<|endoftext|>",
    )


def test_text_dataset_length(dataset: rnnm_text.TextDataset):
    expected_length = len(dataset.text_encoded) - dataset.block_size
    assert len(dataset) == expected_length


def test_text_dataset_getitem(dataset: rnnm_text.TextDataset):
    idx = 0
    x, y = dataset[idx]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    assert len(x) == dataset.block_size
    assert len(y) == dataset.block_size


def test_dense_to_scattered_ids(dataset: rnnm_text.TextDataset):
    dense_ids, _ = dataset[0]

    scattered_ids = dataset.dense_to_scattered_ids(dense_ids)

    assert isinstance(scattered_ids, torch.Tensor)
    assert scattered_ids.dtype == torch.long
    assert not torch.allclose(scattered_ids, dense_ids)
    assert scattered_ids.shape == dense_ids.shape


def test_scattered_ids_to_strings(dataset: rnnm_text.TextDataset):
    dense_ids, _ = dataset[0]
    dense_ids = rearrange(dense_ids, "n -> 1 n")
    scattered_ids = dataset.dense_to_scattered_ids(dense_ids)
    strings = dataset.scattered_ids_to_strings(scattered_ids)

    assert isinstance(strings, list)
    assert all(isinstance(s, str) for s in strings)
    assert len(strings) == dense_ids.shape[0]


def test_dense_ids_to_strings(dataset: rnnm_text.TextDataset):
    dense_ids, _ = dataset[0]
    dense_ids = rearrange(dense_ids, "n -> 1 n")
    scattered_ids = dataset.dense_to_scattered_ids(dense_ids)
    strings = dataset.scattered_ids_to_strings(scattered_ids)

    strings2 = dataset.dense_ids_to_strings(dense_ids)

    assert len(strings) == len(strings2)
    assert all(s1 == s2 for s1, s2 in zip(strings, strings2))


def test_collate_text_dataset_to_block():
    input_data = [
        (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])),
        (torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])),
        (torch.tensor([13, 14, 15]), torch.tensor([16, 17, 18])),
    ]

    expected_output = rnnm_text.TokenIDBlockXY(
        x=torch.tensor([[1, 2, 3], [7, 8, 9], [13, 14, 15]]),
        y=torch.tensor([[4, 5, 6], [10, 11, 12], [16, 17, 18]]),
        batch_size=[3],
    )

    assert (
        rnnm_text.collate_text_dataset_to_block(input_data) == expected_output
    )
