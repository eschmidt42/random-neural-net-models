# -*- coding: utf-8 -*-
import typing as T
from pathlib import Path

import torch
from tensordict import tensorclass
from torch.utils.data import Dataset
import random_neural_net_models.utils as utils
import random_neural_net_models.tokenization as rnnm_tok

logger = utils.logger


def find_files(path: Path, suffix: str) -> T.List[Path]:
    return list(path.rglob(suffix))


def concat_files(files: T.List[Path], join_str: str) -> str:
    lines = [f.read_text(encoding="utf-8") for f in files]
    return join_str.join(lines)


class TextDataset(Dataset):
    def __init__(
        self,
        path: Path,
        suffix: str,
        tokenizer: rnnm_tok.TokenizerBase,
        block_size: int,
        end_of_text_token: str,
    ):
        self.files = find_files(path, suffix)
        self.end_of_text_token = end_of_text_token
        self.text = concat_files(self.files, end_of_text_token)
        self.tokenizer = tokenizer
        self.text_encoded = tokenizer.encode(self.text)
        self.block_size = block_size

        self.map_ids_scattered2dense = {
            id_scattered: id_dense
            for id_dense, id_scattered in enumerate(self.tokenizer.vocab)
        }
        id_end_of_text_token = tokenizer.special_token2id_map[end_of_text_token]
        self.map_ids_scattered2dense[id_end_of_text_token] = (
            max(self.map_ids_scattered2dense.values()) + 1
        )
        self.map_ids_dense2scattered = {
            id_dense: id_scattered
            for id_scattered, id_dense in self.map_ids_scattered2dense.items()
        }

    @property
    def vocab_size(self) -> int:
        return len(
            self.map_ids_scattered2dense
        )  # map_ids_scattered2dense because of end_of_text_token

    def dense_to_scattered_ids(
        self, ids_dense: torch.LongTensor
    ) -> torch.LongTensor:
        shape = ids_dense.shape

        ids_dense_flat = ids_dense.ravel()
        ids_scattered = torch.LongTensor(
            [self.map_ids_dense2scattered[int(_id)] for _id in ids_dense_flat]
        )
        ids_scattered = ids_scattered.reshape(shape)
        return ids_scattered

    def scattered_ids_to_strings(
        self, ids_scattered: torch.LongTensor
    ) -> T.List[str]:
        strings = []
        for ids in ids_scattered:
            ids = rnnm_tok.TokenIDs(ids=map(int, ids))
            strings.append(self.tokenizer.decode(ids))
        return strings

    def dense_ids_to_strings(self, ids_dense: torch.LongTensor) -> T.List[str]:
        ids_scattered = self.dense_to_scattered_ids(ids_dense)
        return self.scattered_ids_to_strings(ids_scattered)

    def text_to_dense_ids(self, text: str) -> torch.LongTensor:
        ids = self.tokenizer.encode(text)
        if len(ids) > self.block_size:
            msg = f"Given text was encoded into {len(ids)} token ids, which exceeds {self.block_size=:_d}, hence keeping only the last {self.block_size}"
            logger.warning(msg)
            ids = ids[-self.block_size :]
        ids = [
            self.map_ids_scattered2dense[id_scattered] for id_scattered in ids
        ]
        return torch.tensor([ids], dtype=torch.long)

    def __len__(self):
        return len(self.text_encoded) - self.block_size

    def __getitem__(self, idx: int):
        scattered_ids = self.text_encoded[idx : idx + self.block_size + 1]
        ids = [
            self.map_ids_scattered2dense[id_scattered]
            for id_scattered in scattered_ids
        ]

        # return as tensors
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


@tensorclass
class TokenIDBlockXY:
    x: torch.LongTensor
    y: torch.LongTensor


def collate_text_dataset_to_block(
    input: T.List[T.Tuple[torch.LongTensor, torch.LongTensor]],
) -> TokenIDBlockXY:
    x = torch.stack([v[0] for v in input])
    y = torch.stack([v[1] for v in input])

    return TokenIDBlockXY(
        x=x,
        y=y,
        batch_size=[x.shape[0]],
    )


@tensorclass
class TokenIDBlockX:
    x: torch.LongTensor
