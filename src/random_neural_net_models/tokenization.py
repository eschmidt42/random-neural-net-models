# -*- coding: utf-8 -*-
import abc
import string
import typing as T
from collections import Counter

import regex
import tqdm
from pydantic import BaseModel, StrictInt, field_validator

import random_neural_net_models.utils as utils

logger = utils.logger


class TokenIDs(BaseModel):
    ids: T.List[StrictInt]

    def model_post_init(self, __context):
        self.ids = tuple(self.ids)

    def __iter__(self) -> T.Iterator[int]:
        return iter(self.ids)

    def __getitem__(self, i) -> int:
        return self.ids[i]

    def __len__(self) -> int:
        return len(self.ids)

    def __add__(self, other: "TokenIDs") -> "TokenIDs":
        new_values = list(self.ids) + list(other.ids)
        return TokenIDs(ids=new_values)


def text_to_ids(text: str) -> TokenIDs:
    return TokenIDs(
        ids=[int(v) for v in text.encode("utf-8", errors="replace")]
    )


def get_stats(token_ids: TokenIDs) -> Counter:
    return Counter(zip(token_ids[:-1], token_ids[1:]))


def merge_token_ids(
    token_ids: TokenIDs,
    pair_to_replace: T.Tuple[int, int],
    replacement_token: int,
) -> TokenIDs:
    new_ids = []
    i = 0
    while i < len(token_ids):
        if (
            i < len(token_ids) - 1
            and tuple(token_ids[i : i + 2]) == pair_to_replace
        ):
            new_ids.append(replacement_token)
            i += 2
        else:
            new_ids.append(token_ids[i])
            i += 1
    return TokenIDs(ids=new_ids)


class TokenIDMergeMap(BaseModel):
    map: T.Dict[T.Tuple[StrictInt, StrictInt], StrictInt]

    @field_validator("map")
    @classmethod
    def ensure_keys_unique(cls, v: dict):
        n_keys = len(v)
        n_vals = len(set(v.values()))
        if n_keys != n_vals:
            msg = f"Expected {n_keys=:_d} == {n_vals=:_d}"
            raise ValueError(msg)
        return v

    def __getitem__(self, pair: T.Tuple[int, int]) -> int:
        return self.map[pair]

    def __len__(self) -> int:
        return len(self.map)

    def items(self):
        return self.map.items()


def repeated_merge(
    token_ids: TokenIDs,
    vocab_size: int,
    show_progress: bool,
    base_tokens: TokenIDs = None,
    return_new_ids: bool = False,
) -> T.Union[T.Tuple[TokenIDMergeMap, TokenIDs], TokenIDMergeMap]:

    n0 = len(token_ids)
    n_used_tokens = len(set(token_ids))
    n_merges = vocab_size - n_used_tokens
    logger.info(
        f"repeatedly merging tokens: {n_merges=} to achieve {vocab_size=} with {n_used_tokens=}"
    )

    replacement_token = (
        max(token_ids + base_tokens) if base_tokens else max(token_ids)
    )
    pair_map = {}
    for _ in tqdm.tqdm(
        range(n_merges), total=n_merges, desc="merge", disable=not show_progress
    ):
        stats = get_stats(token_ids)
        pair_to_replace = stats.most_common()[0][0]
        replacement_token += 1
        token_ids = merge_token_ids(
            token_ids, pair_to_replace, replacement_token
        )
        pair_map[pair_to_replace] = replacement_token
    n1 = len(token_ids)
    logger.info(
        f"result: {n0:_d} -> {n1:_d} tokens = compression to {n1/n0:.2%} of tokens"
    )

    pair_map = TokenIDMergeMap(map=pair_map)
    if return_new_ids:
        return pair_map, token_ids
    return pair_map


def encode(
    text: str,
    pair_map: TokenIDMergeMap,
) -> TokenIDs:
    token_ids = TokenIDs(ids=[int(v) for v in text.encode("utf-8")])
    logger.debug(f"{len(token_ids)=:_d} raw tokens")
    if len(token_ids) == 1:
        logger.debug("Found only one token, returning.")
        return token_ids

    for _ in range(len(pair_map)):
        stats = get_stats(token_ids)
        is_done = not any(p in pair_map for p in stats)
        if is_done:
            logger.debug(f"{len(token_ids)=:_d} after applying pair map")
            return token_ids

        pair = min(stats, key=lambda pair: pair_map.map.get(pair, float("inf")))
        idx = pair_map[pair]
        token_ids = merge_token_ids(token_ids, pair, idx)

    logger.debug(f"{len(token_ids)=:_d} after applying pair map")

    return token_ids


FALLBACK_TOKEN = "🦥"
FALLBACK_TOKEN_BYTES = FALLBACK_TOKEN.encode(encoding="utf-8")


def decode(
    token_ids: TokenIDs,
    vocab: T.Dict[int, bytes],
    fallback: bytes = FALLBACK_TOKEN_BYTES,
) -> str:
    tokens = [vocab.get(token, fallback) for token in token_ids]
    tokens = b"".join(tokens)
    text = tokens.decode("utf-8", errors="replace")
    return text


BASE_SYMBOLS = string.ascii_letters + string.digits + string.punctuation


class TokenizerBase(abc.ABC):

    base_symbols: str
    base_token_ids: TokenIDs
    vocab: T.Dict[int, bytes]
    pair_map: TokenIDMergeMap

    def __init__(self, base_symbols: str = None):
        self.base_symbols = base_symbols if base_symbols else BASE_SYMBOLS
        self.base_token_ids = text_to_ids(self.base_symbols)

    @abc.abstractmethod
    def fit(self, text: str, vocab_size: int, verbose: int = False): ...

    @abc.abstractmethod
    def encode(self, text: str) -> TokenIDs: ...

    @abc.abstractmethod
    def decode(self, tokens: TokenIDs) -> str: ...


class TokenizerSimple(TokenizerBase):

    def fit(self, text: str, vocab_size: int, verbose: int = False):
        token_ids = text_to_ids(text)
        self.pair_map = repeated_merge(
            token_ids,
            vocab_size,
            show_progress=verbose,
            base_tokens=self.base_token_ids,
            return_new_ids=False,
        )
        self.vocab = {
            idx: bytes([idx]) for idx in set(token_ids + self.base_token_ids)
        }
        for (token0, token1), idx in self.pair_map.items():
            self.vocab[idx] = self.vocab[token0] + self.vocab[token1]

    def encode(self, text: str) -> TokenIDs:
        return encode(text, self.pair_map)

    def decode(self, token_ids: TokenIDs) -> str:
        return "".join(decode(token_ids, self.vocab))


GPT4_SPLIT_PATTERN = regex.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


class TokenizerRegex(TokenizerSimple):

    use_special_tokens: bool = False

    def fit(
        self,
        text: str,
        vocab_size: int,
        pattern: regex.Pattern,
    ):
        self.pattern = pattern

        token_ids: T.List[TokenIDs] = [
            text_to_ids(chunk) for chunk in self.pattern.findall(text)
        ]
        unique_tokens = set(tok for chunk in token_ids for tok in chunk)

        n_merges = vocab_size - len(unique_tokens)

        if n_merges <= 0:
            raise ValueError(
                f"{n_merges=} needs to be > 0 ({vocab_size=:_d}, {len(unique_tokens)=:_d})"
            )

        unique_tokens.update(self.base_token_ids)

        self.pair_map = {}

        replacement_token = max(unique_tokens)

        for _ in range(n_merges):

            stats = Counter()
            for chunk in token_ids:
                stats.update(get_stats(chunk))

            pair_to_replace = stats.most_common()[0][0]
            replacement_token += 1

            token_ids = [
                merge_token_ids(chunk, pair_to_replace, replacement_token)
                for chunk in token_ids
            ]

            self.pair_map[pair_to_replace] = replacement_token

        self.pair_map = TokenIDMergeMap(map=self.pair_map)

        self.vocab = {idx: bytes([idx]) for idx in unique_tokens}
        for (token0, token1), idx in self.pair_map.items():
            self.vocab[idx] = self.vocab[token0] + self.vocab[token1]

    def register_special_tokens(
        self,
        special_token2id_map: T.Dict[str, int],
    ):
        self.special_token2id_map = special_token2id_map
        self.inv_special_token2id_map = {
            v: k for k, v in special_token2id_map.items()
        }
        self.special_pattern = (
            "(" + "|".join(regex.escape(k) for k in special_token2id_map) + ")"
        )
        self.use_special_tokens = True

    def _ordinary_encode(self, text: str) -> TokenIDs:
        ids = TokenIDs(ids=[])
        for chunk in self.pattern.findall(text):
            ids += encode(chunk, self.pair_map)
        return ids

    def encode(self, text: str) -> TokenIDs:
        if not self.use_special_tokens:
            return self._ordinary_encode(text)
        else:
            chunks = regex.split(self.special_pattern, text)
            ids = TokenIDs(ids=[])
            for chunk in chunks:
                if chunk in self.special_token2id_map:
                    ids += TokenIDs(ids=[self.special_token2id_map[chunk]])
                else:
                    ids += self._ordinary_encode(chunk)
            return ids

    def decode(self, token_ids: TokenIDs) -> str:

        if not self.use_special_tokens:
            return "".join(decode(token_ids, self.vocab))
        else:
            text = []
            for _id in token_ids:
                if _id in self.inv_special_token2id_map:
                    text.append(self.inv_special_token2id_map[_id])
                else:
                    text.extend(decode(TokenIDs(ids=[_id]), self.vocab))
            return "".join(text)
