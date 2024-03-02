# -*- coding: utf-8 -*-
from collections import Counter

import pytest
from pydantic import ValidationError

import random_neural_net_models.tokenization as rnnm_tok

TEST_STRINGS = {
    "tom lehrer: the elements song": """THE ELEMENTS

There's antimony, arsenic, aluminum, selenium,
And hydrogen and oxygen and nitrogen and rhenium,
And nickel, neodymium, neptunium, germanium,
And iron, americium, ruthenium, uranium,

Europium, zirconium, lutetium, vanadium,
And lanthanum and osmium and astatine and radium,
And gold and protactinium and indium and gallium,
And iodine and thorium and thulium and thallium.

There's yttrium, ytterbium, actinium, rubidium,
And boron, gadolinium, niobium, iridium,
And strontium and silicon and silver and samarium,
And bismuth, bromine, lithium, beryllium, and barium.

There's holmium and helium and hafnium and erbium,
And phosphorus and francium and fluorine and terbium,
And manganese and mercury, molybdenum, magnesium,
Dysprosium and scandium and cerium and cesium.

And lead, praseodymium and platinum, plutonium,
Palladium, promethium, potassium, polonium,
And tantalum, technetium, titanium, tellurium,
And cadmium and calcium and chromium and curium.

There's sulfur, californium and fermium, berkelium,
And also mendelevium, einsteinium, nobelium,
And argon, krypton, neon, radon, xenon, zinc and rhodium,
And chlorine, carbon, cobalt, copper, tungsten, tin and sodium.

These are the only ones o_f which the news has come to Ha'vard,
And there may be many others but they haven't been discavard.""",
    "tom lehrer: motto": "From adolescence to senility, bypassing maturity.",
}


@pytest.mark.parametrize(
    "values",
    [
        tuple([1, 2, 3]),
        tuple([4, 5, 6]),
        [1, 2, 3],
        [4, 5, 6],
    ],
)
def test_tokenids(values):
    ids = rnnm_tok.TokenIDs(ids=values)
    assert ids.ids == tuple(values)

    for i, t in zip(values, ids):
        assert i == t

    for i, expected in enumerate(values):
        assert expected == ids[i]

    assert tuple(values[1:]) == ids[1:]
    assert tuple(values[:-1]) == ids[:-1]
    assert len(ids) == len(values)


def test_tokenids_append():
    a = [1, 2, 3]
    b = [4, 5, 6]

    ids_a = rnnm_tok.TokenIDs(ids=a)
    ids_b = rnnm_tok.TokenIDs(ids=b)
    assert ids_a + ids_b == rnnm_tok.TokenIDs(ids=a + b)


@pytest.mark.parametrize(
    "values",
    [
        "not a list",
        {"key": "value"},
        123,
        [1, 2.0, 3],
        [1.0, 2.0],
        [1, None],
    ],
)
def test_tokens_invalid_values(values):
    with pytest.raises(ValidationError):
        rnnm_tok.TokenIDs(ids=values)


def test_text_to_ids():

    expected = rnnm_tok.TokenIDs(
        ids=(
            70,
            114,
            111,
            109,
            32,
            97,
            100,
            111,
            108,
            101,
            115,
            99,
            101,
            110,
            99,
            101,
            32,
            116,
            111,
            32,
            115,
            101,
            110,
            105,
            108,
            105,
            116,
            121,
            44,
            32,
            98,
            121,
            112,
            97,
            115,
            115,
            105,
            110,
            103,
            32,
            109,
            97,
            116,
            117,
            114,
            105,
            116,
            121,
            46,
        )
    )
    text = TEST_STRINGS["tom lehrer: motto"]
    actual = rnnm_tok.text_to_ids(text)

    assert actual == expected


def test_get_stats():
    text = TEST_STRINGS["tom lehrer: motto"]
    token_ids = [int(v) for v in text.encode("utf-8")]

    # line to test
    stats = rnnm_tok.get_stats(token_ids)

    assert isinstance(stats, Counter)
    assert stats.most_common(5) == [
        ((99, 101), 2),
        ((101, 110), 2),
        ((105, 116), 2),
        ((116, 121), 2),
        ((70, 114), 1),
    ]


def test_merge_token_ids():
    ids_orig = rnnm_tok.TokenIDs(ids=[5, 6, 6, 7, 9, 1])
    expected_ids = rnnm_tok.TokenIDs(ids=[5, 6, 99, 9, 1])
    actual_ids = rnnm_tok.merge_token_ids(ids_orig, (6, 7), 99)
    assert expected_ids.ids == actual_ids.ids


def test_repeated_merge():

    text = TEST_STRINGS["tom lehrer: motto"]
    token_ids = [int(v) for v in text.encode("utf-8")]
    vocab_size = len(set(token_ids)) + 20

    # line to test
    actual_map, actual_ids = rnnm_tok.repeated_merge(
        token_ids=token_ids,
        vocab_size=vocab_size,
        show_progress=False,
        return_new_ids=True,
    )
    assert isinstance(actual_map, rnnm_tok.TokenIDMergeMap)
    assert isinstance(actual_ids, rnnm_tok.TokenIDs)
    assert actual_map.map is not None

    expected_ids = rnnm_tok.TokenIDs(
        ids=(
            141,
            115,
            101,
            110,
            105,
            108,
            124,
            44,
            32,
            98,
            121,
            112,
            97,
            115,
            115,
            105,
            110,
            103,
            32,
            109,
            97,
            116,
            117,
            114,
            124,
            46,
        )
    )
    assert actual_ids == expected_ids

    expected_map = rnnm_tok.TokenIDMergeMap(
        map={
            (99, 101): 122,
            (105, 116): 123,
            (123, 121): 124,
            (70, 114): 125,
            (125, 111): 126,
            (126, 109): 127,
            (127, 32): 128,
            (128, 97): 129,
            (129, 100): 130,
            (130, 111): 131,
            (131, 108): 132,
            (132, 101): 133,
            (133, 115): 134,
            (134, 122): 135,
            (135, 110): 136,
            (136, 122): 137,
            (137, 32): 138,
            (138, 116): 139,
            (139, 111): 140,
            (140, 32): 141,
        }
    )
    assert expected_map == actual_map


@pytest.mark.parametrize("vals", [{(1, 2): 3}, {(1, 2): 3, (4, 5): 6}])
def test_tokenid_merge_map(vals):

    # line to test
    id_map = rnnm_tok.TokenIDMergeMap(map=vals)


@pytest.mark.parametrize("vals", [{(1, 2): "3"}, {(1, 2): 3, (4, 5): 3}])
def test_tokenid_merge_map_fails(vals):

    # line to test
    with pytest.raises(ValidationError):
        id_map = rnnm_tok.TokenIDMergeMap(map=vals)


def test_decode():

    token_ids = rnnm_tok.TokenIDs(
        ids=(
            141,
            115,
            101,
            110,
            105,
            108,
            124,
            44,
            32,
            98,
            121,
            112,
            97,
            115,
            115,
            105,
            110,
            103,
            32,
            109,
            97,
            116,
            117,
            114,
            124,
            46,
        )
    )
    vocab = {
        32: b" ",
        44: b",",
        46: b".",
        48: b"0",
        49: b"1",
        50: b"2",
        51: b"3",
        52: b"4",
        53: b"5",
        54: b"6",
        55: b"7",
        56: b"8",
        57: b"9",
        65: b"A",
        66: b"B",
        67: b"C",
        68: b"D",
        69: b"E",
        70: b"F",
        71: b"G",
        72: b"H",
        73: b"I",
        74: b"J",
        75: b"K",
        76: b"L",
        77: b"M",
        78: b"N",
        79: b"O",
        80: b"P",
        81: b"Q",
        82: b"R",
        83: b"S",
        84: b"T",
        85: b"U",
        86: b"V",
        87: b"W",
        88: b"X",
        89: b"Y",
        90: b"Z",
        97: b"a",
        98: b"b",
        99: b"c",
        100: b"d",
        101: b"e",
        102: b"f",
        103: b"g",
        104: b"h",
        105: b"i",
        106: b"j",
        107: b"k",
        108: b"l",
        109: b"m",
        110: b"n",
        111: b"o",
        112: b"p",
        113: b"q",
        114: b"r",
        115: b"s",
        116: b"t",
        117: b"u",
        118: b"v",
        119: b"w",
        120: b"x",
        121: b"y",
        122: b"ce",
        123: b"it",
        124: b"ity",
        125: b"Fr",
        126: b"Fro",
        127: b"From",
        128: b"From ",
        129: b"From a",
        130: b"From ad",
        131: b"From ado",
        132: b"From adol",
        133: b"From adole",
        134: b"From adoles",
        135: b"From adolesce",
        136: b"From adolescen",
        137: b"From adolescence",
        138: b"From adolescence ",
        139: b"From adolescence t",
        140: b"From adolescence to",
        141: b"From adolescence to ",
    }
    text = rnnm_tok.decode(token_ids=token_ids, vocab=vocab)

    assert text == "From adolescence to senility, bypassing maturity."


def test_encode():
    pair_map = rnnm_tok.TokenIDMergeMap(
        map={
            (99, 101): 122,
            (105, 116): 123,
            (123, 121): 124,
            (70, 114): 125,
            (125, 111): 126,
            (126, 109): 127,
            (127, 32): 128,
            (128, 97): 129,
            (129, 100): 130,
            (130, 111): 131,
            (131, 108): 132,
            (132, 101): 133,
            (133, 115): 134,
            (134, 122): 135,
            (135, 110): 136,
            (136, 122): 137,
            (137, 32): 138,
            (138, 116): 139,
            (139, 111): 140,
            (140, 32): 141,
        }
    )
    actual_ids = rnnm_tok.encode("bla bla and bla", pair_map)

    expected_ids = rnnm_tok.TokenIDs(
        ids=(98, 108, 97, 32, 98, 108, 97, 32, 97, 110, 100, 32, 98, 108, 97)
    )
    assert actual_ids == expected_ids


@pytest.mark.parametrize("name", ["simple", "regex"])
def test_tokenizer(name: str):
    train_text = TEST_STRINGS["tom lehrer: the elements song"]
    eval_text = TEST_STRINGS["tom lehrer: motto"]
    vocab_size = 60

    match name:
        case "simple":
            model = rnnm_tok.TokenizerSimple()
            model.fit(train_text, vocab_size=vocab_size)
        case "regex":
            model = rnnm_tok.TokenizerRegex()
            model.fit(
                train_text,
                vocab_size=vocab_size,
                pattern=rnnm_tok.GPT4_SPLIT_PATTERN,
            )
        case _:
            raise ValueError(f"Received unexpected tokenizer {model}")

    token_ids = model.encode(eval_text)
    decoded_text = model.decode(token_ids)
    assert eval_text == decoded_text


def test_tokenizer_with_special_tokens():
    train_text = TEST_STRINGS["tom lehrer: the elements song"]
    special_strings = """
<|endoftext|>Hello world this is one document
<|endoftext|>And this is another document
<|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
<|endoftext|>Last document!!! <|endofprompt|>
""".strip()

    special_token2id_map = {
        "<|endoftext|>": 100257,
        "<|fim_prefix|>": 100258,
        "<|fim_middle|>": 100259,
        "<|fim_suffix|>": 100260,
        "<|endofprompt|>": 100276,
    }
    vocab_size = 200
    model = rnnm_tok.TokenizerRegex()
    model.fit(
        train_text, vocab_size=vocab_size, pattern=rnnm_tok.GPT4_SPLIT_PATTERN
    )

    model.register_special_tokens(special_token2id_map)

    encoded_ids = model.encode(special_strings)
    decoded_text = model.decode(encoded_ids)

    assert decoded_text == special_strings
