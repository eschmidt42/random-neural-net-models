# -*- coding: utf-8 -*-
import pytest
import torch
import torch.nn.functional as F

import random_neural_net_models.text as rnnm_text
import random_neural_net_models.transformer as rnnm_trans


def test_TokensToQKV():
    dim_in = 5
    q_emb_dim = 10
    kv_emb_dim = 20

    latent_dim = 40

    tokens_to_qkv = rnnm_trans.TokensToQKV(q_emb_dim, kv_emb_dim, latent_dim)

    X_dec = torch.randn(dim_in, q_emb_dim)
    X_enc = torch.randn(dim_in, kv_emb_dim)

    Q, K, V = tokens_to_qkv(X_dec, X_enc)

    assert Q.shape == (dim_in, latent_dim)
    assert K.shape == (dim_in, latent_dim)
    assert V.shape == (dim_in, latent_dim)


def test_SplitHeads():
    num_heads = 4
    dim_in = 5
    latent_dim = 40
    batch_size = 3

    split_heads = rnnm_trans.SplitHeads(num_heads)

    Q = torch.randn(batch_size, dim_in, latent_dim)
    K = torch.randn(batch_size, dim_in, latent_dim)
    V = torch.randn(batch_size, dim_in, latent_dim)

    num_part = split_heads.get_num_part(Q)

    assert num_part == 10

    Q_split, K_split, V_split = split_heads(Q, K, V)

    assert Q_split.shape == (batch_size, num_heads, dim_in, num_part)
    assert K_split.shape == (batch_size, num_heads, dim_in, num_part)
    assert V_split.shape == (batch_size, num_heads, dim_in, num_part)


@pytest.mark.parametrize("n_tokens_q", [10, 11, 12])
def test_Causal(n_tokens_q: int):
    batch_size = 5
    num_heads = 4
    num_part = 10
    n_tokens_k = 11

    attn = torch.randn(batch_size, num_heads, n_tokens_q, n_tokens_k)
    Q = torch.randn(batch_size, num_heads, n_tokens_q, num_part)
    K = torch.randn(batch_size, num_heads, n_tokens_k, num_part)

    causal = rnnm_trans.Causal(n_tokens_q=n_tokens_q, n_tokens_kv=n_tokens_k)
    output = causal(attn, Q, K)

    assert output.shape == attn.shape
    assert not torch.allclose(output, attn)


@pytest.mark.parametrize("n_tokens_q", [10, 11, 12])
def test_NonCausal(n_tokens_q: int):
    batch_size = 5
    num_heads = 4
    num_part = 10
    n_tokens_k = 11

    attn = torch.randn(batch_size, num_heads, n_tokens_q, n_tokens_k)
    Q = torch.randn(batch_size, num_heads, n_tokens_q, num_part)
    K = torch.randn(batch_size, num_heads, n_tokens_k, num_part)

    noncausal = rnnm_trans.NonCausal()
    output = noncausal(attn, Q, K)

    assert output.shape == attn.shape
    assert torch.allclose(output, attn)


@pytest.mark.parametrize("causal", [False, True])
def test_Attention(causal: bool):
    latent_dim = 40
    out_dim = 30
    n_tokens_q = 10
    n_tokens_kv = 11
    num_heads = 4
    batch_size = 5
    num_part = 10

    attention = rnnm_trans.Attention(
        latent_dim,
        out_dim,
        causal=causal,
        n_tokens_q=n_tokens_q,
        n_tokens_kv=n_tokens_kv,
    )

    Q = torch.randn(batch_size, num_heads, n_tokens_q, num_part)
    K = torch.randn(batch_size, num_heads, n_tokens_kv, num_part)
    V = torch.randn(batch_size, num_heads, n_tokens_kv, num_part)

    output = attention(Q, K, V)

    assert output.shape == (batch_size, n_tokens_q, out_dim)
    assert torch.isfinite(output).all()


def test_SkipLayerNorm():
    batch_size = 4
    n_tokens = 10
    emb_dim = 20

    skip_layer_norm = rnnm_trans.SkipLayerNorm(n_tokens, emb_dim)

    x_0 = torch.randn(batch_size, n_tokens, emb_dim)
    x_1 = torch.randn(batch_size, n_tokens, emb_dim)

    output = skip_layer_norm(x_0, x_1)

    assert output.shape == (batch_size, n_tokens, emb_dim)
    assert torch.isfinite(output).all()


def test_sanity_check_latent_dim_and_num_heads():
    latent_dim = 40
    num_heads = 4

    # Test case where latent_dim is divisible by num_heads
    assert (
        rnnm_trans.sanity_check_latent_dim_and_num_heads(latent_dim, num_heads)
        is None
    )

    # Test case where latent_dim is not divisible by num_heads
    with pytest.raises(ValueError):
        rnnm_trans.sanity_check_latent_dim_and_num_heads(
            latent_dim=41, num_heads=num_heads
        )


@pytest.mark.parametrize("dec_n_tokens", [9, 10, 11])
@pytest.mark.parametrize("causal", [False, True])
def test_AttentionBlock(dec_n_tokens: int, causal: bool):
    dec_emb_dim = 10
    enc_emb_dim = 20
    enc_n_tokens = 10
    latent_dim = 40
    num_heads = 4
    batch_size = 6

    attention_block = rnnm_trans.AttentionBlock(
        dec_emb_dim=dec_emb_dim,
        dec_n_tokens=dec_n_tokens,
        enc_emb_dim=enc_emb_dim,
        enc_n_tokens=enc_n_tokens,
        latent_dim=latent_dim,
        num_heads=num_heads,
        causal=causal,
    )

    X_dec = torch.randn(batch_size, dec_n_tokens, dec_emb_dim)
    X_enc = torch.randn(batch_size, enc_n_tokens, enc_emb_dim)

    output = attention_block(X_dec, X_enc)

    assert output.shape == (batch_size, dec_n_tokens, dec_emb_dim)
    assert torch.isfinite(output).all()


def test_FFN():
    batch_size = 3
    dim_in = 5
    dim_hidden = 10
    dropout_rate = 0.2

    ffn = rnnm_trans.FFN(dim_in, dim_hidden, dropout_rate)

    X = torch.randn(batch_size, dim_in)

    output = ffn(X)

    assert output.shape == (batch_size, dim_in)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("causal", [False, True])
def test_EncoderTransformerBlock(causal: bool):
    emb_dim = 10
    n_tokens = 20
    latent_dim = 30
    num_heads = 2

    batch_size = 5

    encoder_block = rnnm_trans.EncoderTransformerBlock(
        emb_dim=emb_dim,
        n_tokens=n_tokens,
        latent_dim=latent_dim,
        num_heads=num_heads,
        causal=causal,
    )

    X_enc = torch.randn(batch_size, n_tokens, emb_dim)

    output = encoder_block(X_enc)

    assert output.shape == (batch_size, n_tokens, emb_dim)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("dec_n_tokens", [19, 20, 21])
@pytest.mark.parametrize("causal", [False, True])
def test_DecoderTransformerBlock(dec_n_tokens: int, causal: bool):
    dec_emb_dim = 10
    enc_emb_dim = 30
    enc_n_tokens = 20
    latent_dim = 40
    num_heads = 2
    batch_size = 5

    decoder_block = rnnm_trans.DecoderTransformerBlock(
        dec_emb_dim=dec_emb_dim,
        dec_n_tokens=dec_n_tokens,
        enc_emb_dim=enc_emb_dim,
        enc_n_tokens=enc_n_tokens,
        latent_dim=latent_dim,
        num_heads=num_heads,
        causal=causal,
    )

    X_dec = torch.randn(batch_size, dec_n_tokens, dec_emb_dim)
    X_enc = torch.randn(batch_size, enc_n_tokens, enc_emb_dim)

    output = decoder_block(X_dec, X_enc)

    assert output.shape == (batch_size, dec_n_tokens, dec_emb_dim)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("causal", [False, True])
def test_TransformerEncoder(causal: bool):
    num_blocks = 3
    enc_emb_dim = 10
    enc_n_tokens = 20
    latent_dim = 30
    num_heads = 2
    vocab_size = 100
    batch_size = 5

    transformer_encoder = rnnm_trans.TransformerEncoder(
        num_blocks=num_blocks,
        enc_emb_dim=enc_emb_dim,
        enc_n_tokens=enc_n_tokens,
        latent_dim=latent_dim,
        num_heads=num_heads,
        causal=causal,
        vocab_size=vocab_size,
    )

    X_enc = torch.randint(
        low=0, high=vocab_size, size=(batch_size, enc_n_tokens)
    )

    output = transformer_encoder(X_enc)

    assert output.shape == (batch_size, enc_n_tokens, enc_emb_dim)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("dec_n_tokens", [19, 20, 21])
@pytest.mark.parametrize("causal", [False, True])
def test_TransformerDecoder(dec_n_tokens: int, causal: bool):
    num_blocks = 3
    dec_emb_dim = 10
    enc_emb_dim = 10
    enc_n_tokens = 40
    latent_dim = 50
    num_heads = 2
    vocab_size = 100
    batch_size = 5

    transformer_decoder = rnnm_trans.TransformerDecoder(
        num_blocks=num_blocks,
        dec_emb_dim=dec_emb_dim,
        dec_n_tokens=dec_n_tokens,
        enc_emb_dim=enc_emb_dim,
        enc_n_tokens=enc_n_tokens,
        latent_dim=latent_dim,
        num_heads=num_heads,
        causal=causal,
        vocab_size=vocab_size,
    )

    X_dec = torch.randint(
        low=0, high=vocab_size, size=(batch_size, dec_n_tokens)
    )
    X_enc = torch.randn(size=(batch_size, enc_n_tokens, enc_emb_dim))

    output = transformer_decoder(X_dec, X_enc)

    assert output.shape == (batch_size, dec_n_tokens, dec_emb_dim)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("dec_n_tokens", [19, 20, 21])
@pytest.mark.parametrize("causal", [False, True])
def test_Transformer(dec_n_tokens: int, causal: int):
    num_blocks = 3
    enc_emb_dim = 10
    enc_n_tokens = 20
    dec_emb_dim = 10
    latent_dim = 30
    num_heads = 2
    vocab_size = 100
    batch_size = 5

    transformer = rnnm_trans.TransformerEncoderDecoder(
        num_blocks=num_blocks,
        enc_emb_dim=enc_emb_dim,
        enc_n_tokens=enc_n_tokens,
        dec_emb_dim=dec_emb_dim,
        dec_n_tokens=dec_n_tokens,
        latent_dim=latent_dim,
        num_heads=num_heads,
        causal=causal,
        vocab_size=vocab_size,
    )

    X_dec = torch.randint(
        low=0, high=vocab_size, size=(batch_size, dec_n_tokens)
    )
    X_enc = torch.randint(
        low=0, high=vocab_size, size=(batch_size, enc_n_tokens)
    )

    output = transformer(X_dec, X_enc)

    assert output.shape == (batch_size, dec_n_tokens, dec_emb_dim)
    assert torch.isfinite(output).all()


def test_LanguageModel():
    vocab_size = 100
    emb_dim = 10
    n_tokens = 20
    latent_dim = 30
    num_heads = 2
    num_blocks = 3
    batch_size = 5

    language_model = rnnm_trans.LanguageModel(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        n_tokens=n_tokens,
        latent_dim=latent_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
    )

    X = torch.randint(low=0, high=vocab_size, size=(batch_size, n_tokens))

    logits = language_model(X)

    assert logits.shape == (batch_size, n_tokens, vocab_size)
    assert torch.isfinite(logits).all()

    probs = F.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, n_tokens))

    generated_ids = language_model.generate(X, max_new_tokens=1)
    assert not torch.allclose(X[:, -1], generated_ids[:, -1])
    assert torch.allclose(X[:, -1], generated_ids[:, -2])


def test_LanguageModelWithTensordict():
    vocab_size = 100
    emb_dim = 10
    n_tokens = 20
    latent_dim = 30
    num_heads = 2
    num_blocks = 3
    batch_size = 5

    language_model = rnnm_trans.LanguageModelWithTensordict(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        n_tokens=n_tokens,
        latent_dim=latent_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
    )

    X = rnnm_text.TokenIDBlockX(
        torch.randint(low=0, high=vocab_size, size=(batch_size, n_tokens)),
        batch_size=[batch_size],
    )

    logits = language_model(X)

    assert logits.shape == (batch_size, n_tokens, vocab_size)
    assert torch.isfinite(logits).all()

    generated_ids = language_model.generate(X, max_new_tokens=1)
    assert not torch.allclose(X.x[:, -1], generated_ids[:, -1])
    assert torch.allclose(X.x[:, -1], generated_ids[:, -2])


def test_CrossEntropyLoss():
    num_classes = 10
    batch_size = 5
    seq_length = 20

    loss_fn = rnnm_trans.CrossEntropyLoss()

    inference = torch.randn(batch_size, seq_length, num_classes)
    input = rnnm_text.TokenIDBlockXY(
        x=torch.randn(batch_size, seq_length),
        y=torch.randint(num_classes, (batch_size, seq_length)),
        batch_size=[batch_size],
    )

    output = loss_fn(inference, input)

    assert output.shape == ()
    assert torch.isfinite(output).all()
