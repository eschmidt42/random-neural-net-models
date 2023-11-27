# -*- coding: utf-8 -*-
import torch

import random_neural_net_models.mingpt.data as gpt_data
import random_neural_net_models.mingpt.model as gpt_model
import random_neural_net_models.mingpt.trainer as gpt_trainer
import random_neural_net_models.mingpt.utils as gpt_utils

gpt_utils.set_seed(3407)


def test_mingpt():
    train_dataset = gpt_data.SortDataset(gpt_data.SET_CHOICE.train)

    model_config = gpt_model.GPT.get_config(
        model_type="gpt-nano",
        vocab_size=train_dataset.get_vocab_size(),
        block_size=train_dataset.get_block_size(),
    )
    model = gpt_model.GPT(model_config)

    train_config = gpt_trainer.Trainer.get_config(
        learning_rate=5e-4,  # the model we're using is so small that we can go a bit faster
        max_iters=200,
        num_workers=0,
    )

    trainer = gpt_trainer.Trainer(train_config, model, train_dataset)

    trainer.run()

    n = train_dataset.length  # naugy direct access shrug
    inp = torch.tensor([[0, 0, 2, 1, 0, 1]], dtype=torch.long).to(
        trainer.device
    )

    with torch.no_grad():
        cat = model.generate(inp, n, do_sample=False)

    # generated output is as expected
    assert torch.allclose(
        cat[:, n:], torch.tensor([[0, 0, 0, 1, 1, 2]], dtype=torch.long)
    )
