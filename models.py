#!/usr/bin/env python

from bigdl.nano.utils.log4Error import invalidInputError
from bigdl.chronos.pytorch import TSTrainer as Trainer
from bigdl.chronos.model.tcn import model_creator
import torch


def generate_forecaster(args):
    input_feature_num = 321 if args.dataset == "tsinghua_electricity" else 1
    output_feature_num = 321 if args.dataset == "tsinghua_electricity" else 1
    metrics = args.metrics
    freq = 'h' if args.dataset == "tsinghua_electricity" else 't'
    if 'ETT' in args.dataset:
        input_feature_num = 7
        output_feature_num = 7

    config = {'input_feature_num':input_feature_num,
            'output_feature_num':output_feature_num,
            'past_seq_len':args.lookback,
            'future_seq_len':args.horizon,
            'kernel_size':3,
            'repo_initialization':True,
            'dropout':0.1,
            'seed': 0,
            'num_channels':[16]*3,
            'normalization':args.normalization,
            'dummy_encoder':args.dummy_encoder
            }

    model = model_creator(config)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())

    lit_model = Trainer.compile(model, loss, optimizer)

    return lit_model