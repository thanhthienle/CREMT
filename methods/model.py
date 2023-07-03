import torch.nn as nn
import torch
import torch.nn.functional as F
from .backbone import BertRelationEncoder


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(args.encoder_output_size * 2, args.encoder_output_size, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(args.encoder_output_size, args.encoder_output_size, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(args.encoder_output_size, args.rel_per_task * args.num_tasks),
        ).to(args.device)

    def forward(self, x):
        return self.head(x)
