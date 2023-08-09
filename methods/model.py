import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from .backbone import BertRelationEncoder


class ETFLinear(nn.Module):
    def __init__(self, feat_in, feat_out, device='cuda', dtype=torch.float32):
        super().__init__()
        P = self._generate_random_orthogonal_matrix(feat_in, feat_out)
        I = torch.eye(feat_out)
        one = torch.ones(feat_out, feat_out)
        M = np.sqrt(feat_out / (feat_out-1)) * \
            torch.matmul(P, I-((1/feat_out) * one))
        self.M = M.to(device)
    
        self.InstanceNorm = nn.InstanceNorm1d(feat_in, affine=False, device=device)
        self.BatchNorm = nn.BatchNorm1d(feat_in, affine=False, device=device)

    def _generate_random_orthogonal_matrix(self, feat_in, feat_out):
        a = np.random.random(size=(feat_in, feat_out))
        P, _ = np.linalg.qr(a)  # This function returns an orthonormal matrix (q) and an upper-triangle matrix r(q)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(feat_out), atol=1e-07), \
            torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(feat_out)))
        return P

    def forward(self, x):
        if x.shape[0] != 1:
            return self.BatchNorm(x) @ self.M
        else:
            return self.InstanceNorm(x) @ self.M


class Classifier(nn.Module):
    def __init__(self, args, final_linear=None):
        top_linear = final_linear if final_linear is not None else nn.Linear(args.encoder_output_size, args.rel_per_task * args.num_tasks)
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(args.encoder_output_size * 2, args.encoder_output_size, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(args.encoder_output_size, args.encoder_output_size, bias=True),
            nn.ReLU(inplace=True),
            top_linear,
        ).to(args.device)

    def forward(self, x: torch.Tensor):
        return self.head(x)
