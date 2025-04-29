import torch
import numpy as np
import torch.nn.functional as F

def cosine_dist(embed1, embed2):
        embed1 = embed1[0, :].squeeze()
        embed2 = embed2[0, :].squeeze()
        return 1 - F.cosine_similarity(embed1, embed2, dim=-1)

def l2_dist(embed1, embed2):
    return torch.linalg.vector_norm(embed1 - embed2, ord=2)
