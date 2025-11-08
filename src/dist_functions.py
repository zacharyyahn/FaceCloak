import torch
import numpy as np
import torch.nn.functional as F

def cosine_dist(embed1, embed2):
    embed1 = F.normalize(embed1, dim=1)
    embed2 = F.normalize(embed2, dim=1).cuda()
    sims = torch.mm(embed1, embed2.t()).squeeze(1)

    return -sims  # smaller = closer

def l2_dist(embed1, embed2):
    return torch.linalg.vector_norm(embed1 - embed2, ord=2)
