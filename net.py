
import torch
from torch import Tensor, tensor
from torch.nn import Embedding, Module, TransformerEncoderLayer, TransformerEncoder
import numpy as np


def make_triu(m: int, device: torch.device): # функция создания треугольной матрицы для маски
    a = torch.full(size=(m, m), fill_value=False, device=device)
    a[np.triu_indices(m, 1)] = True
    return a


class Grokformer(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device):
        super(Grokformer, self).__init__()
        self.embedding = Embedding(num_embeddings=num_embeddings + 2, embedding_dim=embedding_dim, device=device)
        self.n_embeddings = num_embeddings + 2
        layer = TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, dropout=0., batch_first=True)
        self.network = TransformerEncoder(encoder_layer=layer, num_layers=2)
        self.network.to(device)
        self.pos_idx = tensor([num_embeddings + k for k in range(2)], device=device)
        self.device = device
        self.mask = make_triu(2, self.device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Прямой проход
        x: тензор размера (n, 4).
        return: тензор размера (n, num_embeddings).
        """
        n, _ = x.shape
        pos = self.embedding(self.pos_idx)  # pos: (2, embedding_dim)
        pos = pos.view((1, -1))             # pos: (1, 2*embedding_dim)
        pos = pos.repeat((n, 1))            # pos: (n, 2*embedding_dim)

        emb = self.embedding(x)             # emb: (n, 2, embedding_dim)
        emb = emb.view((n, -1))             # emb: (n, 2*embedding_dim)

        src = emb + pos                  # src: (n, 2*embedding_dim)
        src = src.view((n, 2, -1))       # src: (n, 2, embedding_dim)

        res = self.network.forward(src=src, mask=self.mask)     # res: (n, 2, embedding_dim)
        res = res[:, -1, :]             # res: (n, 1, embedding_dim)
        res = res.view((n, -1))         # res: (n, embedding_dim)

        scores = res @ self.embedding.weight[:-2, :].t()    # scores: (n, num_embeddings)

        return scores
