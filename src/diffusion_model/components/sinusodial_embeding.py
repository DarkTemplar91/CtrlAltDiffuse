import torch
import torch.nn as nn
import math


class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_min_frequency: float, embedding_max_frequency: float, embedding_dims: int,
                 device=torch.device('cuda:0')):
        super(SinusoidalEmbedding, self).__init__()
        assert embedding_dims % 2 == 0, "embedding_dims must be an even number"

        self.device = device
        self.embedding_min_frequency = embedding_min_frequency
        self.embedding_max_frequency = embedding_max_frequency
        self.embedding_dims = embedding_dims

        self.frequencies = torch.exp(
            torch.linspace(
                math.log(self.embedding_min_frequency),
                math.log(self.embedding_max_frequency),
                embedding_dims // 2
            )
        ).to(self.device)
        self.angular_speeds = 2.0 * torch.pi * self.frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)

        sin_embeds = torch.sin(self.angular_speeds * x)
        cos_embeds = torch.cos(self.angular_speeds * x)

        embeddings = torch.cat([sin_embeds, cos_embeds], dim=-1)
        return embeddings
