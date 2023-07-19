import torch
from diffusers.models.embeddings import get_timestep_embedding


def get_hour_embedding(
    hours: torch.Tensor, embedding_type: str, emb_size: int = 64
) -> torch.Tensor:
    if embedding_type == "positional":
        hour_emb = get_timestep_embedding(hours.squeeze(), emb_size, max_period=24)
    elif embedding_type == "cyclical":
        hour_emb = torch.stack(
            [
                torch.cos(2 * torch.pi * hours / 24),
                torch.sin(2 * torch.pi * hours / 24),
            ],
            dim=1,
        )
    elif embedding_type in ("class", "timestep"):
        hour_emb = hours
    else:
        hour_emb = None

    return hour_emb
