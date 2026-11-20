import torch.nn as nn
import torch.nn.functional as F

from common import EMBEDDING_SIZE, PADDING


class CaptionTextEncoder(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocabulary_size, EMBEDDING_SIZE, padding_idx=PADDING)
        self.projection = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)

    def forward(self, token_ids):
        # The padding embedding is zero, so summing and dividing by the word count averages the real words.
        word_counts = (token_ids != PADDING).sum(dim=1, keepdim=True).clamp(min=1)
        mean_embedding = self.word_embeddings(token_ids).sum(dim=1) / word_counts
        return F.normalize(self.projection(mean_embedding), dim=-1)
