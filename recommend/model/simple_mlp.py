import torch.nn as nn
import torch

class SimplestRecommender(nn.Module):
    def __init__(self, embed_dim=16, hidden_dim=64):
        super().__init__()
        # We'll have 5 user-emb vectors + 1 item-emb vector = 6 vectors total
        # Each vector is embed_dim in length => final input dimension = 6 * embed_dim
        self.fc1 = nn.Linear(6 * embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, user_embs, item_emb):
        """
        user_embs: (batch_size, 5, embed_dim)
        item_emb:  (batch_size, 1, embed_dim)
        """
        # Concatenate user & item embeddings along dimension=1 => shape (batch_size, 6, embed_dim)
        combined = torch.cat([user_embs, item_emb], dim=1)

        # Flatten => shape (batch_size, 6*embed_dim)
        combined = combined.view(combined.size(0), -1)

        # Pass through MLP
        x = self.relu(self.fc1(combined))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)  # shape (batch_size, 1)
        return out


# Instantiate the model
recommand_model = SimplestRecommender(embed_dim=384, hidden_dim=128)



