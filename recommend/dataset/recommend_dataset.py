import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RecommenderDataset(Dataset):
    def __init__(self, df:pd.DataFrame):
        """
        Args:
            df (pandas.DataFrame): DataFrame containing columns:
                'userid', 'parent_asin', 'user_embedding', 'item_embedding', 'rating'
        """
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Convert user and item embeddings to PyTorch tensors
        user_emb = torch.tensor(row['user_embedding'], dtype=torch.float32)
        item_emb = torch.tensor(row['item_embedding'], dtype=torch.float32)
        item_emb = item_emb.unsqueeze(0)
        rating = torch.tensor(float(row['rating']), dtype=torch.float32)
        return user_emb, item_emb, rating

# # Assuming your DataFrame is named df:
# dataset = RecommenderDataset(join_user_item)

# # Create a DataLoader from the dataset
# # For training, you typically want to shuffle the data.
# train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# # Example of iterating over the DataLoader
# for user_emb, item_emb, batch_ratings in train_loader:
#     outputs = model(user_emb, item_emb)
#     outputs = outputs.view(-1)
#     batch_ratings = batch_ratings.view(-1)
#     print("Batch features shape:", outputs.shape)
#     print("Batch ratings shape:", batch_ratings.shape)
#     break  # Remove this break to iterate over the entire DataLoader during training
