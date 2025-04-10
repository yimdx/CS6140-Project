import torch
import numpy as np
from metrics.rmse import rmse
def validate(model, test_loader, device, verbose=False):
    pred = []
    true_label = []
    with torch.no_grad():
        for user_emb, item_emb, rating in test_loader:
            # print(user_emb.shape, item_emb.shape)
            user_emb = user_emb.to(device)
            item_emb = item_emb.to(device)
            rating = rating.to(device)
            outputs = model(user_emb, item_emb)
            pred.extend(outputs.tolist())
            true_label.extend(rating.tolist())

    predictions = np.array(pred)
    ground_truths = np.array(true_label)
    rmse = rmse(predictions, ground_truths)
    if verbose:
        print("Test RMSE:", rmse)
    return rmse