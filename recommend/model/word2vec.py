from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

class word2vec:
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.model_name)
    
    def get_embedding(self, text):
        return self.model.encode(text)

embedding_model = word2vec()