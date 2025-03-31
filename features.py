import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data
df = pd.read_csv('All_Beauty.csv')
# df = df.dropna(subset=['user_id', 'item_id', 'review_text'])

# Aggregate item features
item_features = df.groupby('item_id')['review_text'].apply(' '.join).reset_index()

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
item_vectors = tfidf.fit_transform(item_features['review_text'])

# Create user profiles (example for one user)
user_ratings = df[df['user_id'] == 'USER123'][['item_id', 'rating']]
user_item_indices = user_ratings['item_id'].map(item_features['item_id'].index.tolist())
weights = user_ratings['rating'].values
weighted_vectors = item_vectors[user_item_indices].multiply(weights[:, None])
user_profile = weighted_vectors.sum(axis=0)

# Generate recommendations
similarities = cosine_similarity(user_profile, item_vectors)
top_items = np.argsort(similarities[0])[::-1][:10]
recommended_products = item_features['item_id'].iloc[top_items]