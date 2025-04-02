import pandas as pd
import numpy as np

from recommend.model import embedding_model

def pool(all_embedding):
    all_embedding = np.array(all_embedding)
    return all_embedding.mean(axis=0)

def filter_meta(df_meta: pd.DataFrame = None, columns:list = None):
    if columns == None:
        columns = ["main_category", "title", "average_rating", "parent_asin"]
    df_meta = df_meta[columns]
    return df_meta

def concat_meta(df:pd.DataFrame = None, df_meta: pd.DataFrame = None, verbose = False):
    df_merged = df.merge(df_meta, left_on='parent_asin', right_on='parent_asin', how='left')
    if verbose:
        print("shape of concated: ", df_meta.shape)
    return df_merged

def get_history(df:pd.DataFrame = None):
    df_history = (
        df.groupby(["user_id", "rating"])["title"]
        .apply(lambda x: list(x))  # or use another delimiter if you prefer
        .reset_index(name="history")
    )
    return df_history

def add_embedding(df: pd.DataFrame = None, column_name:str = None, target_name: str = None):
    df[target_name] = df[column_name].apply(lambda x: embedding_model.get_embedding(x))
    return df

def add_embedding_list(df: pd.DataFrame = None, column_name:str = None, target_name: str = None):
    df[target_name] = df[column_name].apply(lambda x: pool([embedding_model.get_embedding(his) for his in x]))
    return df

def default_preprocess(df, df_meta):
    df_meta = filter_meta(df_meta)
    df_merged = concat_meta(df, df_meta)
    return df_merged

def build_user_pivot(train_df_merged):
    train_df_history = get_history(train_df_merged)
    train_df_history = add_embedding_list(train_df_history, "history", "embedding")
    # build user pivot
    user_pivot = train_df_history.pivot(index='user_id', columns='rating', values='embedding')
    user_pivot.columns = [f'embedding_{int(col)}' for col in user_pivot.columns]
    user_pivot = user_pivot.reset_index()
    for i in range(1, 6):
        col = f'embedding_{i}'
        user_pivot[col] = user_pivot[col].apply(
            lambda x: x if isinstance(x, np.ndarray) else np.zeros(384)
        )
    user_pivot['user_embedding'] = user_pivot.apply(
        lambda x: np.array([x[f'embedding_{i}'] for i in range(1, 6)]), 
        axis=1
    )
    return user_pivot

def build_item_embedding(df):
    df_item = (
    df.groupby(["parent_asin"])["title"]
        .last()  # or use another delimiter if you prefer
        .reset_index(name="title")
    )
    df_item = add_embedding(df_item, "item_embedding", "title")
    return df_item

def pipeline(train_df, test_df, meta_df):
    # train
    train_df_merged = default_preprocess(train_df, meta_df)
    user_pivot = build_user_pivot(train_df_merged)
    join_user  = pd.merge(df, user_pivot, on=["user_id"], how="left")
    df_item = build_item_embedding(train_df_merged)
    join_user_item  = pd.merge(join_user, df_item, on=["parent_asin"], how="left")

    # test
    test_df_merged = default_preprocess(test_df, meta_df)
    test_df = build_item_embedding(test_df_merged)
    test_df = test_df.merge(user_pivot, on=['user_id'], how = "left")

    return join_user_item, test_df

if __name__ == "__main__":
    df_meta = pd.read_json("../data/meta_All_Beauty.jsonl", lines=True)
    df = pd.read_csv('../data/All_Beauty.csv')
    default_preprocess(df,df_meta)

