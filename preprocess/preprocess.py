import pandas as pd

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

def default_preprocess(df, df_meta):
    df_meta = filter_meta(df_meta)
    return concat_meta(df, df_meta)

if __name__ == "__main__":
    df_meta = pd.read_json("../data/meta_All_Beauty.jsonl", lines=True)
    df = pd.read_csv('../data/All_Beauty.csv')
    default_preprocess(df,df_meta)

