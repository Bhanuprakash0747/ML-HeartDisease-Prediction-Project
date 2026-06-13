import numpy as np

def preprocess(df):

    df.replace("?", np.nan, inplace=True)

    df = df.apply(pd.to_numeric, errors="coerce")

    df.dropna(inplace=True)

    return df