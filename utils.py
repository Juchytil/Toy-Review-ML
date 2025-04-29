import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def label_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df['sentiment'] = df['rating'].apply(lambda r: 1 if r >= 4 else 0)
    return df

def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['review'])
    return df
