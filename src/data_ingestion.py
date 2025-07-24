import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging

logging.basicConfig(filename='ingestion.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_params(file_path: str) -> dict:
    with open(file_path, "r") as file:
        params = yaml.safe_load(file)
    return params

def load_dataset(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        logging.info("Dataset loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=['tweet_id'], inplace=True)
    final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
    final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})
    logging.info("Preprocessing completed.")
    return final_df

def split_and_save_dataset(df: pd.DataFrame, test_size: float, train_path: str, test_path: str):
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    logging.info("Data split and saved successfully.")

if __name__ == "__main__":
    params = load_params("params.yaml")
    df = load_dataset('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    processed_df = preprocess_dataset(df)
    split_and_save_dataset(processed_df, params["data_ingestion"]["test_size"], "data/raw/train.csv", "data/raw/test.csv")