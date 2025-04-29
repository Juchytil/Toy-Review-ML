from utils import load_data, label_sentiment, clean_reviews
from ml_model import train_and_evaluate

def main():
    df = load_data('amazon_baby.csv')
    df = label_sentiment(df)
    df = clean_reviews(df)

    model, metrics = train_and_evaluate(df)
    print("Model performance:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
