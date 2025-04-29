from utils import load_data, label_sentiment, clean_reviews
from ml_model import train_and_evaluate

def main():
    df = load_data('amazon_baby.csv')
    df = label_sentiment(df)
    df = clean_reviews(df)

    model, metrics = train_and_evaluate(df)
    print("\nModel performance:")
    for k, v in metrics.items():
        if k == "confusion_matrix":
            print(f"\nConfusion Matrix:\n{v}")
        else:
            print(f"{k}: {float(v):.4f}")
        
if __name__ == "__main__":
    main()
