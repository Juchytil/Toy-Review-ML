from utils import load_data, label_sentiment, clean_reviews
from ml_model import train_and_evaluate
import matplotlib.pyplot as plt

def main():
    df = load_data('amazon_baby.csv')
    df = label_sentiment(df)
    df = clean_reviews(df)

    model, metrics = train_and_evaluate(df)

    print("\nModel performance:")
    for k, v in metrics.items():
        if k == "confusion_matrix":
            print(f"\nConfusion Matrix:\n{v}")
        elif k == "roc_auc":
            print(f"{k}: {v:.4f}")
        elif k in ["roc_curve", "top_words"]:
            continue  # skip printing directly
        else:
            print(f"{k}: {float(v):.4f}")

    # Plot ROC Curve
    fpr, tpr = metrics["roc_curve"]
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Print Top Positive and Negative Words
    top_words = metrics.get("top_words")
    if top_words:
        print("\nTop Positive Words:")
        for word in top_words["positive_words"]:
            print(f"  {word}")

        print("\nTop Negative Words:")
        for word in top_words["negative_words"]:
            print(f"  {word}")

if __name__ == "__main__":
    main()
