import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

def build_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(solver='liblinear'))
    ])

def train_and_evaluate(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'], test_size=0.2, random_state=42
    )
    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    # Top word coefficients
    feature_names = model.named_steps['tfidf'].get_feature_names_out()
    coefficients = model.named_steps['clf'].coef_[0]
    top_pos_idx = np.argsort(coefficients)[-10:]
    top_neg_idx = np.argsort(coefficients)[:10]

    top_words = {
        "positive_words": feature_names[top_pos_idx],
        "negative_words": feature_names[top_neg_idx]
    }

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='binary'),
        "f1_score": f1_score(y_test, y_pred, average='binary'),
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "roc_curve": (fpr, tpr),
        "top_words": top_words
    }

    return model, metrics
