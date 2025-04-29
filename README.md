# Amazon Toy Review Sentiment Classifier

This project analyzes Amazon toy reviews to build a machine learning model that classifies review sentiment as **positive** (ratings 4–5) or **negative** (ratings 1–3).

The goal is to demonstrate end-to-end machine learning workflows using:
- Text preprocessing with TF-IDF
- Logistic regression classification
- Clean modular code structure ready for production

## Workflow
├── Amazon_Toy_Review_ML_Model.ipynb # Original exploratory notebook ├── utils.py # Data loading and cleaning functions ├── my_project_model.py # Model pipeline and training logic ├── train_model.py # Command-line script for training ├── amazon_baby.csv # Dataset file (must be downloaded separately) └── README.md # Project documentation

## Dataset
The dataset contains over 180,000 Amazon toy reviews and can be downloaded from a shared source like Google Drive:

    !gdown 1-kTqWb23FqbnjDKTas4n8UoxtXotWP3z
    
Ensure the CSV is named amazon_toy.csv and placed in the root directory.

## How to run
Once set up, you can train the model directly from the command line:

    python train_model.py

This will:
- Load and clean the data
- Build a TF-IDF + Logistic Regression pipeline
- Train the model
- Print performance metrics (accuracy, precision, F1-score)

## Components overview
utils.py
- load_data(path): Loads the review dataset
- label_sentiment(df): Creates binary sentiment labels
- clean_reviews(df): Removes missing reviews

my_project_model.py

- build_pipeline(): Constructs a text classification pipeline
- train_and_evaluate(df): Splits data, trains model, and returns evaluation metrics

train_model.py

- Script that runs the full training pipeline with minimal setup




    
