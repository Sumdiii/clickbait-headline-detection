# Clickbait Headline Detection Using Machine Learning

## Dataset
This project uses the exact Kaggle dataset selected for the project:
https://www.kaggle.com/datasets/amananandrai/clickbait-dataset

Published descriptions of this dataset report 32,000 news headlines:
- `Headline`: headline text
- `Clickbait`: binary label
  - 1 = clickbait
  - 0 = non-clickbait

The headlines were collected from multiple news sources. The project does not add another dataset.

## Project workflow
Dataset -> Cleaning -> EDA -> Text preprocessing -> TF-IDF -> ML models -> Evaluation -> Best model -> Streamlit dashboard

## Models
1. Logistic Regression
2. Multinomial Naive Bayes
3. Random Forest
4. Linear SVM

## Folder setup
Download the Kaggle dataset and put the CSV file inside:

data/raw/

The code automatically searches for CSV files and identifies the `Headline` and `Clickbait` columns.

## Run in VS Code

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python prepare_data.py
python train_models.py
streamlit run app.py
```

If NLTK stopword data is missing, the scripts automatically download it.

## Dashboard
The Streamlit app lets you:
- enter a headline
- predict Clickbait / Not Clickbait
- see confidence/probability
- compare model performance
- view dataset statistics
- view the confusion matrix
- view important words

## Academic note
The model learns patterns present in this dataset. A prediction is not a factual determination that a headline is deceptive; it is a machine-learning classification based on the training data.
