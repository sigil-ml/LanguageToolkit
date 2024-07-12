import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.exceptions import ConvergenceWarning
import joblib
import os
from time import time
import warnings

warnings.filterwarnings('ignore', category=ConvergenceWarning)


def replace_acronyms(text, acronym_dict):
    words = text.split()
    replaced_text = ' '.join([acronym_dict.get(word, word) for word in words])
    return replaced_text

# Load dataset
test_data = pd.read_csv("src/language_toolkit/tests/data/(CUI) alexa_816th_file_1a1.csv")
acronym_data = pd.read_csv("src/language_toolkit/tests/data/acronyms.csv")
x_train = test_data['Message']  
y_train = test_data['labels']  

# Create a dictionary for acronyms
acronyms = acronym_data['Jargon'].tolist()
acronym_meaning = acronym_data['Meaning'].tolist()
acronym_dict = dict(zip(acronyms, acronym_meaning))

# Replace acronyms in x_train
x_train = x_train.apply(lambda text: replace_acronyms(text, acronym_dict))

# Vectorize the text data
vectorizer = CountVectorizer()
x_train_vectorized = vectorizer.fit_transform(x_train)

os.makedirs("models", exist_ok=True)

# Function to fit model and measure time
def fit_model(grid_search, X, y):
    start_time = time()
    grid_search.fit(X, y)
    end_time = time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    return grid_search

# Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

print("Tuning Random Forest")
rf = RandomForestClassifier()
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1)
grid_search_rf = fit_model(grid_search_rf, x_train_vectorized, y_train)
best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(x_train_vectorized)
print("Random Forest Accuracy:", accuracy_score(y_train, y_pred_rf))
joblib.dump(best_rf, 'models/best_rf.pkl')

# Logistic Regression
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 300, 500]
}

print("Tuning Logistic Regression")
lr = LogisticRegression()
grid_search_lr = GridSearchCV(estimator=lr, param_grid=param_grid_lr, cv=5, n_jobs=-1)
grid_search_lr = fit_model(grid_search_lr, x_train_vectorized, y_train)
best_lr = grid_search_lr.best_estimator_
y_pred_lr = best_lr.predict(x_train_vectorized)
print("Logistic Regression Accuracy:", accuracy_score(y_train, y_pred_lr))
joblib.dump(best_lr, 'models/best_lr.pkl')

# Naive Bayes
param_grid_nb = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
}

print("Tuning Naive Bayes")
nb = GaussianNB()
grid_search_nb = GridSearchCV(estimator=nb, param_grid=param_grid_nb, cv=5, n_jobs=-1)
grid_search_nb = fit_model(grid_search_nb, x_train_vectorized.toarray(), y_train)
best_nb = grid_search_nb.best_estimator_
y_pred_nb = best_nb.predict(x_train_vectorized.toarray())
print("Naive Bayes Accuracy:", accuracy_score(y_train, y_pred_nb))
joblib.dump(best_nb, 'models/best_nb.pkl')
