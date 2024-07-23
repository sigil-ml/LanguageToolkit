from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

import joblib
import os
from time import time
import numpy as np
import pandas as pd

def replace_acronyms(text, acronym_dict):
    words = text.split()
    replaced_text = ' '.join([acronym_dict.get(word, word) for word in words])
    return replaced_text

def refactor(data):
    def transform_value(value):
        if value == 0:
            return 0
        else:
            return 1
   
    if isinstance(data, pd.Series):
        return data.apply(transform_value)
    elif isinstance(data, np.ndarray):
        return np.array([transform_value(value) for value in data])
    elif isinstance(data, np.int64):
        return transform_value(data)
    else:
        raise ValueError("Input must be a pandas Series, ndarray, or int64")

# Load dataset
test_data = pd.read_csv("src/language_toolkit/tests/data/(CUI) alexa_816th_file_1a1.csv")
acronym_data = pd.read_csv("src/language_toolkit/tests/data/acronyms.csv")
x_train = test_data['Message']  
y_train = refactor(test_data['labels'])

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
# param_grid_rf = {
#     'n_estimators': [10, 50, 100, 200], # 200
#     'max_depth': [None, 10, 20, 30], # None
#     'min_samples_split': [2, 5, 10], # 5
#     'min_samples_leaf': [1, 2, 4], # 1 
#     'bootstrap': [True, False] # False
# }

# Tuned Parameters
param_grid_rf = {
    'n_estimators': [200],
    'max_depth': [None], 
    'min_samples_split': [5],
    'min_samples_leaf': [1], 
    'bootstrap': [False]
}

print("Tuning Random Forest")
rf = RandomForestClassifier()
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1)
grid_search_rf = fit_model(grid_search_rf, x_train_vectorized, y_train)
best_rf = grid_search_rf.best_estimator_
y_pred_rf = refactor(best_rf.predict(x_train_vectorized))
print("Random Forest Accuracy:", accuracy_score(y_train, y_pred_rf))
joblib.dump(best_rf, 'models/best_rf.pkl')

# Logistic Regression
# param_grid_lr = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100], #1
#     'solver': ['newton-cg', 'sag', 'liblinear', 'saga', 'lbfgs'], #'newton-cg'
#     'max_iter': [100, 200, 300, 500] #100
# }

# Tuned Parameters
param_grid_lr = {
    'C': [1],
    'solver': ['newton-cg'],
    'max_iter': [100],
}

print("Tuning Logistic Regression")
lr = LogisticRegression(max_iter=200)
grid_search_lr = GridSearchCV(estimator=lr, param_grid=param_grid_lr, cv=5, n_jobs=-1)
grid_search_lr = fit_model(grid_search_lr, x_train_vectorized, y_train)
best_lr = grid_search_lr.best_estimator_
y_pred_lr = refactor(best_lr.predict(x_train_vectorized))
print("Logistic Regression Accuracy:", accuracy_score(y_train, y_pred_lr))
joblib.dump(best_lr, 'models/best_lr.pkl')

print("Tuning QDA")
qda = QuadraticDiscriminantAnalysis()
grid_search_qda = GridSearchCV(estimator=qda, param_grid={}, cv=5, n_jobs=-1)
grid_search_qda = fit_model(grid_search_qda, x_train_vectorized.toarray(), y_train)
best_qda = grid_search_qda.best_estimator_
y_pred_qda = refactor(best_qda.predict(x_train_vectorized.toarray()))
print("QDA Accuracy:", accuracy_score(y_train, y_pred_qda))
joblib.dump(best_qda, 'models/best_qda.pkl')
