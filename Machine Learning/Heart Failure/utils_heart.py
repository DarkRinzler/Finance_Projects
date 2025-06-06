from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Input, Dense
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import BinaryCrossentropy
# from tensorflow.keras.callbacks import EarlyStopping


class TreeModel:
    def __init__(self, model, params_grid: Dict[str, Any]):
        self.model = model
        self.params_grid = params_grid

    # Function to apply pre_processing of the data
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()

        # Identification of the categorical columns to be one-hot coded
        categorical_cols = data.select_dtypes(include = ['object']).columns
        return pd.get_dummies(data = data, prefix = categorical_cols, columns = categorical_cols, dtype = int)

    def train(self, data: pd.DataFrame, target_col: str) -> None:
        # Select target column and features column for training
        x = data.drop(columns = [target_col], axis = 1)
        target = data[target_col]

        # Pre_process the features columns to have only integer values
        x_processed = self.preprocess(x)

        # Splitting of data in training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x_processed, target, test_size = 0.3, random_state = 42)

        # Initialise the Decision Tree Classifier for all possible combinations of the parameters dictionary
        search = GridSearchCV(
            estimator = self.model,
            param_grid = self.params_grid,
            scoring = 'accuracy',
            n_jobs = -1,
            refit = 'accuracy',
            cv = 2
        )

        # Fitting of the model on the split pre-processed data and storing of the best model
        search.fit(x_train, y_train)
        self.model = search.best_estimator_

        self.data_splits = (x_train, x_test, y_train, y_test)

    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        x_new = self.preprocess(new_data)
        return self.model.predict(x_new)

    def metrics_evaluation(self) -> Dict[str, float]:
        # Unpacking of tuple values
        x_train, x_test, y_train, y_test = self.data_splits

        # Predictions of the model
        y_pred = self.model.predict(x_test)

        # Train set accuracy score
        train_score = self.model.score(x_train, y_train)

        # Cross-validation accuracy score
        cross_val = cross_validate(self.model, x_train, y_train, cv = 5, scoring = 'accuracy')
        cross_val_score = sum(cross_val['test_score']) / len(cross_val['test_score'])

        # Test set accuracy score
        test_score = accuracy_score(y_test, y_pred)

        # Precision score
        report = classification_report(y_test, y_pred, output_dict = True)
        precision_score = report['1']['precision']
        # Recall score
        recall_score = report['1']['recall']

        # F1 score
        f1_score = report['1']['f1-score']

        # Save all scores in an instance dictionary
        self.metrics = {
            'Train_accuracy': f'{train_score:.2%}',
            'Cv_accuracy': f'{cross_val_score:.2%}',
            'Test_accuracy': f'{test_score:.2%}',
            'Precision': f'{precision_score:.2%}',
            'Recall': f'{recall_score:.2%}',
            'F1_score': f'{f1_score:.2%}'
        }
        return self.metrics
