"""
Script Name: heart_failure.py

Description:
This script develops machine learning models to predict how likely a particular person is in developing cardiovascular diseases.
It reads data from a CSV file and trains three models: Linear Regression, Decision Tree, and a Neural Network.
For each model, it computes evaluation metrics including accuracy, precision, and recall.
After selecting the best-performing model, the script analyzes the most important features contributing to the prediction.

Attribute Information:

01. Age: age of the patient [years]
02. Sex: sex of the patient [M: Male, F: Female]
03. ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
04. RestingBP: resting blood pressure [mm Hg]
05. Cholesterol: serum cholesterol [mm/dl]
06. FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
07. RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation
    or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
08. MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
09. ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
10. Oldpeak: oldpeak = ST [Numeric value measured in depression]
11. ST_Slope: the slope of the peak exercise ST segment [Up: up sloping, Flat: flat, Down: down sloping]
12. HeartDisease: output class [1: heart disease, 0: Normal]

Dependencies:
- pandas
- scikit-learn
- tensorflow

Usage:
$ python3 heart_failure.py

Author: Riccardo Nicolò Iorio
Date: [Add date here]
"""

import pandas as pd
from tabulate import tabulate
from utils_heart import TreeModel

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# Constants
const_num: int = 10
random_state = 42

# Set display options
pd.set_option('display.width', 500)  # Increase the display width
pd.set_option('display.max_columns', None)  # Ensure all columns are displayed

# Models hyperparameters
model_hyperparams = [
    {
        decision_tree_model_params = {
            'criterion': ['entropy'],
            'splitter': ['best'],
            'max_depth': [2 ** x for x in range(6)] + [None],
            'min_samples_split': [2, 10, 30, 50, 100, 200, 300, 600]
            }
    }
]


random_forest_model_params = {
    'n_estimators': [10, 30, 50, 100, 200],
    'criterion': ['entropy'],
    'max_depth': [2 ** x for x in range(6)] + [None],
    'min_samples_split': [2, 10, 30, 50, 100, 200, 300, 600]
}

xgboost_model_params = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.03, 0.06, 0.1],
    'max_depth': [x for x in range(10)],
}

# Model configuration types
model_config = [
    {
        'model_name': 'Decision Tree Classifier',
        'model_type': DecisionTreeClassifier(random_state = 42),
        'params': decision_tree_model_params
    },
    {
        'model_name': 'Random Forest Classifier',
        'model_type': RandomForestClassifier(n_jobs = -1, verbose = 0, random_state = 42),
        'params': random_forest_model_params
    },
    {
        'model_name': 'XGBoost Classifier',
        'model_type': XGBClassifier(objective = 'binary:logistic', eval_metric = 'logloss', random_state = 42),
        'params': xgboost_model_params
    }
]

# Dictionary for containing model type and related accuracy scores
models_scores = {}

# Read the DataFrame from the CSV file
df = pd.read_csv("heart.csv")

# Check relevant information about the data at hand
print(df.info())

# Check if the DataFrame contains Nan values
print(df.isna().sum())

for config in model_config:
    model = TreeModel(config['model_type'], config['params'])
    model.train(df, 'HeartDisease')
    models_scores[config['model_name']] = model.metrics_evaluation()

print(tabulate(pd.DataFrame(models_scores), headers = 'keys', tablefmt = 'grid', stralign = 'center', showindex = True))

# # Check which feature is more correlated to the heart disease
# correlation_matrix = df.corr()
# utils_heart.heatmap(correlation_matrix)
#
#
#
# print(data.head())
#
# # Splitting of data in training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(data, y_target, test_size = 0.3, random_state = random_state)
# print(f"Number of training examples: {len(X_train)} ")
# print(f"Number of test examples: {len(X_test)}")
# try:
#     target_ratio = sum(y_train) / len(y_train)
#     if target_ratio > 0.50:
#         print(f"Target_ratio: {target_ratio:.4f}")
#     else:
#         raise ValueError("The target ratio between training and test cases is too large")
# except ValueError as err:
#     print(f"Error: {err}")
#
# # Logistic Regression Model
# model_dict['Logistic Regression'] = {}
#
# # Decision Trees Models
# model_dict['Decision Tree'] = {}
# # Dictionary containing parameters as the minimum number of samples required to split an internal node and the maximum depth of the tree
# parameters = {
#     'criterion': ['entropy'],
#     'splitter': ['best'],
#     'min_samples_split': [2, 10, 30, 50, 100, 200],
#     'max_depth': [2, 4, 8, 16, 32, 64, None]
# }
#
# # Initialise the Decision Tree Classifier for all possible combinations of the parameters dictionary
# grid_search = GridSearchCV(
#     estimator = DecisionTreeClassifier(random_state = random_state),
#     param_grid = parameters,
#     scoring = 'accuracy',
#     n_jobs = -1,
#     refit = 'accuracy',
#     cv = 2
# )
#
# # Fitting of the training and test example subsets
# grid_search.fit(X_train, y_train)
#
# # Get the best model and test score
# best_model = grid_search.best_estimator_
# best_score = best_model.score(X_train, y_train)
# model_dict['Decision Tree']['training_score'] = f"{best_score:.4f}"
# print(best_model)
#
# # Get the average cross-validation score
# cross_val = cross_validate(best_model, X_train, y_train, cv = 5, scoring = 'accuracy')
# avg_score_val = sum(cross_val['test_score']) / len(cross_val['test_score'])
# model_dict['Decision Tree']['avg_cross_val_score'] = f"{avg_score_val:.4f}"
#
# # Get the accuracy score on the test set given the best model
# y_pred = best_model.predict(X_test)
# test_score = accuracy_score(y_test, y_pred)
# model_dict['Decision Tree']['prediction_score'] = f"{test_score:.4f}"

# # Plot confusion matrix to compute precision, recall and F1 score of the model
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, cmap = 'Blues', annot = True, fmt = 'd', square = True, cbar = False)
# plt.xlabel('Predicted Values')
# plt.ylabel('True Values')
# plt.title('Decision Tree Model')
# plt.show()

# # Initialise the DataFrame of the metrics of all combinations of the parameters dictionary
# results = pd.DataFrame(grid_search.cv_results_)
#
# # Plots of the accuracy versus the minimum number of samples required to split an internal node for fixed tree depth
# plt.figure(figsize = (12, 10))
#
# for i, depth in enumerate(parameters['max_depth']):
#     if depth is None:
#         subset = results[results['param_max_depth'].isna()]
#     else:
#         subset = results[results['param_max_depth'] == depth]
#     plt.plot(subset['param_min_samples_split'], subset['mean_test_score'], label = f"depth = {depth}")
#
# plt.title(f"Training accuracies")
# plt.xlabel("min_samples_split")
# plt.ylabel("accuracy")
# plt.xticks(ticks = parameters['min_samples_split'])
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Neural Network Model
# model_dict['Neural Network'] = {}
#
# # Standardise the features of the dataset using the train subset data (Z-standardisation)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # Neural Network Model definition
# model = Sequential([
#     Input(shape = (data.shape[1],)),
#     Dense(units = 16, activation='relu'),
#     Dense(units = 8, activation = 'relu'),
#     Dense(units = 1, activation='linear')
# ])
#
# model.summary()
#
# #Early stopping function to avoid overfitting
# early_stop = EarlyStopping(
#     monitor = 'val_loss',
#     patience = 5,
#     restore_best_weights = True)
#
# # Compile model
# model.compile(
#     loss = BinaryCrossentropy(from_logits = True),
#     optimizer = Adam(learning_rate = 0.01),
#     metrics = ['accuracy'])
#
# # Fit model
# model.fit(
#     X_train_scaled, y_train,
#     epochs = 100,
#     batch_size = 32,
#     validation_split = 0.2,
#     callbacks = [early_stop],
#     verbose = 0)
#
# # Predict
# y_pred_logits = model.predict(X_test_scaled)
# y_pred_prob = tf.nn.sigmoid(y_pred_logits).numpy()
#
# # Convert probabilities to binary predictions (threshold 0.5)
# y_pred_binary = (y_pred_prob > 0.5).astype(int)
#
# # Neural network accuracy
# nn_accuracy = accuracy_score(y_test, y_pred_binary)

# def main() -> None:
#     print(df)
#
# if __name__ == "__main__":
#     main()