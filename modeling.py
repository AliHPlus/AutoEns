from flask import request, jsonify
import pandas as pd
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np 
import matplotlib

matplotlib.use('Agg')

def save_models_to_csv(models, output_directory):
    df = pd.DataFrame.from_dict(models, orient='index', columns=['MSE'])
    df.index.name = 'Ensemble technique'
    filename = f'{output_directory}/models.csv'
    df.to_csv(filename)
    return filename

# Function to calculate Mean Squared Error
def mse(model_name, y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)  # Mean Squared Error
    return mse

# Function to perform ensemble modeling and select the best model
def perform_modeling(dataset_name, label):
    datasets_path = 'datasets/'
    models_path = 'models/'

    dataset = pd.read_csv(os.path.join(datasets_path, dataset_name + '.csv'))

    output_directory = os.path.join(models_path, dataset_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Label encoding for categorical columns
    categorical_columns = dataset.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        dataset[column] = pd.factorize(dataset[column])[0]

    # After label encoding for categorical columns, store the label encoding mappings
    encodings = {}  # Store the label encoding mappings
    for column in categorical_columns:
        encodings[column] = dict(zip(dataset[column], pd.factorize(dataset[column])[0]))

    # Save the label encodings to a file named "encodings.pkl"
    with open(output_directory + '/encodings.pkl', 'wb') as f:
        pickle.dump(encodings, f)

    # Split the dataset into features (X) and the target variable (y)
    X = dataset.drop(label, axis=1)
    y = dataset[label]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize classifiers for Decision Tree, K-Nearest Neighbors, Random Forest, and Logistic Regression
    classifier1 = DecisionTreeClassifier()
    classifier2 = KNeighborsClassifier()
    classifier3 = RandomForestClassifier()
    classifier5 = LogisticRegression()

    # Classifier 1: Decision Tree
    classifier1.fit(X_train, y_train)
    classifier1_pred = classifier1.predict(X_test)

    # Classifier 2: K-Nearest Neighbors
    classifier2.fit(X_train, y_train)
    classifier2_pred = classifier2.predict(X_test)

    # Classifier 3: Random Forest
    classifier3.fit(X_train, y_train)
    classifier3_pred = classifier3.predict(X_test)

    # Voting Ensemble
    voting_classifier = VotingClassifier(estimators=[('DecisionTree', classifier1), ('KNeighbors', classifier2), ('RandomForest', classifier3)], voting='hard')
    voting_classifier.fit(X_train, y_train)
    voting_pred = voting_classifier.predict(X_test)

    # Averaging Ensemble
    averaging_classifier = VotingClassifier(estimators=[('DecisionTree', classifier1), ('KNeighbors', classifier2), ('RandomForest', classifier3)], voting='soft')
    averaging_classifier.fit(X_train, y_train)
    averaging_pred = averaging_classifier.predict(X_test)

    # Weighted Averaging Ensemble
    weighted_classifier = VotingClassifier(estimators=[('DecisionTree', classifier1), ('KNeighbors', classifier2), ('RandomForest', classifier3)], voting='soft', weights=[2, 1, 3])
    weighted_classifier.fit(X_train, y_train)
    weighted_pred = weighted_classifier.predict(X_test)

    # Stacking Ensemble
    stacking_classifier = StackingClassifier(estimators=[('DecisionTree', classifier1), ('KNeighbors', classifier2), ('RandomForest', classifier3)], final_estimator=classifier5)
    stacking_classifier.fit(X_train, y_train)
    stacking_pred = stacking_classifier.predict(X_test)

    # Bagging Ensemble
    bagging_classifier = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
    bagging_classifier.fit(X_train, y_train)
    bagging_pred = bagging_classifier.predict(X_test)

    # Adaboost Ensemble
    adaboost_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
    adaboost_classifier.fit(X_train, y_train)
    adaboost_pred = adaboost_classifier.predict(X_test)

    # Gradient Boosting Ensemble
    gradientboost_classifier = GradientBoostingClassifier(n_estimators=10, random_state=42)
    gradientboost_classifier.fit(X_train, y_train)
    gradientboost_pred = gradientboost_classifier.predict(X_test)

    # Evaluate the performance of each ensemble technique using Mean Squared Error
    models = {
        'Max Voting': mse('Max Voting', y_test, voting_pred),
        'Averaging': mse('Averaging', y_test, averaging_pred),
        'Weighted Averaging': mse('Weighted Averaging', y_test, weighted_pred),
        'Stacking': mse('Stacking', y_test, stacking_pred),
        'Bagging': mse('Bagging', y_test, bagging_pred),
        'AdaBoost': mse('AdaBoost', y_test, adaboost_pred),
        'Gradient Boosting': mse('Gradient Boosting', y_test, gradientboost_pred)
    }

    # Save the models dictionary as CSV
    save_models_to_csv(models, output_directory)

    # Find the model with the lowest MSE (the best model)
    model_name = min(models, key=models.get)
    model_mse = models[model_name]

    # Get the actual model object corresponding to the best model name
    model_object = None
    if model_name == 'Max Voting':
        model_object = voting_classifier
    elif model_name == 'Averaging':
        model_object = averaging_classifier
    elif model_name == 'Weighted Averaging':
        model_object = weighted_classifier
    elif model_name == 'Stacking':
        model_object = stacking_classifier
    elif model_name == 'Bagging':
        model_object = bagging_classifier
    elif model_name == 'AdaBoost':
        model_object = adaboost_classifier
    elif model_name == 'Gradient Boosting':
        model_object = gradientboost_classifier

    # Save the best model to a file named "model.pkl"
    with open(output_directory + '/model.pkl', 'wb') as f:
        pickle.dump(model_object, f)

    # Return the filename of the best model and its corresponding MSE value
    return output_directory + '/model.pkl (Ensemble technique: {}, MSE: {})'.format(model_name, model_mse)