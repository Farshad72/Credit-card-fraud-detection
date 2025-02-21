## import libraries
import kagglehub
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

class credit_card_fraud():
    def __init__(self):
        self.kagglehub = kagglehub
        self.pd = pd
        self.os = os
        self.train_test_split = train_test_split
        self.SMOTE = SMOTE
        self.LogisticRegression = LogisticRegression
        self.classification_report = classification_report
        self.confusion_matrix = confusion_matrix
        self.plt = plt

    def load_data(self):
        # Download latest version
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

        print("Path to dataset files:", path)

        csv_file = None
        for file in os.listdir(path):
            if file.endswith(".csv"):
                csv_file = os.path.join(path, file)
                break

        if csv_file:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(csv_file)
            print("Dataset loaded successfully")
            print(df.head())
        else:
            print("CSV file not found in the downloaded dataset")
        return df

    def split_data(self, df):
        X = df.drop('Class', axis=1)
        y = df['Class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        return X_train, X_test, y_train, y_test

    def resample_data(self, X_train, y_train):
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        print("Training set size after resampling:", X_train_resampled.shape)
        print("Class distribution in training set after resampling:", pd.Series(y_train_resampled).value_counts())
        return X_train_resampled, y_train_resampled

    def train_model(self, X_train_resampled, y_train_resampled):
        model = LogisticRegression(random_state=42)
        model.fit(X_train_resampled, y_train_resampled)
        return model

    def evaluate_model(self, model, X_test, y_test):
        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        return y_pred
