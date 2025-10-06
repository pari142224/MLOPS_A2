# Inside misc.py
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# Function 1: Preprocessing (e.g., Scaling)
def preprocess_data(X_train, X_test):
    """Scales the numerical features."""
    # For simplicity, we assume all features need scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Function 2: Split Data
def split_data(df, target_column='MEDV', test_size=0.2, random_state=42):
    """Splits the dataframe into training and testing sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# Function 3: Train Model
def train_model(model, X_train, y_train):
    """Trains a given scikit-learn model."""
    model.fit(X_train, y_train)
    return model

# Function 4: Evaluate Model
def evaluate_model(model, X_test, y_test):
    """Calculates and returns the Mean Squared Error (MSE)."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse
