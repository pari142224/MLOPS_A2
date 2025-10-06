# Inside train.py
from sklearn.tree import DecisionTreeRegressor
# Import from the file containing load_data (e.g., data.py or misc.py)
# Assuming load_data is in data.py for this example
from data import load_data 
from misc import split_data, preprocess_data, train_model, evaluate_model

if __name__ == "__main__":
    # 1. Data Loading
    df = load_data()

    # 2. Data Splitting
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Data Preprocessing (Scaling)
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

    # 4. Model Training
    dt_model = DecisionTreeRegressor(random_state=42)
    trained_dt_model = train_model(dt_model, X_train_scaled, y_train)

    # 5. Model Evaluation
    mse = evaluate_model(trained_dt_model, X_test_scaled, y_test)

    # Display result
    print(f"Decision Tree Regressor - Average MSE on Test Set: {mse:.4f}") # [cite: 20]
