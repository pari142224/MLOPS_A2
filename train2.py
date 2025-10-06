# Inside train2.py
from sklearn.kernel_ridge import KernelRidge
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
    # Use a common kernel like 'rbf' for KernelRidge
    kr_model = KernelRidge(kernel='rbf', alpha=1.0) 
    trained_kr_model = train_model(kr_model, X_train_scaled, y_train)

    # 5. Model Evaluation
    mse = evaluate_model(trained_kr_model, X_test_scaled, y_test)

    # Display result
    print(f"Kernel Ridge Regressor - Average MSE on Test Set: {mse:.4f}") # [cite: 25]
