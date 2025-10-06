import warnings
 warnings.filterwarnings("ignore")
 from sklearn.tree import DecisionTreeRegressor
 from misc import load_data, prepare_xy, split_data, train_model, evaluate_model, cross_val_mse,
save_model
 def main():
 df = load_data()
 X, y = prepare_xy(df)
 X_train, X_test, y_train, y_test = split_data(X, y)
 model = DecisionTreeRegressor(random_state=42)
 model = train_model(model, X_train, y_train)
 test_mse = evaluate_model(model, X_test, y_test)
 cv_mse = cross_val_mse(DecisionTreeRegressor(random_state=42), X, y)
 print("DecisionTreeRegressor")
 print(test_mse)
 print(cv_mse)
 save_model(model, "dtree_model.joblib")
 if __name__ == "__main__":
 main()