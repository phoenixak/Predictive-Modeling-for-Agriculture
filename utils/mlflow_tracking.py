import mlflow
import mlflow.sklearn

def log_model_to_mlflow(model, X_train, X_test, y_train, y_test, accuracy):
    """Logs the model and its performance metrics to MLflow."""
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "logistic_regression_model")
        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_metric("accuracy", accuracy)
