from data.data_loader import load_data
from models.model_training import train_model
from models.model_evaluation import evaluate_model, evaluate_features
from utils.logger import setup_logger
from utils.mlflow_tracking import log_model_to_mlflow
from sklearn.model_selection import train_test_split

def main():
    logger = setup_logger()

    # Load data
    logger.info("Loading data...")
    file_path = "data set/soil_measures.csv"
    crops = load_data(file_path)

    # Split data into features and target
    X = crops.drop('crop', axis=1)
    y = crops['crop']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    logger.info("Training model...")
    model = train_model(X_train, y_train)

    # Evaluate model
    logger.info("Evaluating model...")
    accuracy, y_pred = evaluate_model(model, X_test, y_test)
    logger.info(f"Model Accuracy: {accuracy}")

    # Log model to MLflow
    log_model_to_mlflow(model, X_train, X_test, y_train, y_test, accuracy)

    # Evaluate features
    logger.info("Evaluating features...")
    feature_performance = evaluate_features(X_train, y_train, X_test, y_test)
    best_feature = max(feature_performance, key=feature_performance.get)
    logger.info(f"Best Predictive Feature: {best_feature} with F1-score: {feature_performance[best_feature]}")

if __name__ == "__main__":
    main()
