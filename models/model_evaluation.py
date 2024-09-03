from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the test set."""
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy, y_pred

def evaluate_features(X_train, y_train, X_test, y_test):
    """Evaluates each feature's predictive power using Logistic Regression."""
    feature_performance = {}
    for feature in ["N", "P", "K", "ph"]:
        log_reg = LogisticRegression(multi_class="multinomial")
        log_reg.fit(X_train[[feature]], y_train)
        y_pred = log_reg.predict(X_test[[feature]])
        f1 = metrics.f1_score(y_test, y_pred, average="weighted")
        feature_performance[feature] = f1
    return feature_performance
