from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    """Trains a Logistic Regression model."""
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(X_train, y_train)
    return model
