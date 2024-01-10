from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

def load_and_prepare_data():
    """
    Loads the digits dataset and reshapes the image data for analysis.

    Returns:
        tuple: A tuple containing the reshaped data and corresponding targets.
    """
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data, digits.target

def train_test_split_data(data, target):
    """
    Splits the data into training and testing subsets.

    Args:
        data (array): The input data to split.
        target (array): The target labels associated with the data.

    Returns:
        tuple: A tuple containing the split training and testing data and labels.
    """
    return train_test_split(data, target, test_size=0.5, shuffle=False)

def train_and_evaluate_classifier(X_train, y_train, X_test, y_test):
    """
    Trains a Support Vector Classifier on the training data and evaluates it on the test data.

    Args:
        X_train (array): Training data.
        y_train (array): Training labels.
        X_test (array): Testing data.
        y_test (array): Testing labels.

    Returns:
        svm.SVC: The trained classifier.
        array: The predictions made by the classifier on the test data.
    """
    clf = svm.SVC(gamma=0.001)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    return clf, predicted

def print_classification_report(clf, y_test, predicted):
    """
    Prints the classification report for the classifier.

    Args:
        clf (svm.SVC): The trained classifier.
        y_test (array): The true labels for the test data.
        predicted (array): The predicted labels for the test data.
    """
    print(f"Classification report for classifier {clf}:\n"
          f"{metrics.classification_report(y_test, predicted)}\n")

if __name__ == '__main__':
    data, target = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split_data(data, target)
    clf, predicted = train_and_evaluate_classifier(X_train, y_train, X_test, y_test)
    print_classification_report(clf, y_test, predicted)