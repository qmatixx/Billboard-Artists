from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Script that calculates importance of features in classification model using Decision Tree Classifier, but with more details about categorical features

if __name__ == "__main__":
    data = pd.read_csv('artists.csv', index_col=[0])

    X = data[['Age', 'Country', 'Genres', 'Number of genres', 'Years active', 'Gender']]
    y = data['Place']

    X_encoded = pd.get_dummies(X, columns=['Country', 'Genres', 'Gender'])
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    feature_importances = clf.feature_importances_
    feature_names = X_encoded.columns

    for feature_name, importance in zip(feature_names, feature_importances):
        print(feature_name, ":", importance)