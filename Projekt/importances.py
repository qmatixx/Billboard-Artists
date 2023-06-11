import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Script that calculates importance of features in classification model using Decision Tree Classifier

if __name__ == "__main__":
    data = pd.read_csv('artists.csv')

    label_encoder = LabelEncoder()
    data['Country'] = label_encoder.fit_transform(data['Country'])
    data['Genres'] = label_encoder.fit_transform(data['Genres'])
    data['Gender'] = label_encoder.fit_transform(data['Gender'])

    X = data[['Age', 'Country', 'Genres', 'Number of genres', 'Years active', 'Gender']]
    y = data['Place']

    model = DecisionTreeClassifier()
    model.fit(X, y)

    importance = model.feature_importances_
    for i, feature in enumerate(X.columns):
        print(f'Znaczenie {feature}: {importance[i]}')