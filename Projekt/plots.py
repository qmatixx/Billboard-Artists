import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder

# Sciprt that plots data and polynomial regression curves for each feature

if __name__ == "__main__":
    
    data = pd.read_csv('artists.csv')
    
    # Encoding categorical data
    label_encoder_country = LabelEncoder()
    data['Country'] = label_encoder_country.fit_transform(data['Country'])
    label_encoder_genres = LabelEncoder()
    data['Genres'] = label_encoder_genres.fit_transform(data['Genres'])
    label_encoder_gender = LabelEncoder()
    data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])

    # Plotting data and polynomial regression curves for integer features
    features = ['Age', 'Number of genres', 'Years active']
    for feature in features:
        plt.figure()
        X = data[feature].values.reshape(-1, 1)
        y = data['Place'].values.reshape(-1, 1)
        polynomial_features = PolynomialFeatures(degree=10)
        X_poly = polynomial_features.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_range_poly = polynomial_features.transform(X_range)
        y_pred = model.predict(X_range_poly)
        plt.scatter(X, y, label='Dane')
        plt.plot(X_range, y_pred, color='red', label='Regresja wielomianowa')
        plt.xlabel(feature)
        plt.ylabel('Place')
        plt.title('Zależność Place od {}'.format(feature))
        plt.legend()
        plt.show()

    # Plotting data and polynomial regression for countries
    plt.figure()
    X = data['Country'].values.reshape(-1, 1)
    y = data['Place'].values.reshape(-1, 1)
    polynomial_features = PolynomialFeatures(degree=10)
    X_poly = polynomial_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_range_poly = polynomial_features.transform(X_range)
    y_pred = model.predict(X_range_poly)
    plt.scatter(X, y, label='Dane')
    plt.plot(X_range, y_pred, color='red', label='Regresja wielomianowa')
    plt.xlabel('Country')
    plt.ylabel('Place')
    plt.title('Zależność Place od Country')
    countries = set(data['Country'].values)
    legend_elements = []
    for i in countries:
        legend_elements.append(plt.Line2D([0], [0], color='w', marker='o', markerfacecolor='w', label=f"{i}. {label_encoder_country.inverse_transform([i])[0]}"))
    plt.legend(handles=legend_elements)
    plt.show()

    # Plotting data and polynomial regression for genres
    plt.figure()
    X = data['Genres'].values.reshape(-1, 1)
    y = data['Place'].values.reshape(-1, 1)
    polynomial_features = PolynomialFeatures(degree=10)
    X_poly = polynomial_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_range_poly = polynomial_features.transform(X_range)
    y_pred = model.predict(X_range_poly)
    plt.scatter(X, y, label='Dane')
    plt.plot(X_range, y_pred, color='red', label='Regresja wielomianowa')
    plt.xlabel('Genres')
    plt.ylabel('Place')
    plt.title('Zależność Place od Genres')
    genres = set(data['Genres'].values)
    legend_elements = []
    for i in genres:
        legend_elements.append(plt.Line2D([0], [0], color='w', marker='o', markerfacecolor='w', label=f"{i}. {label_encoder_genres.inverse_transform([i])[0]}"))
    plt.legend(handles=legend_elements)
    plt.show()

    # Plotting data and polynomial regression for gender
    plt.figure()
    X = data['Gender'].values.reshape(-1, 1)
    y = data['Place'].values.reshape(-1, 1)
    polynomial_features = PolynomialFeatures(degree=10)
    X_poly = polynomial_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_range_poly = polynomial_features.transform(X_range)
    y_pred = model.predict(X_range_poly)
    plt.scatter(X, y, label='Dane')
    plt.plot(X_range, y_pred, color='red', label='Regresja wielomianowa')
    plt.xlabel('Gender')
    plt.ylabel('Place')
    plt.title('Zależność Place od Gender')
    genders = set(data['Gender'].values)
    legend_elements = []
    for i in genders:
        legend_elements.append(plt.Line2D([0], [0], color='w', marker='o', markerfacecolor='w', label=f"{i}. {label_encoder_gender.inverse_transform([i])[0]}"))
    plt.legend(handles=legend_elements)
    plt.show()