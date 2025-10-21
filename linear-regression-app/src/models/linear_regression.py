from sklearn.linear_model import LinearRegression
import streamlit as st
import matplotlib.pyplot as plt


class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def plot_regression(self, X, y, x_label, y_label):
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color="blue", alpha=0.5)
        plt.plot(X, self.model.predict(X), color="red", linewidth=2)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("Regresión Lineal")
        st.pyplot(plt)
