# Linear Regression App

This project is a Streamlit application that implements a linear regression model. It allows users to upload CSV files containing data, which can then be used to train the model and make predictions.

## Features

- Upload CSV files to input data.
- Train a linear regression model on the uploaded data.
- Make predictions based on the trained model.
- Restore values from previously uploaded data.

## Project Structure

- `src/app.py`: Entry point of the application.
- `src/models/linear_regression.py`: Contains the `LinearRegressionModel` class.
- `src/utils/data_loader.py`: Function to load data from CSV files.
- `src/utils/preprocessing.py`: Functions for data preprocessing.
- `src/config/settings.py`: Configuration settings for the application.
- `data/.gitkeep`: Keeps the data directory in version control.
- `tests/test_model.py`: Unit tests for the linear regression model.
- `requirements.txt`: Lists project dependencies.

## Requirements

To run this application, you need to install the required dependencies. You can do this by running:

```
pip install -r requirements.txt
```

## Running the Application

To start the application, run the following command:

```
streamlit run src/app.py
```

## License

This project is licensed under the MIT License.