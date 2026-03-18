#  Smart AQI Forecasting System using LSTM

##  Overview

The **Smart AQI Forecasting System** is a machine learning-based web application that predicts future Air Quality Index (AQI) values using a Long Short-Term Memory (LSTM) neural network.

The system fetches real-time PM2.5 data from the Open-Meteo API, converts it into AQI values, and provides a **7-day forecast**, along with **model evaluation metrics, visualizations, and geospatial insights**.

---

## Features

###  7-Day AQI Forecast

* Predicts AQI for the next 7 days
* Helps users plan activities based on air quality

###  Model Evaluation

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* Evaluates prediction accuracy

###  AQI Trend & Forecast Visualization

* Historical AQI trends
* Forecast values plotted alongside real data

###  AQI Category Distribution

* Shows frequency of AQI categories
* Helps analyze pollution patterns

### India AQI Map

* Interactive map using PyDeck
* Displays AQI of major Indian cities

### Satellite Location Visualization

* Shows selected city location on map

###  AQI Classification Table

* AQI ranges with health impact information


## Working Flow

1. Fetch PM2.5 data from Open-Meteo API
2. Convert PM2.5 to AQI
3. Preprocess and normalize data
4. Train LSTM model
5. Predict future AQI values
6. Evaluate model performance
7. Visualize results and maps

---

##  Technologies Used

* Python
* Streamlit
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* TensorFlow (Keras LSTM)
* PyDeck
* Open-Meteo API


## Before vs After Improvements

### Before

* 3-day forecast
* No evaluation metrics
* Limited visualization

### After

* 7-day forecast
* Added MAE & RMSE
* AQI trend graphs
* Category distribution
* India AQI map
* Satellite visualization
* Improved UI

---


