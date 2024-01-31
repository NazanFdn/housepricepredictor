import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import time

start = time.time()

# read the dataset
read_data = pd.read_csv('HousePricesData.csv')

# Encoding(Replacing strings with integers like 0, 1, 2 for accessing those expecting numerical input)
label_encoder_type = LabelEncoder()
label_encoder_district = LabelEncoder()
label_encoder_city = LabelEncoder()
label_encoder_postcode = LabelEncoder()

read_data['type'] = label_encoder_type.fit_transform(read_data['type'])
read_data['district'] = label_encoder_district.fit_transform(read_data['district'])
read_data['city'] = label_encoder_city.fit_transform(read_data['city'])
read_data['postcode'] = label_encoder_postcode.fit_transform(read_data['postcode'])


# Split the data into features (X) and target (y)
data = read_data.drop(columns=['name', 'no'])
X = data.drop(columns=['price'])
y = data['price']

# scikit-learn is used to normalize the data before training
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Split the data into training and testing sets (%20 for testing and %80 for training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# model building (sequential model) -linear layer
build_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),   # The activation function Rectified Linear Unit (ReLU)
    tf.keras.layers.Dense(1)  # Output layer with a single neuron (for regression)
])

build_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])    # Configures the model for training.
history = build_model.fit(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_test, y_test))  # Model trained for 80 epochs,

house_id = label_encoder_type.transform(['House'])
district_id = label_encoder_district.transform(['Shoreditch'])
city_id = label_encoder_city.transform(['London'])
postcode_id = label_encoder_postcode.transform(['E2 7BX'])

new_data = np.array([[house_id[0], 1995, 5, 5, 5, district_id[0], city_id[0], postcode_id[0]]])  # Example house features to predict
new_data = scaler.transform(new_data)  # normalize the new data
predicted_price = build_model.predict(new_data)

# R-squared (R2) score
y_predict = build_model.predict(X_test)
r_squared = r2_score(y_test, y_predict)
# Mean squared error:
mean_squared = mean_squared_error(y_test, y_predict)
# Mean absolute error
test_loss, test_mae = build_model.evaluate(X_test, y_test)
end = time.time()

train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot training and validation loss across epochs
plt.plot(range(1, 81), train_loss, label='Training Loss', marker='o')   # 80 points to be marked on the chart
plt.plot(range(1, 81), val_loss, label='Validation Loss', marker='x')   # 80 points to be marked on the chart
plt.xlabel('Iteration')     # x-axis showing iteration
plt.ylabel('Mean Squared Error')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

x_series = []
prices = []
predicted_prices = []

for i in range(20):     # Comparison will be made for 20 houses
    x_series.append(i)
    prices.append(data['price'][i])     # adding the price to the list. And adding all the features...
    house_size = data['area'][i]
    house_bedroom = data['bedrooms'][i]
    house_bathrooms = data['bathrooms'][i]
    receptions = data['receptions'][i]
    house_id = data['type'][i]
    district_id = data['district'][i]
    city_id = data['city'][i]
    postcode_id = data['postcode'][i]
    new_data = np.array([[house_id, house_size, house_bedroom, house_bathrooms, receptions, district_id, city_id, postcode_id]])  # Example house features to predict
    new_data = scaler.transform(new_data)  # normalize the new data
    predicted_price = build_model.predict(new_data)     # Predict the prices
    predicted_prices.append(predicted_price[0][0])

#There will be 2 different price for one house: Predicted and actual price. So there will be 2 y-series and one x-series in the graph.
# Plotting the first y-series against the x-series
plt.plot(x_series, prices, label='Price', marker='o')

# Plotting the second y-series against the same x-series
plt.plot(x_series, predicted_prices, label='Prediction', marker='x')

# Adding labels
plt.xlabel('Prop #')
plt.ylabel('Price (£ x1,000,000)')  # show the price in millions
plt.title('Actual Price vs Prediction')
plt.legend()
plt.show()

# Print the performance metrics, predicted price and execution ime
print("R-Square (R2)", r_squared)
print("Mean Squared Error", mean_squared)
print("Test Mean Absolute Error", test_mae)  # The average absolute difference between actual values and the predicted values
print(f"Predicted Price of the House: {predicted_price[0][0]}")
print("Runtime:", end-start)
