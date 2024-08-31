import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
heart_data = pd.read_csv(r"C:\Users\SRUJANA\Downloads\heartdiseasepred.csv")

# Display the first few rows of the dataset
print(heart_data.head())

# Display the shape of the dataset
print(heart_data.shape)

# Check for any missing values
print(heart_data.isnull().values.any())

# Display the column names
print(heart_data.columns)

# Convert the 'Heart Disease' column to binary (0 or 1)
output = pd.get_dummies(heart_data['Heart Disease'], drop_first=True)
heart_data['Heart Disease'] = output

# Display the first few rows of the updated dataset
print(heart_data.head())

# Separate features and target
x = heart_data[['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120',
                'EKG results', 'Max HR', 'Exercise angina', 'ST depression',
                'Slope of ST', 'Number of vessels fluro', 'Thallium']].values
y = heart_data[['Heart Disease']].values

# Standardize the feature data
scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)

# Update x and y
x = standardized_data
y = heart_data['Heart Disease']

# Print the standardized feature data and target data
print(x)
print(y)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, stratify=y, random_state=2)

# Print the shape of the data splits
print(x.shape, x_train.shape, x_test.shape)

# Train the SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

# Calculate and print the training accuracy for SVM
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print("SVM Training Accuracy:", training_data_accuracy)

# Calculate and print the test accuracy for SVM
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print("SVM Test Accuracy:", test_data_accuracy)

# Train the Logistic Regression model (Optional)
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_training_accuracy = lr.score(x_train, y_train)
print("Logistic Regression Training Accuracy:", lr_training_accuracy)

# Train the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=2)
rf_classifier.fit(x_train, y_train)

# Calculate and print the training accuracy for Random Forest
rf_train_prediction = rf_classifier.predict(x_train)
rf_training_data_accuracy = accuracy_score(rf_train_prediction, y_train)
print("Random Forest Training Accuracy:", rf_training_data_accuracy)

# Calculate and print the test accuracy for Random Forest
rf_test_prediction = rf_classifier.predict(x_test)
rf_test_data_accuracy = accuracy_score(rf_test_prediction, y_test)
print("Random Forest Test Accuracy:", rf_test_data_accuracy)

# Train the Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier(random_state=2)
gb_classifier.fit(x_train, y_train)

# Calculate and print the training accuracy for Gradient Boosting
gb_train_prediction = gb_classifier.predict(x_train)
gb_training_data_accuracy = accuracy_score(gb_train_prediction, y_train)
print("Gradient Boosting Training Accuracy:", gb_training_data_accuracy)

# Calculate and print the test accuracy for Gradient Boosting
gb_test_prediction = gb_classifier.predict(x_test)
gb_test_data_accuracy = accuracy_score(gb_test_prediction, y_test)
print("Gradient Boosting Test Accuracy:", gb_test_data_accuracy)

# Example prediction with new input data using Random Forest
input_data = (57, 1, 2, 124, 261, 0, 0, 141, 0, 0.3, 1, 0, 7)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
standardized_data = scaler.transform(input_data_reshaped)

# Predict with Random Forest
rf_prediction = rf_classifier.predict(standardized_data)
print("Random Forest Prediction:", rf_prediction)

if rf_prediction[0] == 0:
    print('Person has heart disease')
else:
    print('Person does not have heart disease')

# Predict with Gradient Boosting
gb_prediction = gb_classifier.predict(standardized_data)
print("Gradient Boosting Prediction:", gb_prediction)

if gb_prediction[0] == 0:
    print('Person has heart disease')
else:
    print('Person does not have heart disease')
