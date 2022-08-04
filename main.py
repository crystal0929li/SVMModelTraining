import pandas as pd

import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('/Users/crystalli/Documents/Senior/AmazonPTA/DataFile.csv')
# Print the first five rows
# NaN means missing data


import matplotlib.pyplot as plt

df['target_label'].value_counts().plot.bar()
plt.show()

import numpy as np
np.set_printoptions(threshold=np.inf) # use this for datasets with more columns, to print all columns


# Grab model features/inputs and target/output
numerical_features = ["ASIN_STATIC_ITEM_PACKAGE_WEIGHT",
                      "ASIN_STATIC_LIST_PRICE"]

model_features = numerical_features
model_target = 'target_label'

# Data Cleansing: Cleaning numerical features
for i in range(0,len(numerical_features)):
    print(df[numerical_features[i]].value_counts(bins=10, sort=False))

# Remove Outliers
# print(df[df[numerical_features[1]] > 3000000])
dropIndexes = df[df[numerical_features[1]] > 3000000].index
df.drop(dropIndexes , inplace=True)
df[numerical_features[1]].value_counts(bins=10, sort=False)

# Check Missing Value
print(df[numerical_features].isna().sum())

# Train Dataset
from sklearn.model_selection import train_test_split

training_data, test_data = train_test_split(df, test_size=0.1, shuffle=True, random_state=23)
train_data, val_data = train_test_split(training_data, test_size=0.15, shuffle=True, random_state=23)

# Print the shapes of the Train - Validation - Test Datasets
print('Train - Validation - Test Datasets shapes: ', train_data.shape, val_data.shape, test_data.shape)

# Data Processing with Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


### PIPELINE ###
################

# Pipeline desired data transformers, along with an estimator at the end
# For each step specify: a name, the actual transformer/estimator with its parameters
classifier = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler()),
    ('estimator', SVC())
])

# Visualize the pipeline
# This will come in handy especially when building more complex pipelines, stringing together multiple preprocessing steps
from sklearn import set_config
set_config(display='diagram')
print(classifier)

# Train and Tune a Classifier
# Get train data to train the classifier
X_train = train_data[model_features]
y_train = train_data[model_target]

# Fit the classifier to the train data
# Train data going through the Pipeline it's imputed (with means from the train data),
#   scaled (with the min/max from the train data),
#   and finally used to fit the model
classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Use the fitted model to make predictions on the train dataset
# Train data going through the Pipeline it's imputed (with means from the train data),
# scaled (with the min/max from the train data),
# and finally used to make predictions
train_predictions = classifier.predict(X_train)

print('Model performance on the train set:')
print(confusion_matrix(y_train, train_predictions))
print(classification_report(y_train, train_predictions))
print("Train accuracy:", accuracy_score(y_train, train_predictions))

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Get validation data to validate the classifier
X_val = val_data[model_features]
y_val = val_data[model_target]

# Use the fitted model to make predictions on the validation dataset
# Validation data going through the Pipeline it's imputed (with means from the train data),
#   scaled (with the min/max from the train data),
#   and finally used to make predictions
val_predictions = classifier.predict(X_val)

print('Model performance on the validation set:')
print(confusion_matrix(y_val, val_predictions))
print(classification_report(y_val, val_predictions))
print("Validation accuracy:", accuracy_score(y_val, val_predictions))

# Model Tuning