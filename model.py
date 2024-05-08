import pandas
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, confusion_matrix


# loading the Dataset
abalone_attributes = pandas.read_excel('./Data/abalone_attributes_cleaned.xlsx')
abalone_targets = pandas.read_excel('./Data/abalone_targets_cleaned.xlsx')

# Select the data for the test and train sets
attributes = abalone_attributes.drop(columns='Unnamed: 0', axis=1)
targets = abalone_targets['Rings']

# Split the data into train and test sets
attributes_train, attributes_test, targets_train, targets_test = train_test_split(
    attributes, targets, test_size=200, random_state=33
)


## Model training ##

# Chose an appropriate Model
model = RandomForestRegressor(random_state=33)

# Train the Model
model.fit(attributes_train, targets_train)


## Predictions ##

# Generate Predictions
predictions = model.predict(attributes_test)

# Check the predictions
results = pandas.DataFrame({'Prediction': predictions, 'Actual Value': targets_test})
print(results)

# Save the Results
# results.to_excel('./Data/abalone_results.xlsx', index=False)


## Evaluate Deviation ##

# Calculate average deviation in rings and percent
deviation = mean_absolute_error(targets_test, predictions)
deviation_p = deviation / 29 * 100

# Check deviation
print("Deviation in rings: ", deviation)
print("Deviation in percentage: ", deviation_p)


## Important Attributes ##

# Extract the important attributes
important_attributes = model.feature_importances_

# Convert to percent
important_attributes = important_attributes * 100

# Create a dataframe with the important attributes
important_attributes_dataframe = pandas.DataFrame({'Feature': attributes.columns, 'Importance': important_attributes})

# Sort the dataframe
important_attributes_dataframe = important_attributes_dataframe.sort_values(by='Importance', ascending=False)

# Visualize the sorted attributes
plt.figure(figsize=(13, 7))
plt.barh(important_attributes_dataframe['Feature'], important_attributes_dataframe['Importance'])
plt.xlabel('Importance in %')
plt.ylabel('Attribute')
plt.title('Attribute Importance')
plt.show()

# Check the top 5 attributes
top_k_attributes = important_attributes_dataframe.head(5)
print(top_k_attributes)


## Confusion Matrix ##

# prepare the Values
predictions = predictions.round().astype(int)
targets_test = targets_test.round().astype(int)

# Calculating the Matrix
confusion_matrix = confusion_matrix(targets_test, predictions)
print('Confusion Matrix:')
print(confusion_matrix)

# Calculating Sensitivity and Specificity
true_positive = confusion_matrix[1, 1]
false_positive = confusion_matrix[0, 1]
true_negative = confusion_matrix[0, 0]
false_negative = confusion_matrix[1, 0]

sensitivity = true_positive / (true_positive + false_negative)
specificity = true_negative / (true_negative + false_positive)

print('Sensitivity:', sensitivity)
print('Specificity:', specificity)

