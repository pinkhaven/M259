import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


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

# Chose an appropriate Model
model = RandomForestRegressor(random_state=33)

# Train the Model
model.fit(attributes_train, targets_train)

# Generate Predictions
predictions = model.predict(attributes_test)

# Check the predictions
results = pandas.DataFrame({'Prediction': predictions, 'Actual Value': targets_test})
print(results)

# Calculate average deviation in rings and percent
deviation = mean_absolute_error(targets_test, predictions)
deviation_p = deviation / 29 * 100

# Check deviation
print("Deviation in rings: ", deviation)
print("Deviation in percentage: ", deviation_p)

# Save the Results
results.to_excel('./Data/abalone_results.xlsx', index=False)
