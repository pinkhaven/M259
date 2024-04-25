import pandas
from sklearn import preprocessing

# Loading the Dataset
abalone_dataset = pandas.read_excel('abalone_features.xlsx')

# Check for empty values
print(abalone_dataset.isna().sum())

# Drop Instances with empty values
abalone_dataset = abalone_dataset.dropna()

# Check for duplicates
print(abalone_dataset.duplicated().sum())

# Drop Duplicates
abalone_dataset = abalone_dataset.drop_duplicates()

# In the following part, we change M,F and I to 1,2 and 3, and also Normalize the Values, so they are between 0 and 1.

# Create a lable encoder object
le = preprocessing.LabelEncoder()

# Fit the encoder to the pandas column
le.fit(abalone_dataset['Sex'])

# View the labels
print(list(le.classes_))

# Apply the fitted encoder to the pandas column
abalone_dataset['Sex'] = le.transform(abalone_dataset['Sex'])

# Normalize the encoded values to be between 0 and 1
abalone_dataset['Sex'] = abalone_dataset['Sex'] / max(abalone_dataset['Sex'])


# Select the first 1000 rows
abalone_dataset = abalone_dataset.head(1000)

# Export the cleaned Dataset
abalone_dataset.to_excel('abalone_attributes_cleaned.xlsx', index=False)
