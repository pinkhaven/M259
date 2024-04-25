import pandas

# Loading the Dataset
abalone_dataset = pandas.read_excel('./Data/abalone_targets.xlsx')

# Check for empty values
print(abalone_dataset.isna().sum())

# Drop Instances with empty values
abalone_dataset = abalone_dataset.dropna()

# Check for duplicates
print(abalone_dataset.duplicated().sum())

# Drop Duplicates
abalone_dataset = abalone_dataset.drop_duplicates()

# Select the first 1000 rows
abalone_dataset = abalone_dataset.head(1000)

# Export the cleaned Dataset
abalone_dataset.to_excel('./Data/abalone_targets_cleaned.xlsx', index=False)

