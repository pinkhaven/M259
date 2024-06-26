import pandas
import matplotlib.pyplot as plt

# Loading the Datasets
abalone_attributes = pandas.read_excel('./Data/abalone_attributes_cleaned.xlsx')
abalone_targets = pandas.read_excel('./Data/abalone_targets_cleaned.xlsx')

# Statistic Information on each attribute
statistics_attributes = abalone_attributes.describe()
statistics_targets = abalone_targets.describe()

# To display the statistics
print(statistics_attributes)
print(statistics_targets)

# Creating the "Distribution of Rings" Histogramm
ring_counts = abalone_targets['Rings'].value_counts(normalize=True) * 100
plt.figure(figsize=(10, 6))
plt.bar(ring_counts.index, ring_counts.values, color='b', alpha=0.7)
plt.xlabel('Number of Rings')
plt.ylabel('Percentage (%)')
plt.title('Distribution of Rings in Abalone Dataset')
plt.show()

