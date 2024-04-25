from ucimlrepo import fetch_ucirepo

# Loading the Dataset
abalone_dataset = fetch_ucirepo(name="Abalone")

# Exporting the Data as Excel Files
abalone_dataset.data.features.to_excel('./Data/abalone_attributes.xlsx')
abalone_dataset.data.targets.to_excel('./Data/abalone_targets.xlsx')
