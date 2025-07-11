import pandas as pd
import numpy as np
import csv

# Load the data from the CSV file
df = pd.read_csv('minimal_data_schema.csv')

# Get the column headers
headers = df.columns

# Create an identity matrix with the same headers for columns and index
identity_matrix = pd.DataFrame(np.identity(len(headers)), columns=headers, index=headers)

# Save the identity matrix to a new CSV file
identity_matrix.to_csv('identity_matrix.csv')

print("Identity matrix created and saved to 'identity_matrix.csv'")
print(identity_matrix)