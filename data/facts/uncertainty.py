#%%

import pandas as pd

# Step 2: Load the CSV file
df = pd.read_csv('facts_true_false.csv')

# Step 3: Drop the 'label' column
# axis=1 denotes that we are referring to a column, not a row
df = df[df['label']==1]
df['label'] = 0

# Step 4: Save the modified DataFrame back to a CSV file
df.to_csv('uncertainty.csv', index=False)

# %%

import pandas as pd
from datasets import load_dataset

dataset = load_dataset("notrichardren/truthfulness_all")

dataset = dataset['combined']

# Filter by qa_type = 0
filtered_dataset = dataset.filter(lambda example: example['qa_type'] == 0)
filtered_dataset = filtered_dataset.filter(lambda example: example['label'] == 1)

# Select the first 600 rows
selected_rows = filtered_dataset.select(range(600))

# Extract the 'claim' column
claims = selected_rows['claim']

claims_df = pd.DataFrame(claims, columns=['claim'])

claims_df['label'] = 1

csv_file_path = 'claims.csv'

claims_df.to_csv(csv_file_path, index=False)
# %%

import pandas as pd

def add_period_to_claims(csv_file_path, output_csv_file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Check if the 'claims' column exists to avoid KeyError
    if 'claims' in df.columns:
        # Process each claim in the 'claims' column
        df['claims'] = df['claims'].apply(lambda claim: claim if claim.endswith('.') else claim + '.')

        # Write the modified DataFrame back to a new CSV file
        df.to_csv(output_csv_file_path, index=False)
    else:
        print("The 'claims' column does not exist in the provided CSV file.")

# Example usage
csv_file_path = 'path/to/your/input.csv'
output_csv_file_path = 'path/to/your/output.csv'
add_period_to_claims(csv_file_path, output_csv_file_path)

# %%
