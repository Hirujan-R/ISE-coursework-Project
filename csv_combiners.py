
# List of CSV file paths
import pandas as pd

csv_files = ['./datasets/pytorch.csv', './datasets/tensorflow.csv', './datasets/keras.csv',
             './datasets/incubator-mxnet.csv', './datasets/caffe.csv']
# Load each CSV file into a separate DataFrame and add an identifier for each software
dfs = []
for i, file in enumerate(csv_files):
    df = pd.read_csv(file)
    df['software'] = f'software{i+1}'  # Add a new column to identify the software
    dfs.append(df)

# Concatenate all DataFrames into one single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Check the combined data
print(combined_df.head())

# Optionally, save the combined DataFrame to a CSV file
combined_df.to_csv('combined_dataset.csv', index=False)