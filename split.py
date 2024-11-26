import pandas as pd
#THIS CODE FOR BIG FILES, WE CAN TAKE RANDOM ROWS AND STORE THEM IN A NEW FILE. 
# Load the dataset (assuming the CSV file is named 'bbc_test.csv')
df = pd.read_csv('Data/bbc_test.csv')

# Take a random sample of 100 rows
df_sample = df.sample(n=100, random_state=42)

# Save the sampled data to a new CSV file
df_sample.to_csv('bbc_test_sample.csv', index=False)

print("Sample data saved to 'bbc_test_sample.csv'")