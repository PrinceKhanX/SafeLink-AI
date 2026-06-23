import pandas as pd

df = pd.read_csv('data/PhiUSIIL_Phishing_URL_Dataset.csv')

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nNull counts:\n", df.isnull().sum())
print("\nLabel value counts:\n", df['label'].value_counts())
