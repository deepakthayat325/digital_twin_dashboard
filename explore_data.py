import pandas as pd
from pathlib import Path

# Load the data
base_dir = Path(__file__).resolve().parents[1]
data_path = base_dir / 'Data' / 'dataset_with_tool_life.csv'
df = pd.read_csv(data_path)

print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Data types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())
print("\nUnique sequence_ids:", df['sequence_id'].nunique())
print("Sample sequence_id counts:")
print(df['sequence_id'].value_counts().head())

# Check if RPM, Feed, DOC are constant per sequence
grouped = df.groupby('sequence_id')
print("\nChecking constancy per sequence:")
for seq in df['sequence_id'].unique()[:3]:  # first 3
    seq_data = grouped.get_group(seq)
    print(f"Sequence {seq}: RPM unique: {seq_data['RPM'].nunique()}, Feed unique: {seq_data['Feed_mm_per_sec'].nunique()}, DOC unique: {seq_data['DOC_mm'].nunique()}")

# Aggregate data
agg_df = grouped.agg({
    'RPM': 'first',
    'Feed_mm_per_sec': 'first',
    'DOC_mm': 'first',
    'Tool_Life_HSS_min': 'first',
    'Force Reaction (X) [N]': 'max',
    'Force Reaction (Y) [N]': 'max',
    'Force Reaction (Z) [N]': 'max'
}).reset_index()

agg_df.rename(columns={
    'Force Reaction (X) [N]': 'Max_Force_X',
    'Force Reaction (Y) [N]': 'Max_Force_Y',
    'Force Reaction (Z) [N]': 'Max_Force_Z'
}, inplace=True)

print("\nAggregated data shape:", agg_df.shape)
print("Aggregated data head:")
print(agg_df.head())
print("\nDescribe aggregated data:")
print(agg_df.describe())