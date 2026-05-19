import pandas as pd
from pathlib import Path

def preprocess_data(data_path):
    df = pd.read_csv(data_path)
    
    # Group by sequence_id and aggregate
    grouped = df.groupby('sequence_id')
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
    
    return agg_df

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / 'Data' / 'dataset_with_tool_life.csv'
    processed_data = preprocess_data(data_path)
    output_path = base_dir / 'Data' / 'processed_data.csv'
    processed_data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    print(processed_data.head())