import numpy as np
import pandas as pd
from pathlib import Path


def preprocess_data(data_path):
    df = pd.read_csv(data_path)

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

    agg_df['Resultant_Force'] = np.sqrt(
        agg_df['Max_Force_X'] ** 2 +
        agg_df['Max_Force_Y'] ** 2 +
        agg_df['Max_Force_Z'] ** 2
    )

    return agg_df


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parents[1]
    input_path = base_dir / 'Data' / 'dataset_with_tool_life.csv'
    output_path = base_dir / 'Data' / 'processed_data.csv'

    processed = preprocess_data(input_path)
    processed.to_csv(output_path, index=False)
    print(f'Processed data saved to {output_path}')
    print(processed.head())
