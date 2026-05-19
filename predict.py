import joblib
import pandas as pd
from pathlib import Path


def predict(model_path, rpm, feed, doc):
    model = joblib.load(model_path)

    input_data = pd.DataFrame([[rpm, feed, doc]], columns=['RPM', 'Feed_mm_per_sec', 'DOC_mm'])
    prediction = model.predict(input_data)[0]

    return {
        'Tool_Life_HSS_min': float(prediction[0]),
        'Max_Force_X': float(prediction[1]),
        'Max_Force_Y': float(prediction[2]),
        'Max_Force_Z': float(prediction[3])
    }


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / 'models' / 'tool_life_model.pkl'
    result = predict(model_path, 1000, 2.0, 1.0)
    print(result)
