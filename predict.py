import joblib
import pandas as pd
from pathlib import Path

def predict(model_path, rpm, feed, doc):
    model = joblib.load(model_path)
    
    input_data = pd.DataFrame([[rpm, feed, doc]], columns=['RPM', 'Feed_mm_per_sec', 'DOC_mm'])
    prediction = model.predict(input_data)
    
    return {
        'Tool_Life_HSS_min': prediction[0][0],
        'Max_Force_X': prediction[0][1],
        'Max_Force_Y': prediction[0][2],
        'Max_Force_Z': prediction[0][3]
    }

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / 'models' / 'tool_life_model.pkl'
    # Example prediction
    result = predict(model_path, 1000, 2, 1)
    print(result)