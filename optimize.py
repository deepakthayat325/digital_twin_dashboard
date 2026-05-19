import numpy as np
import pandas as pd
import joblib
from pathlib import Path


def load_model(model_path):
    return joblib.load(model_path)


def predict_with_model(model, rpm, feed, doc):
    input_data = pd.DataFrame([[rpm, feed, doc]], columns=['RPM', 'Feed_mm_per_sec', 'DOC_mm'])
    prediction = model.predict(input_data)[0]
    return {
        'Tool_Life_HSS_min': float(prediction[0]),
        'Max_Force_X': float(prediction[1]),
        'Max_Force_Y': float(prediction[2]),
        'Max_Force_Z': float(prediction[3])
    }


def optimization_score(prediction):
    tool_life = prediction['Tool_Life_HSS_min']
    forces = np.abs([prediction['Max_Force_X'], prediction['Max_Force_Y'], prediction['Max_Force_Z']])
    avg_force = np.mean(forces)
    # Focus primarily on tool life; use a very small force penalty only to break ties.
    return tool_life - 0.01 * avg_force


def optimize_machining_parameters(
    model_path,
    rpm_min=1000,
    rpm_max=5000,
    feed_min=2.0,
    feed_max=6.0,
    doc_min=1.0,
    doc_max=5.0,
    rpm_step=250,
    feed_step=0.25,
    doc_step=0.25
):
    model = load_model(model_path)

    rpm_values = np.arange(rpm_min, rpm_max + 1, rpm_step, dtype=int)
    feed_values = np.arange(feed_min, feed_max + feed_step / 2, feed_step)
    doc_values = np.arange(doc_min, doc_max + doc_step / 2, doc_step)

    best_score = -np.inf
    best_result = None

    for rpm in rpm_values:
        for feed in feed_values:
            for doc in doc_values:
                prediction = predict_with_model(model, rpm, round(float(feed), 3), round(float(doc), 3))
                score = optimization_score(prediction)
                if score > best_score:
                    best_score = score
                    best_result = {
                        'RPM': int(rpm),
                        'Feed_mm_per_sec': float(round(feed, 3)),
                        'DOC_mm': float(round(doc, 3)),
                        **prediction,
                        'Objective_Score': float(round(score, 4))
                    }

    if best_result is not None:
        best_result['Predicted_Resultant_Force'] = float(np.sqrt(
            best_result['Max_Force_X'] ** 2 +
            best_result['Max_Force_Y'] ** 2 +
            best_result['Max_Force_Z'] ** 2
        ))

    return best_result


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / 'models' / 'tool_life_model.pkl'
    print(optimize_machining_parameters(str(model_path)))
