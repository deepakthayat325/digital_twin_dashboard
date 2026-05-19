import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path


def train_model(data_path, model_path):
    df = pd.read_csv(data_path)

    features = ['RPM', 'Feed_mm_per_sec', 'DOC_mm']
    targets = ['Tool_Life_HSS_min', 'Max_Force_X', 'Max_Force_Y', 'Max_Force_Z']

    X = df[features]
    y = df[targets]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultiOutputRegressor(GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.9,
        random_state=42
    ))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')

    print('Training complete.')
    print('MSE:', mse)
    print('R2:', r2)

    joblib.dump(model, model_path)
    print(f'Model saved to {model_path}')

    return model


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / 'Data' / 'processed_data.csv'
    model_path = base_dir / 'models' / 'tool_life_model.pkl'
    train_model(data_path, model_path)
