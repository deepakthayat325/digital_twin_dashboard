import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_model(data_path, model_path):
    print("Starting model training and tuning...")
    df = pd.read_csv(data_path)
    
    # Features and targets
    features = ['RPM', 'Feed_mm_per_sec', 'DOC_mm']
    targets = ['Tool_Life_HSS_min', 'Max_Force_X', 'Max_Force_Y', 'Max_Force_Z']
    
    X = df[features]
    y = df[targets]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Models to tune
    models = {
        'GradientBoosting': MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
    }
    
    # Hyperparameters for tuning
    param_grid = {
        'GradientBoosting': {
            'estimator__n_estimators': [100, 200],
            'estimator__learning_rate': [0.05, 0.1, 0.2],
            'estimator__max_depth': [3, 5, 7]
        }
    }
    
    # Use tuned GradientBoosting model
    print("Using hyperparameter-tuned GradientBoosting model...")
    try:
        best_model = MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.9,
            random_state=42
        ))
        best_model.fit(X_train, y_train)
        print("Model fitted successfully")
        best_name = 'GradientBoosting (Tuned)'
    except Exception as e:
        print(f"Error in fitting: {e}")
        return None
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
    best_score = r2
    
    print(f"\nBest model: {best_name} with average R²: {best_score}")
    
    # Final evaluation
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    
    print("Final MSE for each target:", mse)
    print("Final R² for each target:", r2)
    
    # Save model
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")
    
    return best_model