# Milling Tool Life Prediction Project

This project trains a machine learning model to predict tool life and maximum forces in X, Y, Z directions for milling operations based on parameters like RPM, Feed, and Depth of Cut (DOC). It also includes a Streamlit dashboard for interactive predictions.

## Architecture

- **Data Preprocessing**: Aggregates time-series data per milling sequence to compute maximum forces and extract features.
- **Model Training**: Uses Multi-Output Random Forest Regressor to predict multiple targets simultaneously.
- **Dashboard**: Streamlit web app for user input and displaying predictions.

## Project Structure

```
Project/
├── Data/
│   └── dataset_with_tool_life.csv  # Raw data
│   └── processed_data.csv          # Processed aggregated data
├── src/
│   ├── explore_data.py             # Data exploration script
│   ├── preprocess.py               # Data preprocessing
│   ├── train.py                    # Model training
│   └── predict.py                  # Prediction function
├── models/
│   └── tool_life_model.pkl         # Trained model
├── notebooks/                      # For any Jupyter notebooks
├── app.py                          # Streamlit dashboard
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Step-by-Step Guide

### 1. Setup Environment
- Ensure Python 3.8+ is installed.
- Create virtual environment: `python -m venv venv`
- Activate: `venv\Scripts\activate` (Windows)
- Install dependencies: `pip install -r requirements.txt`

### 2. Data Preprocessing
Run the preprocessing script to aggregate the data:
```
python src/preprocess.py
```
This creates `Data/processed_data.csv` with one row per milling sequence.

### 3. Train the Model
Train the machine learning model:
```
python src/train.py
```
This saves the trained model to `models/tool_life_model.pkl` and prints evaluation metrics.

### 4. Run the Dashboard
Launch the Streamlit dashboard:
```
streamlit run app.py
```
Open the provided URL in your browser. Input RPM, Feed, and DOC values, then click "Predict" to see the results.

### 5. Make Predictions Programmatically
Use the predict script for command-line predictions:
```
python src/predict.py
```
Modify the script for custom inputs.

## Data Description
- **Raw Data**: Time-series force measurements and parameters for 27 milling sequences.
- **Features**: RPM, Feed (mm/sec), Depth of Cut (mm)
- **Targets**: Tool Life (min), Max Force X, Y, Z (N)

## Model Details
- Algorithm: Multi-Output Random Forest Regressor
- Evaluation: Mean Squared Error and R² score for each target
- Note: With only 27 samples, the model may not generalize well. Consider collecting more data for better performance.

## Dashboard Features

The Streamlit dashboard provides a comprehensive digital twin interface for the CNC milling machine:

### Tool Life Prediction Tab
- Input sliders for RPM, Feed, and Depth of Cut
- Real-time predictions for tool life and maximum forces
- Interactive bar chart visualization of predicted forces

### Data Overview Tab  
- Summary statistics of the processed dataset
- Correlation matrix heatmap
- Scatter plots showing relationships between parameters

### Force Analysis Tab
- Select any milling sequence to view force profiles over time
- Interactive time-series plots of X, Y, Z forces
- Display of sequence parameters (RPM, Feed, DOC, Tool Life)

### Model Insights Tab
- Performance metrics (R² scores) for each prediction target
- Bar chart visualization of model accuracy
- Model details and limitations

## Future Improvements
- Integrate real-time sensor data for live monitoring
- Add predictive maintenance alerts
- Implement 3D visualization of milling simulation
- Expand dataset for better model performance
- Add user authentication and data logging