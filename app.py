import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.predict import predict

st.set_page_config(page_title="CNC Milling Digital Twin Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    processed_data = pd.read_csv(r'c:\Users\deepak thayat\Desktop\Project\Data\processed_data.csv')
    raw_data = pd.read_csv(r'c:\Users\deepak thayat\Desktop\Project\Data\dataset_with_tool_life.csv')
    return processed_data, raw_data

processed_data, raw_data = load_data()

st.title("CNC Milling Machine Digital Twin Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["Tool Life Prediction", "Data Overview", "Force Analysis", "Model Insights"])

model_path = r'c:\Users\deepak thayat\Desktop\Project\models\tool_life_model.pkl'

with tab1:
    st.header("Tool Life and Force Prediction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameters")
        rpm = st.slider("RPM", 1000, 5000, 1000, step=1000, help="Spindle speed in revolutions per minute")
        feed = st.slider("Feed (mm/sec)", 1.0, 6.0, 2.0, help="Feed rate in mm per second")
        doc = st.slider("Depth of Cut (mm)", 1.0, 5.0, 1.0, help="Depth of cut in mm")
        
        if st.button("Predict", type="primary"):
            try:
                result = predict(model_path, rpm, feed, doc)
                
                st.success("Prediction completed!")
                
                # Display results
                st.subheader("Predicted Results")
                
                # Tool Life prominently displayed
                st.markdown("### Tool Life Prediction")
                tool_life = result['Tool_Life_HSS_min']
                st.metric("Estimated Tool Life", f"{tool_life:.2f} minutes", 
                         delta=f"{'Good' if tool_life > 30 else 'Monitor' if tool_life > 15 else 'Replace Soon'}",
                         delta_color="normal")
                
                # Force metrics
                st.markdown("### Maximum Cutting Forces")
                force_values = [abs(result['Max_Force_X']), abs(result['Max_Force_Y']), abs(result['Max_Force_Z'])]
                force_labels = ["X Direction (N)", "Y Direction (N)", "Z Direction (N)"]
                force_cols = st.columns(3)
                
                for col, label, value in zip(force_cols, force_labels, force_values):
                    with col:
                        st.markdown(f"**{label}**")
                        st.markdown(f"# {value:.2f}")
                    
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
    
    with col2:
        st.subheader("Prediction Visualization")
        if 'result' in locals():
            # Tool Life Gauge
            tool_life = result['Tool_Life_HSS_min']
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = tool_life,
                title = {'text': "Tool Life (minutes)"},
                gauge = {
                    'axis': {'range': [0, 60]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 15], 'color': "red"},
                        {'range': [15, 30], 'color': "orange"},
                        {'range': [30, 60], 'color': "green"}],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': tool_life}}))
            st.plotly_chart(fig_gauge)
            
            # Forces Bar Chart with magnitude
            forces = [abs(result['Max_Force_X']), abs(result['Max_Force_Y']), abs(result['Max_Force_Z'])]
            force_colors = ['#5dade2', '#f4d03f', '#e74c3c']
            fig_bar = go.Figure(data=go.Bar(
                x=['X', 'Y', 'Z'],
                y=forces,
                marker_color=force_colors,
                text=[f"{v:.1f} N" for v in forces],
                textposition='auto'
            ))
            fig_bar.update_layout(
                title="Predicted Maximum Force Magnitudes",
                yaxis_title="Force Magnitude (N)",
                xaxis_title="Direction",
                template='plotly_white',
                yaxis=dict(showgrid=True)
            )
            st.plotly_chart(fig_bar)

with tab2:
    st.header("Data Overview")
    
    st.subheader("Processed Dataset Summary")
    st.dataframe(processed_data.describe())
    
    st.subheader("Feature Correlations")
    corr = processed_data[['RPM', 'Feed_mm_per_sec', 'DOC_mm', 'Tool_Life_HSS_min', 'Max_Force_X', 'Max_Force_Y', 'Max_Force_Z']].corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
    st.plotly_chart(fig)
    
    st.subheader("Scatter Plots")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(processed_data, x='RPM', y='Tool_Life_HSS_min', title="RPM vs Tool Life")
        st.plotly_chart(fig)
    with col2:
        fig = px.scatter(processed_data, x='DOC_mm', y='Max_Force_X', title="DOC vs Max Force X")
        st.plotly_chart(fig)

with tab3:
    st.header("Force Analysis")
    
    sequence = st.selectbox("Select Milling Sequence", processed_data['sequence_id'].unique())
    
    seq_data = raw_data[raw_data['sequence_id'] == sequence]
    
    st.subheader(f"Force Profiles for Sequence {sequence}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seq_data['Time [s]'], y=seq_data['Force Reaction (X) [N]'], mode='lines', name='Force X'))
    fig.add_trace(go.Scatter(x=seq_data['Time [s]'], y=seq_data['Force Reaction (Y) [N]'], mode='lines', name='Force Y'))
    fig.add_trace(go.Scatter(x=seq_data['Time [s]'], y=seq_data['Force Reaction (Z) [N]'], mode='lines', name='Force Z'))
    fig.update_layout(title=f"Force Reactions Over Time - Sequence {sequence}", xaxis_title="Time (s)", yaxis_title="Force (N)")
    st.plotly_chart(fig)
    
    st.subheader("Sequence Parameters")
    params = processed_data[processed_data['sequence_id'] == sequence][['RPM', 'Feed_mm_per_sec', 'DOC_mm', 'Tool_Life_HSS_min']].iloc[0]
    st.write(f"RPM: {params['RPM']}, Feed: {params['Feed_mm_per_sec']} mm/sec, DOC: {params['DOC_mm']} mm, Tool Life: {params['Tool_Life_HSS_min']:.2f} min")

with tab4:
    st.header("Model Insights")
    
    st.subheader("Model Performance Metrics")
    metrics = {
        'Target': ['Tool Life', 'Max Force X', 'Max Force Y', 'Max Force Z'],
        'R² Score': [0.997, 0.784, 0.747, 0.565]  # Updated if better
    }
    metrics_df = pd.DataFrame(metrics)
    st.dataframe(metrics_df)
    
    fig = px.bar(metrics_df, x='Target', y='R² Score', title="Model R² Scores by Target", color='R² Score', color_continuous_scale='viridis')
    st.plotly_chart(fig)
    
    st.subheader("Model Details")
    st.write("**Best Model:** Gradient Boosting Regressor (Hyperparameter Tuned)")
    st.write("**Tuned Parameters:** n_estimators=200, learning_rate=0.1, max_depth=5, min_samples_split=5, min_samples_leaf=2, subsample=0.9")
    st.write("**Features:** RPM, Feed (mm/sec), Depth of Cut (mm)")
    st.write("**Targets:** Tool Life (min), Max Forces X/Y/Z (N)")
    st.write("**Training Data:** 27 milling sequences")
    st.write("**Note:** Model performance optimized through hyperparameter tuning. Force predictions for Z-direction have lower accuracy due to limited training data.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Digital Twin Features:**")
st.sidebar.markdown("- Predictive tool life estimation")
st.sidebar.markdown("- Force monitoring and analysis") 
st.sidebar.markdown("- Data-driven insights")
st.sidebar.markdown("- Interactive visualizations")