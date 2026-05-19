import contextlib
import io
import sys
from pathlib import Path


def import_streamlit():
    try:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            import streamlit as st
            from streamlit.runtime.scriptrunner import get_script_run_ctx
        return st, get_script_run_ctx
    except Exception:
        return None, None


st, get_script_run_ctx = import_streamlit()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).resolve().parent))

from src.predict import predict
from src.optimize import optimize_machining_parameters

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'Data'
MODEL_PATH = BASE_DIR / 'models' / 'tool_life_model.pkl'


def load_data():
    processed_data = pd.read_csv(DATA_DIR / 'processed_data.csv')
    raw_data = pd.read_csv(DATA_DIR / 'dataset_with_tool_life.csv')
    return processed_data, raw_data


def format_force(value):
    return f"{abs(value):.2f} N"


def main():
    st.set_page_config(page_title='CNC Milling Optimization Dashboard', layout='wide')
    processed_data, raw_data = load_data()

    st.title('CNC Milling Optimization Dashboard')
    tab1, tab2, tab3, tab4 = st.tabs([
        'Tool Life Prediction',
        'Data Overview',
        'Force Analysis',
        'Model Insights'
    ])

    with tab1:
        st.header('Tool Life and Optimization')
        col1, col2 = st.columns([1, 2])

        if 'prediction' not in st.session_state:
            st.session_state.prediction = None
        if 'optimization' not in st.session_state:
            st.session_state.optimization = None

        with col1:
            st.subheader('Input Parameters')
            rpm = st.slider('RPM', 1000, 5000, 1000, step=500)
            feed = st.slider('Feed (mm/sec)', 2.0, 6.0, 2.0, step=0.25)
            doc = st.slider('Depth of Cut (mm)', 1.0, 5.0, 1.0, step=0.25)

            if st.button('Predict', type='primary'):
                st.session_state.prediction = predict(str(MODEL_PATH), rpm, feed, doc)
                st.success('Prediction completed!')

            st.markdown('---')
            st.subheader('Optimize Machining Parameters')
            st.write('Search for the best RPM, Feed, and DOC combination to maximize tool life.')

            if st.button('Optimize Parameters', key='optimize'):
                st.session_state.optimization = optimize_machining_parameters(
                    str(MODEL_PATH),
                    rpm_min=1000,
                    rpm_max=5000,
                    feed_min=2.0,
                    feed_max=6.0,
                    doc_min=1.0,
                    doc_max=5.0,
                    rpm_step=250,
                    feed_step=0.25,
                    doc_step=0.25
                )
                st.success('Optimized parameters found!')

        with col2:
            st.subheader('Prediction & Optimization Results')
            if st.session_state.prediction is not None:
                result = st.session_state.prediction
                tool_life = result['Tool_Life_HSS_min']
                st.markdown('### Predicted Outputs')

                gauge_fig = go.Figure(go.Indicator(
                    mode='gauge+number',
                    value=tool_life,
                    title={'text': 'Tool Life (minutes)'},
                    number={'suffix': ' min', 'font': {'size': 42}},
                    gauge={
                        'axis': {'range': [0, 60], 'tickwidth': 1, 'tickcolor': 'darkgray'},
                        'bar': {'color': '#192a56'},
                        'steps': [
                            {'range': [0, 15], 'color': '#e74c3c'},
                            {'range': [15, 30], 'color': '#f39c12'},
                            {'range': [30, 60], 'color': '#27ae60'}
                        ],
                        'threshold': {
                            'line': {'color': 'black', 'width': 4},
                            'thickness': 0.75,
                            'value': tool_life
                        }
                    }
                ))
                gauge_fig.update_layout(margin={'t': 30, 'b': 0, 'l': 0, 'r': 0}, height=360)
                st.plotly_chart(gauge_fig)

                forces = [result['Max_Force_X'], result['Max_Force_Y'], result['Max_Force_Z']]
                fig_bar = go.Figure(data=go.Bar(
                    x=['X', 'Y', 'Z'],
                    y=[abs(v) for v in forces],
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                    text=[format_force(v) for v in forces],
                    textposition='auto'
                ))
                fig_bar.update_layout(title='Predicted Maximum Forces', yaxis_title='Force (N)')
                st.plotly_chart(fig_bar)

                resultant = (result['Max_Force_X'] ** 2 + result['Max_Force_Y'] ** 2 + result['Max_Force_Z'] ** 2) ** 0.5
                st.metric('Predicted Resultant Force', f'{resultant:.2f} N')

            if st.session_state.optimization is not None:
                opt_result = st.session_state.optimization
                st.markdown('---')
                st.subheader('Optimized Settings')
                cols = st.columns(3)
                cols[0].metric('RPM', opt_result['RPM'])
                cols[1].metric('Feed', f"{opt_result['Feed_mm_per_sec']:.2f} mm/sec")
                cols[2].metric('DOC', f"{opt_result['DOC_mm']:.2f} mm")

                st.metric('Optimized Tool Life', f"{opt_result['Tool_Life_HSS_min']:.2f} minutes")
                st.metric('Optimized Resultant Force', f"{opt_result['Predicted_Resultant_Force']:.2f} N")

                fig_opt = go.Figure(data=go.Bar(
                    x=['X', 'Y', 'Z'],
                    y=[abs(opt_result['Max_Force_X']), abs(opt_result['Max_Force_Y']), abs(opt_result['Max_Force_Z'])],
                    marker_color=['#636efa', '#ef553b', '#00cc96'],
                    text=[format_force(opt_result['Max_Force_X']), format_force(opt_result['Max_Force_Y']), format_force(opt_result['Max_Force_Z'])],
                    textposition='auto'
                ))
                fig_opt.update_layout(title='Optimized Force Prediction', yaxis_title='Force (N)')
                st.plotly_chart(fig_opt)

    with tab2:
        st.header('Data Overview')
        st.subheader('Processed Dataset')
        st.dataframe(processed_data)

        st.subheader('Feature Correlations')
        corr_columns = ['RPM', 'Feed_mm_per_sec', 'DOC_mm', 'Tool_Life_HSS_min', 'Max_Force_X', 'Max_Force_Y', 'Max_Force_Z', 'Resultant_Force']
        corr = processed_data[corr_columns].corr()
        fig_corr = px.imshow(corr, text_auto=True, title='Correlation Matrix')
        st.plotly_chart(fig_corr)

        st.subheader('Resultant Force Comparison')
        fig_resultant = px.scatter(
            processed_data,
            x='sequence_id',
            y=['Max_Force_X', 'Max_Force_Y', 'Max_Force_Z', 'Resultant_Force'],
            title='Sequence Forces vs Resultant Force',
            labels={'value': 'Force (N)', 'sequence_id': 'Sequence ID'},
            template='plotly_white'
        )
        fig_resultant.update_layout(legend_title='Force Type')
        st.plotly_chart(fig_resultant)

    with tab3:
        st.header('Force Analysis')
        sequence = st.selectbox('Select Milling Sequence', processed_data['sequence_id'].unique())
        seq_data = raw_data[raw_data['sequence_id'] == sequence]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=seq_data['Time [s]'], y=seq_data['Force Reaction (X) [N]'], mode='lines', name='Force X'))
        fig.add_trace(go.Scatter(x=seq_data['Time [s]'], y=seq_data['Force Reaction (Y) [N]'], mode='lines', name='Force Y'))
        fig.add_trace(go.Scatter(x=seq_data['Time [s]'], y=seq_data['Force Reaction (Z) [N]'], mode='lines', name='Force Z'))
        fig.update_layout(title=f'Force Reactions Over Time - Sequence {sequence}', xaxis_title='Time (s)', yaxis_title='Force (N)')
        st.plotly_chart(fig)

        params = processed_data[processed_data['sequence_id'] == sequence][['RPM', 'Feed_mm_per_sec', 'DOC_mm', 'Tool_Life_HSS_min', 'Resultant_Force']].iloc[0]
        st.write(f"RPM: {params['RPM']}, Feed: {params['Feed_mm_per_sec']} mm/sec, DOC: {params['DOC_mm']} mm, Tool Life: {params['Tool_Life_HSS_min']:.2f} min")
        st.write(f"Resultant Force: {params['Resultant_Force']:.2f} N")

    with tab4:
        st.header('Model Insights')
        metrics = {
            'Target': ['Tool Life', 'Max Force X', 'Max Force Y', 'Max Force Z'],
            'R² Score': [0.997, 0.784, 0.747, 0.565]
        }
        metrics_df = pd.DataFrame(metrics)
        st.dataframe(metrics_df)
        fig_metrics = px.bar(metrics_df, x='Target', y='R² Score', title='Model R² Scores by Target', color='R² Score')
        st.plotly_chart(fig_metrics)
        st.write('**Note:** Use `streamlit run app.py` to start this dashboard. Running with `python app.py` may show bare-mode warnings.')


if __name__ == '__main__':
    if get_script_run_ctx() is None:
        print('Please launch this app with Streamlit:')
        print('    streamlit run app.py')
    else:
        main()
