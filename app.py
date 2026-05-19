import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ================= PATH SETUP =================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset_with_tool_life.csv"
PROCESSED_PATH = BASE_DIR / "processed_data.csv"
MODEL_PATH = BASE_DIR / "tool_life_model.pkl"

# ================= CUSTOM IMPORTS =================

from predict import predict
from optimize import optimize_machining_parameters

# ================= LOAD DATA =================

@st.cache_data
def load_data():

    processed_data = pd.read_csv(PROCESSED_PATH)
    raw_data = pd.read_csv(DATASET_PATH)

    # Create resultant force if not present
    if "Resultant_Force" not in processed_data.columns:

        processed_data["Resultant_Force"] = (
            processed_data["Max_Force_X"]**2
            + processed_data["Max_Force_Y"]**2
            + processed_data["Max_Force_Z"]**2
        )**0.5

    return processed_data, raw_data


def format_force(value):
    return f"{abs(value):.2f} N"


# ================= MAIN APP =================

def main():

    st.set_page_config(
        page_title="CNC Milling Optimization Dashboard",
        layout="wide"
    )

    processed_data, raw_data = load_data()

    st.title("CNC Milling Optimization Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Tool Life Prediction",
        "Data Overview",
        "Force Analysis",
        "Model Insights"
    ])

    # ==========================================
    # TAB 1
    # ==========================================

    with tab1:

        st.header("Tool Life Prediction & Optimization")

        col1, col2 = st.columns([1,2])

        if "prediction" not in st.session_state:
            st.session_state.prediction = None

        if "optimization" not in st.session_state:
            st.session_state.optimization = None

        with col1:

            st.subheader("Input Parameters")

            rpm = st.slider(
                "RPM",
                min_value=1000,
                max_value=5000,
                value=1000,
                step=500
            )

            feed = st.slider(
                "Feed (mm/sec)",
                min_value=2.0,
                max_value=6.0,
                value=2.0,
                step=0.25
            )

            doc = st.slider(
                "Depth of Cut (mm)",
                min_value=1.0,
                max_value=5.0,
                value=1.0,
                step=0.25
            )

            if st.button(
                "Predict",
                type="primary"
            ):

                st.session_state.prediction = predict(
                    MODEL_PATH,
                    rpm,
                    feed,
                    doc
                )

                st.success(
                    "Prediction completed"
                )

            st.markdown("---")

            st.subheader(
                "Optimization"
            )

            if st.button(
                "Optimize Parameters"
            ):

                st.session_state.optimization = (
                    optimize_machining_parameters(
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
                )

                st.success(
                    "Optimization completed"
                )

        with col2:

            st.subheader(
                "Prediction Results"
            )

            if st.session_state.prediction:

                result = st.session_state.prediction

                tool_life = result[
                    "Tool_Life_HSS_min"
                ]

                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=tool_life,
                        title={
                            "text":"Tool Life (min)"
                        }
                    )
                )

                st.plotly_chart(
                    gauge,
                    use_container_width=True
                )

                forces = [
                    result["Max_Force_X"],
                    result["Max_Force_Y"],
                    result["Max_Force_Z"]
                ]

                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=["X","Y","Z"],
                            y=[abs(i) for i in forces],
                            text=[
                                format_force(i)
                                for i in forces
                            ],
                            textposition="auto"
                        )
                    ]
                )

                fig.update_layout(
                    title="Predicted Maximum Forces",
                    yaxis_title="Force (N)"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                resultant = (
                    result["Max_Force_X"]**2
                    + result["Max_Force_Y"]**2
                    + result["Max_Force_Z"]**2
                )**0.5

                st.metric(
                    "Resultant Force",
                    f"{resultant:.2f} N"
                )

            if st.session_state.optimization:

                st.markdown("---")

                opt = st.session_state.optimization

                st.subheader(
                    "Optimized Parameters"
                )

                c1,c2,c3 = st.columns(3)

                c1.metric(
                    "RPM",
                    opt["RPM"]
                )

                c2.metric(
                    "Feed",
                    f"{opt['Feed_mm_per_sec']:.2f}"
                )

                c3.metric(
                    "DOC",
                    f"{opt['DOC_mm']:.2f}"
                )

    # ==========================================
    # TAB 2
    # ==========================================

    with tab2:

        st.header("Processed Dataset")

        st.dataframe(
            processed_data
        )

        corr_columns = [
            'RPM',
            'Feed_mm_per_sec',
            'DOC_mm',
            'Tool_Life_HSS_min',
            'Max_Force_X',
            'Max_Force_Y',
            'Max_Force_Z',
            'Resultant_Force'
        ]

        corr = processed_data[
            corr_columns
        ].corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            title="Correlation Matrix"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ==========================================
    # TAB 3
    # ==========================================

    with tab3:

        sequence = st.selectbox(
            "Select Sequence",
            processed_data[
                "sequence_id"
            ].unique()
        )

        seq = raw_data[
            raw_data["sequence_id"]
            == sequence
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=seq["Time [s]"],
                y=seq["Force Reaction (X) [N]"],
                mode="lines",
                name="Force X"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=seq["Time [s]"],
                y=seq["Force Reaction (Y) [N]"],
                mode="lines",
                name="Force Y"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=seq["Time [s]"],
                y=seq["Force Reaction (Z) [N]"],
                mode="lines",
                name="Force Z"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ==========================================
    # TAB 4
    # ==========================================

    with tab4:

        metrics = pd.DataFrame({

            "Target":[
                "Tool Life",
                "Max Force X",
                "Max Force Y",
                "Max Force Z"
            ],

            "R² Score":[
                0.997,
                0.784,
                0.747,
                0.565
            ]

        })

        st.dataframe(metrics)

        fig = px.bar(
            metrics,
            x="Target",
            y="R² Score",
            title="Model Performance"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


if __name__ == "__main__":
    main()
