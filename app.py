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

# ================= IMPORTS =================

from predict import predict
from optimize import optimize_machining_parameters


# ================= LOAD DATA =================

@st.cache_data
def load_data():

    processed_data = pd.read_csv(PROCESSED_PATH)
    raw_data = pd.read_csv(DATASET_PATH)

    if "Resultant_Force" not in processed_data.columns:

        processed_data["Resultant_Force"]=(

            processed_data["Max_Force_X"]**2
            + processed_data["Max_Force_Y"]**2
            + processed_data["Max_Force_Z"]**2

        )**0.5

    return processed_data,raw_data


def format_force(v):
    return f"{abs(v):.2f} N"


# ================= MAIN APP =================

def main():

    st.set_page_config(
        page_title="CNC Milling Optimization Dashboard",
        layout="wide"
    )

    processed_data,raw_data=load_data()

    st.title(
        "CNC Milling Optimization Dashboard"
    )

    tab1,tab2,tab3,tab4=st.tabs([

        "Tool Life Prediction",
        "Data Overview",
        "Force Analysis",
        "Model Insights"

    ])

# ===================================================
# TAB1
# ===================================================

    with tab1:

        st.header(
            "Tool Life and Optimization"
        )

        col1,col2=st.columns([1,2])

        if "prediction" not in st.session_state:
            st.session_state.prediction=None

        if "optimization" not in st.session_state:
            st.session_state.optimization=None


# ================= INPUT =================

        with col1:

            st.subheader(
                "Input Parameters"
            )

            rpm=st.slider(
                "RPM",
                1000,
                5000,
                3500,
                step=250
            )

            feed=st.slider(
                "Feed (mm/sec)",
                2.0,
                6.0,
                5.0,
                step=0.25
            )

            doc=st.slider(
                "Depth of Cut (mm)",
                1.0,
                5.0,
                4.0,
                step=0.25
            )


            if st.button(
                "Predict",
                type="primary"
            ):

                st.session_state.prediction=(

                    predict(
                        MODEL_PATH,
                        rpm,
                        feed,
                        doc
                    )

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

                st.session_state.optimization=(

                    optimize_machining_parameters(

                        str(MODEL_PATH),

                        rpm_min=1000,
                        rpm_max=5000,

                        feed_min=2,
                        feed_max=6,

                        doc_min=1,
                        doc_max=5,

                        rpm_step=250,
                        feed_step=0.25,
                        doc_step=0.25

                    )

                )

                st.success(
                    "Optimization completed"
                )


# ================= RESULTS =================

        with col2:

            st.subheader(
                "Prediction Results"
            )

            if st.session_state.prediction:

                result=st.session_state.prediction

                tool_life=result[
                    "Tool_Life_HSS_min"
                ]

# ================= TOOL LIFE GAUGE =================

                gauge=go.Figure(

                    go.Indicator(

                        mode="gauge+number",

                        value=tool_life,

                        number={

                            "suffix":" min",
                            "font":{"size":50}

                        },

                        gauge={

                            "axis":{
                                "range":[0,60]
                            },

                            "bar":{
                                "color":"#001f5b"
                            },

                            "steps":[

                                {
                                    "range":[0,10],
                                    "color":"#ff3131"
                                },

                                {
                                    "range":[10,20],
                                    "color":"#ff6600"
                                },

                                {
                                    "range":[20,30],
                                    "color":"#ffcc00"
                                },

                                {
                                    "range":[30,60],
                                    "color":"#1db954"
                                }

                            ]

                        }

                    )

                )

                gauge.update_layout(
                    height=300
                )

                st.plotly_chart(
                    gauge,
                    use_container_width=True
                )

# ================= FORCE BAR =================

                forces=[

                    result["Max_Force_X"],
                    result["Max_Force_Y"],
                    result["Max_Force_Z"]

                ]

                fig=go.Figure(

                    data=[

                        go.Bar(

                            x=["X","Y","Z"],

                            y=forces,

                            text=[
                                f"{x:.2f} N"
                                for x in forces
                            ],

                            textposition="outside",

                            marker_color=[

                                "#3399ff",
                                "#ff7b00",
                                "#1ea83c"

                            ]

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

                resultant=(

                    result["Max_Force_X"]**2
                    + result["Max_Force_Y"]**2
                    + result["Max_Force_Z"]**2

                )**0.5

                st.metric(

                    "Predicted Resultant Force",
                    f"{resultant:.2f} N"

                )


# ================= OPTIMIZATION =================

            if st.session_state.optimization:

                st.markdown("---")

                st.subheader(
                    "Optimized Settings"
                )

                opt=st.session_state.optimization

                c1,c2,c3=st.columns(3)

                c1.metric(
                    "RPM",
                    opt["RPM"]
                )

                c2.metric(
                    "Feed",
                    f"{opt['Feed_mm_per_sec']:.2f} mm/sec"
                )

                c3.metric(
                    "DOC",
                    f"{opt['DOC_mm']:.2f} mm"
                )

                st.metric(

                    "Optimized Tool Life",
                    f"{opt['Tool_Life_HSS_min']:.2f} min"

                )


                optimized_force=(

                    opt["Max_Force_X"]**2
                    + opt["Max_Force_Y"]**2
                    + opt["Max_Force_Z"]**2

                )**0.5


                st.metric(

                    "Optimized Resultant Force",
                    f"{optimized_force:.2f} N"

                )

                fig2=go.Figure(

                    data=[

                        go.Bar(

                            x=["X","Y","Z"],

                            y=[

                                opt["Max_Force_X"],
                                opt["Max_Force_Y"],
                                opt["Max_Force_Z"]

                            ],

                            textposition="outside",

                            marker_color=[

                                "#2d5bff",
                                "#ff2222",
                                "#22bb55"

                            ]

                        )

                    ]

                )

                fig2.update_layout(

                    title="Optimized Force Prediction",
                    yaxis_title="Force (N)"

                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

# ===================================================
# TAB2
# ===================================================

    with tab2:

        st.dataframe(
            processed_data
        )


# ===================================================
# TAB3
# ===================================================

    with tab3:

        sequence=st.selectbox(

            "Select Sequence",
            processed_data[
                "sequence_id"
            ].unique()

        )

        seq=raw_data[
            raw_data["sequence_id"]==sequence
        ]

        fig=go.Figure()

        fig.add_trace(
            go.Scatter(
                x=seq["Time [s]"],
                y=seq["Force Reaction (X) [N]"],
                name="Force X"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=seq["Time [s]"],
                y=seq["Force Reaction (Y) [N]"],
                name="Force Y"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=seq["Time [s]"],
                y=seq["Force Reaction (Z) [N]"],
                name="Force Z"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ===================================================
# TAB4
# ===================================================

    with tab4:

        st.write(
            "Model Insights"
        )


if __name__=="__main__":
    main()
