#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/13 15:21
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   preparation.py
# @Desc     :

from pandas import DataFrame, concat
from sklearn.datasets import make_moons
from streamlit import (empty, sidebar, subheader, session_state, number_input,
                       selectbox, button, spinner, rerun)

from utils.helper import Timer, scatter_visualiser

empty_messages: empty = empty()
empty_data_title: empty = empty()
empty_data_chart: empty = empty()
empty_data_table: empty = empty()

pre_sessions: list[str] = ["data", "timer_pre", "X", "Y"]
for session in pre_sessions:
    session_state.setdefault(session, None)

with sidebar:
    subheader("Data Preparation Settings")

    if session_state["data"] is None:
        empty_messages.error("Please generate the dataset first.")

        n_samples: int = number_input(
            "Number of Samples",
            min_value=100,
            max_value=1_000,
            value=100,
            step=10,
            help="The total number of points generated.",
        )

        noises: float = number_input(
            "Noise",
            min_value=0.1,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Standard deviation of Gaussian noise added to the data.",
        )

        seed: int = number_input(
            "Random State",
            min_value=0,
            max_value=10_000,
            value=27,
            step=1,
            help="Determines random number generation for dataset creation.",
        )

        statuses: list[bool] = [True, False]
        random_status: bool = selectbox(
            "Shuffle the samples",
            options=statuses,
            index=0,
            help="Whether to shuffle the samples.",
        )

        if button("Generate Moons Dataset", type="primary", width="stretch"):
            with spinner("Generating Moons Dataset", show_time=True, width="stretch"):
                with Timer("Dataset Generation") as t:
                    X, y = make_moons(
                        n_samples=n_samples,
                        noise=noises,
                        random_state=seed,
                        shuffle=random_status,
                    )
                    session_state["X"]: DataFrame = DataFrame(X, columns=["Feature 1", "Feature 2"])
                    session_state["Y"]: DataFrame = DataFrame(y, columns=["Category"])
                    session_state["data"]: DataFrame = concat([session_state["X"], session_state["Y"]], axis=1)
                    session_state["timer_pre"] = t
            rerun()
    else:
        empty_messages.success(f"{session_state["timer_pre"]} Dataset generated successfully.")

        empty_data_title.markdown(f"#### Moons Dataset {session_state['data'].shape}")
        fig = scatter_visualiser(data=session_state["X"], categories=session_state["Y"])
        empty_data_chart.plotly_chart(fig, theme="streamlit", use_container_width=True)
        empty_data_table.data_editor(data=session_state["data"], hide_index=True, disabled=True, width="stretch")

        if button("Reset & Generate Again", type="secondary", width="stretch"):
            for session in pre_sessions:
                session_state[session] = None
            rerun()
