#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/14 13:23
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   finder.py
# @Desc     :   

from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from streamlit import (empty, sidebar, subheader, session_state, slider,
                       caption, spinner, button, rerun, columns, metric)

from utils.helper import Timer

empty_messages: empty = empty()
left, right = columns(2, gap="small")
empty_best_title: empty = empty()
empty_best_chart: empty = empty()
empty_best_table: empty = empty()

pre_sessions: list[str] = ["data"]
for session in pre_sessions:
    session_state.setdefault(session, None)
knn_sessions: list[str] = ["x_train", "x_test", "y_train", "y_test"]
for session in knn_sessions:
    session_state.setdefault(session, None)
finder_sessions: list[str] = ["timer_finder", "trier", "min_value", "max_value"]
for session in finder_sessions:
    session_state.setdefault(session, None)
acc_sessions: list[str] = ["acc_train", "acc_test"]
for session in acc_sessions:
    session_state.setdefault(session, [])

with sidebar:
    if session_state["data"] is None:
        empty_messages.error("Please go to **`Data Preparation`** page to generate the dataset first.")
    else:
        if session_state["x_train"] is None:
            empty_messages.error("Please go to **`Model Training`** page to get the `x_train` data first.")
        else:
            subheader("Best Neighbors Finder Settings")

            if session_state["trier"] is None:
                empty_messages.info("You can adjust the settings below to find the best neighbors.")

                session_state["min_value"]: int = slider(
                    "Minimum Neighbors",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1,
                    help="The minimum number of neighbors to consider.",
                )

                session_state["max_value"]: int = slider(
                    "Maximum Neighbors",
                    min_value=10,
                    max_value=20,
                    value=10,
                    step=1,
                    help="The maximum number of neighbors to consider.",
                )
                caption(
                    f"The best number of k-neighbors will be searched between {session_state["min_value"]} and {session_state["max_value"]}."
                )

                if button("Find the Best Neighbors", type="primary", width="stretch"):
                    with spinner("Finding the Best Neighbors...", show_time=True, width="stretch"):
                        with Timer("Finding the Best Neighbors") as session_state["timer_finder"]:
                            for i in range(session_state["min_value"], session_state["max_value"] + 1):
                                session_state["trier"] = KNeighborsClassifier(n_neighbors=i)
                                session_state["trier"].fit(session_state["x_train"], session_state["y_train"].squeeze())
                                y_pred_train = session_state["trier"].predict(session_state["x_train"])
                                y_pred_test = session_state["trier"].predict(session_state["x_test"])
                                session_state["acc_train"].append(
                                    accuracy_score(session_state["y_train"], y_pred_train)
                                )
                                session_state["acc_test"].append(
                                    accuracy_score(session_state["y_test"], y_pred_test)
                                )
                    rerun()
            else:
                empty_messages.success(f" {session_state["timer_finder"]} The best neighbors have been found.")

                result: DataFrame = DataFrame(
                    {
                        "Train Accuracy": session_state["acc_train"],
                        "Test Accuracy": session_state["acc_test"],
                    },
                    index=list(range(session_state["min_value"], session_state["max_value"] + 1))
                )
                result.index.name = "Number of Neighbors (best_k)"
                empty_best_title.markdown("#### Best Neighbors Results")
                empty_best_chart.line_chart(result, use_container_width=True)
                empty_best_table.data_editor(result, hide_index=True, disabled=True, width="stretch")
                best_index_train = session_state["acc_train"].index(
                    max(session_state["acc_train"])
                ) + session_state["min_value"]
                best_index_test = session_state["acc_test"].index(
                    max(session_state["acc_test"])
                ) + session_state["min_value"]
                with left:
                    metric(
                        label="Best Neighbors on Training Set",
                        value=best_index_train,
                        delta=f"{max(session_state['acc_train']) * 100:.2f}%",
                        delta_color="normal",
                    )
                with right:
                    metric(
                        label="Best Neighbors on Testing Set",
                        value=best_index_test,
                        delta=f"{max(session_state['acc_test']) * 100:.2f}%",
                        delta_color="normal",
                    )

                if button("Reset", type="secondary", width="stretch"):
                    for session in finder_sessions:
                        session_state[session] = None
                    for session in acc_sessions:
                        session_state[session] = []
                    rerun()
