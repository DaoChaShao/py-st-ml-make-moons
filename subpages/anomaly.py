#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/13 16:05
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   anomaly.py
# @Desc     :

from pandas import DataFrame, concat
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from streamlit import (empty, sidebar, subheader, session_state, button,
                       spinner, number_input, rerun, selectbox, columns,
                       metric, plotly_chart, markdown, caption)

from utils.helper import Timer, scatter_visualiser, decision_boundary_adder

empty_messages: empty = empty()
empty_perf_title: empty = empty()
left, right = columns(2, gap="small")
empty_cm_title: empty = empty()
A, B, C, D, E, F = columns(6, gap="small")
before, after = columns(2, gap="small")
empty_importance_title: empty = empty()
empty_importance_chart: empty = empty()
empty_anomaly_title: empty = empty()
empty_anomaly_chart: empty = empty()
empty_anomaly_table: empty = empty()

pre_sessions: list[str] = ["data", "X", "Y"]
for session in pre_sessions:
    session_state.setdefault(session, None)
anomaly_sessions: list[str] = ["elliptic_model", "timer_anomaly"]
for session in anomaly_sessions:
    session_state.setdefault(session, None)
pca_sessions: list[str] = ["pca", "timer_pca", "X_scaled"]
for session in pca_sessions:
    session_state.setdefault(session, None)
knn_sessions: list[str] = ["knn", "timer_knn", "x_train", "x_test", "y_train", "y_test"]
for session in knn_sessions:
    session_state.setdefault(session, None)

with sidebar:
    if session_state["data"] is None:
        empty_messages.error("Please go to 'Data Preparation' page to generate the dataset first.")
    else:
        subheader("Model Training Settings")
        empty_messages.info("Dataset is ready. You can add model training settings here.")

        empty_anomaly_title.markdown(f"#### Moons Data Preview")
        fig = scatter_visualiser(session_state["X"], session_state["Y"])
        empty_anomaly_chart.plotly_chart(fig, theme="streamlit", use_container_width=True)

        if session_state["elliptic_model"] is None:
            seed: int = number_input(
                "Random State",
                min_value=0,
                max_value=10_000,
                value=27,
                step=1,
                help="Determines random number generation for model training.",
            )
            contamination: float = number_input(
                "Contamination",
                min_value=0.01,
                max_value=0.5,
                value=0.02,
                step=0.01,
                help="The amount of contamination of the data set, i.e., the proportion of outliers in the data set.",
            )

            if button("Train the Elliptic Envelope Model", type="primary", width="stretch"):
                with spinner("Training the model..."):
                    with Timer("Elliptic Envelope Model Training") as t:
                        session_state["elliptic_model"] = EllipticEnvelope(
                            contamination=contamination, random_state=seed
                        )
                        session_state["elliptic_model"].fit(session_state["X"])
                        session_state["timer_anomaly"] = t
                empty_messages.success("Model training completed!")
                rerun()
        else:
            pred_arr = session_state["elliptic_model"].predict(session_state["X"])
            pred: DataFrame = DataFrame(pred_arr, columns=["Anomaly"])
            table: DataFrame = concat([session_state["X"], session_state["Y"], pred], axis=1)

            empty_anomaly_title.markdown(f"#### Anomaly Detection Results")
            empty_anomaly_table.data_editor(table, hide_index=True, disabled=True, width="stretch")
            fig = scatter_visualiser(session_state["X"], pred)
            empty_anomaly_chart.plotly_chart(fig, theme="streamlit", use_container_width=True)

            if session_state["pca"] is None:
                empty_messages.success(
                    f"{session_state['timer_anomaly']} You can check the importance of the features using PCA."
                )

                if button("Display PCA Feature Importance", type="primary", width="stretch"):
                    with spinner("Performing PCA..."):
                        with Timer("PCA Analysis") as t:
                            scaler = StandardScaler()
                            session_state["X_scaled"] = scaler.fit_transform(session_state["X"])

                            session_state["pca"] = PCA()
                            session_state["pca"].fit(session_state["X_scaled"])

                            session_state["timer_pca"] = t
                    rerun()
            else:
                ratio = session_state["pca"].explained_variance_ratio_
                importance: DataFrame = DataFrame(
                    ratio,
                    index=[f"PC{i + 1}" for i in range(len(ratio))],
                    columns=["Explained Variance Ratio"]
                )
                empty_importance_title.markdown("#### PCA Feature Importance")
                empty_importance_chart.bar_chart(importance, use_container_width=True)

                if session_state["knn"] is None:
                    empty_messages.info(f"{session_state["timer_pca"]} You can train the KNN model!")

                    size: float = number_input(
                        "Test Size",
                        min_value=0.1,
                        max_value=0.5,
                        value=0.3,
                        step=0.1,
                        help="The proportion of the dataset to include in the test split."
                    )

                    state: int = number_input(
                        "Random State",
                        min_value=0,
                        max_value=10_000,
                        value=27,
                        step=1,
                        help="Controls the shuffling applied to the data before applying the split.",
                    )

                    randomness: bool = selectbox(
                        "Shuffle the data before splitting",
                        options=[True, False],
                        index=0,
                        help="Whether or not to shuffle the data before splitting.",
                    )

                    n_neighbors: int = number_input(
                        "Number of Neighbors",
                        min_value=1,
                        max_value=20,
                        value=3,
                        step=1,
                        help="Number of neighbors to use by default for neighbors queries.",
                    )
                    caption("**8** neighbours could be a good choice for this dataset.")

                    (
                        session_state["x_train"], session_state["x_test"],
                        session_state["y_train"], session_state["y_test"],
                    ) = train_test_split(
                        session_state["X_scaled"], session_state["Y"],
                        test_size=size, random_state=state, shuffle=randomness
                    )

                    if button("Train the KNN Model", type="primary", width="stretch"):
                        with spinner("Training the KNN model..."):
                            with Timer("KNN Model Training") as t:
                                session_state["knn"] = KNeighborsClassifier(n_neighbors=n_neighbors)
                                session_state["knn"].fit(session_state["x_train"], session_state["y_train"].squeeze())
                                session_state["timer_knn"] = t
                        rerun()
                else:
                    empty_messages.success(f"{session_state['timer_knn']} KNN model training completed!")

                    pred_train = session_state["knn"].predict(session_state["x_train"])
                    pred_test = session_state["knn"].predict(session_state["x_test"])
                    r2_train = r2_score(session_state["y_train"], pred_train)
                    r2_test = r2_score(session_state["y_test"], pred_test)
                    acc_train = accuracy_score(session_state["y_train"], pred_train)
                    acc_test = accuracy_score(session_state["y_test"], pred_test)

                    empty_perf_title.markdown("#### KNN Model Performance Metrics")
                    with left:
                        metric(
                            "Train R² Score based on the KNN model: Aim to 1",
                            f"{r2_train:.2%}", delta=f"{1 - r2_train:.2%}",
                            delta_color="inverse"
                        )
                        metric(
                            "Train Accuracy based on the KNN model",
                            f"{acc_train:.2%}", delta=f"{1 - acc_train:.2%}",
                            delta_color="inverse"
                        )
                    with right:
                        metric(
                            "Test R² Score based on the KNN model: Aim to 1",
                            f"{r2_test:.2%}", delta=f"{1 - r2_test:.2%}",
                            delta_color="inverse"
                        )
                        metric(
                            "Test Accuracy based on the KNN model",
                            f"{acc_test:.2%}", delta=f"{1 - acc_test:.2%}",
                            delta_color="inverse"
                        )

                    empty_cm_title.markdown("#### Confusion Matrix for KNN Model")
                    cm = confusion_matrix(session_state["y_test"], pred_test)
                    TP, TN, FP, FN = cm[1, 1], cm[0, 0], cm[0, 1], cm[1, 0]
                    with A:
                        # 在整体样本中，预测正确样本的比例
                        accuracy = (TP + TN) / cm.sum()
                        metric("Accuracy", f"{accuracy:.2%}", delta=f"{1 - accuracy:.2%}", delta_color="inverse")
                    with B:
                        # 在正样本中，预测正确样本的比例
                        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
                        metric("Recall", f"{recall:.2%}", delta=f"{1 - recall:.2%}", delta_color="inverse")
                    with C:
                        # 在负样本中，预测正确样本的比例
                        spec = TN / (TN + FP) if (TN + FP) != 0 else 0
                        metric("Specificity", f"{spec:.2%}", delta=f"{1 - spec:.2%}", delta_color="inverse")
                    with D:
                        # 在预测为正样本中，实际为正样本的比例
                        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                        metric("Precision", f"{precision:.2%}", delta=f"{1 - precision:.2%}", delta_color="inverse")
                    with E:
                        # F1-Score是精确率和召回率的调和平均数
                        f1_score_i = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
                        metric("F1-Score", f"{f1_score_i:.2%}", delta=f"{1 - f1_score_i:.2%}", delta_color="inverse")
                    with F:
                        f1_score_ii = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
                        metric("F1-Score", f"{f1_score_ii:.2%}", delta=f"{1 - f1_score_ii:.2%}", delta_color="inverse")

                        X_TRAIN: DataFrame = DataFrame(session_state["x_train"], columns=["Feature 1", "Feature 2"])
                X_TEST: DataFrame = DataFrame(session_state["x_test"], columns=["Feature 1", "Feature 2"])
                pred_TRAIN: DataFrame = DataFrame(pred_train, columns=["Predicted"])
                pred_TEST: DataFrame = DataFrame(pred_test, columns=["Predicted"])
                with before:
                    markdown("#### Train Chart Preview Before Training")
                    fig_before_train = scatter_visualiser(X_TRAIN, session_state["y_train"])
                    plotly_chart(fig_before_train, theme="streamlit", use_container_width=True)
                    markdown("#### Test Chart Preview Before Training")
                    fig_before_test = scatter_visualiser(X_TEST, session_state["y_test"])
                    plotly_chart(fig_before_test, theme="streamlit", use_container_width=True)
                with after:
                    markdown("#### Train Chart Preview After Training")
                    fig_after_train = scatter_visualiser(X_TRAIN, pred_TRAIN)
                    fig_after_train = decision_boundary_adder(fig_after_train, session_state["knn"], X_TRAIN)
                    plotly_chart(fig_after_train, theme="streamlit", use_container_width=True)
                    markdown("#### Test Chart Preview After Training")
                    fig_after_test = scatter_visualiser(X_TEST, pred_TEST)
                    fig_after_test = decision_boundary_adder(fig_after_test, session_state["knn"], X_TEST)
                    plotly_chart(fig_after_test, theme="streamlit", use_container_width=True)

                if button("Reset the KNN Model", type="secondary", width="stretch"):
                    for session in knn_sessions:
                        session_state[session] = None
                    rerun()

            if button("Reset the PCA Model", type="secondary", width="stretch"):
                for session in pca_sessions:
                    session_state[session] = None
                rerun()

        if button("Retrain the Elliptic Envelope Model", type="secondary", width="stretch"):
            for session in anomaly_sessions:
                session_state[session] = None
            rerun()
