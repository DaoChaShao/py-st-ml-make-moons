#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/13 15:21
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   about.py
# @Desc     :

from streamlit import title, expander, caption

title("**Application Information**")
with expander("About this application", expanded=True):
    caption("- Provides interactive visualization of data, including scatter plots with predicted labels.")
    caption("- Shows decision boundaries of the trained KNN model on 2D data.")
    caption("- Displays performance metrics: R², Accuracy, Confusion Matrix, Precision, Recall, F1-Score.")
    caption("- Allows resetting dataset, model, or neighbor search to start over.")
