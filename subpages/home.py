#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/13 15:21
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   home.py
# @Desc     :

from streamlit import title, expander, caption, empty

empty_message = empty()
empty_message.info("Please check the details at the different pages of core functions.")

title("ML - Make Moons")
with expander("**INTRODUCTION**", expanded=True):
    caption("+ This app allows you to generate 2D datasets such as moons dataset for classification tasks.")
    caption("+ You can configure dataset parameters: number of samples, noise level, and whether to shuffle.")
    caption("+ Train a K-Nearest Neighbors (KNN) model interactively with adjustable neighbors.")
    caption("+ Find the optimal number of neighbors that maximizes training and testing accuracy.")
