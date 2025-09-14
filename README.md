<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**INTRODUCTION**
---
This app uses the make_moons dataset, a two-class “moon-shaped” dataset often used for testing classification
algorithms. This Streamlit application allows you to explore and train K-Nearest Neighbors (KNN) classifiers
interactively on 2D datasets (e.g., moons dataset). You can adjust dataset parameters, train the model, and find the
optimal number of neighbors for classification performance. You can also adjust parameters such as the number of
neighbors and noise level to see how the decision boundary changes and to better understand the behavior of KNN in
non-linear classification tasks.

**FEATURES**
---

- **Data Generation**: Generate synthetic 2D datasets such as the moons dataset with configurable number of samples,
  noise, and shuffle options.
- **KNN Model Training**: Train a KNN classifier with adjustable number of neighbors.
- **Best Neighbors Finder**: Automatically search for the best number of neighbors that maximizes train and test
  accuracy.
- **Interactive Visualization**: Display train/test scatter plots with predicted labels and decision boundaries.
- **Performance Metrics**: View R², Accuracy, Confusion Matrix, Precision, Recall, and F1-Score for the trained KNN
  model.
- **Reset Options**: Reset dataset, model, or neighbor search to start over.

**WEB DEVELOPMENT**
---

1. Install NiceGUI with the command `pip install streamlit`.
2. Run the command `pip show streamlit` or `pip show streamlit | grep Version` to check whether the package has been
   installed and its version.
3. Run the command `streamlit run app.py` to start the web application.

**PRIVACY NOTICE**
---
This application may require inputting personal information or private data to generate customised suggestions,
recommendations, and necessary results. However, please rest assured that the application does **NOT** collect, store,
or transmit your personal information. All processing occurs locally in the browser or runtime environment, and **NO**
data is sent to any external server or third-party service. The entire codebase is open and transparent — you are
welcome to review the code [here](./) at any time to verify how your data is handled.

**LICENCE**
---
This application is licensed under the [BSD-3-Clause License](LICENSE). You can click the link to read the licence.

**CHANGELOG**
---
This guide outlines the steps to automatically generate and maintain a project changelog using git-changelog.

1. Install the required dependencies with the command `pip install git-changelog`.
2. Run the command `pip show git-changelog` or `pip show git-changelog | grep Version` to check whether the changelog
   package has been installed and its version.
3. Prepare the configuration file of `pyproject.toml` at the root of the file.
4. The changelog style is [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
5. Run the command `git-changelog`, creating the `Changelog.md` file.
6. Add the file `Changelog.md` to version control with the command `git add Changelog.md` or using the UI interface.
7. Run the command `git-changelog --output CHANGELOG.md` committing the changes and updating the changelog.
8. Push the changes to the remote repository with the command `git push origin main` or using the UI interface.