<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**应用简介**
---
本应用使用了 make_moons 数据集，这是一个常用于测试分类算法的双类“月牙形”数据集。这个 Streamlit 应用程序允许你在二维数据集（如
moons 数据集）上交互式地探索和训练 K-Nearest Neighbors (KNN)
分类器。你可以调整数据集参数、训练模型，并找到分类性能最优的邻居数量。你还可以调整邻居数量和噪声水平等参数，以观察决策边界的变化，更好地理解
KNN 在非线性分类任务中的表现。

**特色功能**
---

- **数据生成**：生成二维合成数据集，例如 moons 数据集，可配置样本数量、噪声强度和是否打乱。
- **KNN 模型训练**：训练 KNN 分类器，可调节邻居数量。
- **最佳邻居搜索**：自动搜索能最大化训练集和测试集准确率的邻居数量。
- **交互式可视化**：显示训练集/测试集散点图，包含预测标签和决策边界。
- **性能指标**：查看训练模型的 R²、准确率、混淆矩阵、精确率、召回率及 F1 分数。
- **重置选项**：可重置数据集、模型或邻居搜索，重新开始操作。

**网页开发**
---

1. 使用命令`pip install streamlit`安装`Streamlit`平台。
2. 执行`pip show streamlit`或者`pip show git-streamlit | grep Version`检查是否已正确安装该包及其版本。
3. 执行命令`streamlit run app.py`启动网页应用。

**隐私声明**
---
本应用可能需要您输入个人信息或隐私数据，以生成定制建议和结果。但请放心，应用程序 **不会**
收集、存储或传输您的任何个人信息。所有计算和数据处理均在本地浏览器或运行环境中完成，**不会** 向任何外部服务器或第三方服务发送数据。

整个代码库是开放透明的，您可以随时查看 [这里](./) 的代码，以验证您的数据处理方式。

**许可协议**
---
本应用基于 **BSD-3-Clause 许可证** 开源发布。您可以点击链接阅读完整协议内容：👉 [BSD-3-Clause License](./LICENSE)。

**更新日志**
---
本指南概述了如何使用 git-changelog 自动生成并维护项目的变更日志的步骤。

1. 使用命令`pip install git-changelog`安装所需依赖项。
2. 执行`pip show git-changelog`或者`pip show git-changelog | grep Version`检查是否已正确安装该包及其版本。
3. 在项目根目录下准备`pyproject.toml`配置文件。
4. 更新日志遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/v1.0.0/) 提交规范。
5. 执行命令`git-changelog`创建`Changelog.md`文件。
6. 使用`git add Changelog.md`或图形界面将该文件添加到版本控制中。
7. 执行`git-changelog --output CHANGELOG.md`提交变更并更新日志。
8. 使用`git push origin main`或 UI 工具将变更推送至远程仓库。
