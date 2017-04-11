# Edward中文文档

[github地址](https://github.com/blei-lab/edward) [Paper](https://arxiv.org/pdf/1701.03757v1.pdf) [中文文档地址](https://edward-cn.readthedocs.io/zh/latest/) [Edward 官方英文文档](edwardlib.org)

Edward 是一个用于概率建模、推理和评估的 Python 库。它是一个用于快速实验和研究概率模型的测试平台，其涵盖的模型范围从在小数据集上的经典层次模型到在大数据集上的复杂深度概率模型。Edward 融合了以下三个领域：贝叶斯统计学和机器学习、深度学习、概率编程。

它支持以下方式的建模：

* 定向图模型
* 神经网络（通过[Keras](http://keras.io) 和 [TensorFlowSlim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)等库）
* 条件特定的无向模型
* 贝叶斯非参数和概率程序

它支持以下方式的推理：

- 变分推理（Variational inference）


- 黑箱变分推理
- 随机变分推理
- 包容 KL 散度（Inclusive KL divergence）
- 最大后验估计

蒙特卡洛（Monte Carlo）

- 哈密尔顿蒙特卡罗（Hamiltonian Monte Carlo）
- 随机梯度 Langevin 动态
- Metropolis-Hastings

推理的组成

- 期望最大化（Expectation-Maximization）
- 伪边界和 ABC 方法（Pseudo-marginal and ABC methods）
- 消息传递算法（Message passing algorithms）

它支持以下的模型评估和推理

- 基于点的评估（Point-based evaluations）
- 后验预测检查（Posterior predictive checks）

同时，由于Edward 构建于 TensorFlow 之上。它支持诸如计算图、分布式训练、CPU/GPU 集成、自动微分等功能，也可以用 TensorBoard 可视化。

## 注

欢迎有能力的小伙伴参加翻译（chengzehua@outlook.com)。

## 关于变分推理

我和Dustin关于VI找了许多资料，认为：

1.  [Kucukelbir and Blei (2016)](https://arxiv.org/abs/1601.00670)
2.  [NIPS 2016 tutorial ](https://nips.cc/Conferences/2016/Schedule?showEvent=6199)

十分有帮助，如果有对变分推理有兴趣的同学可以借由这两篇paper去了解一下。

## Resources

- [Edward website](http://edwardlib.org)
- [Edward Forum](http://discuss.edwardlib.org)
- [Edward Gitter channel](http://gitter.im/blei-lab/edward)
- [Edward releases](https://github.com/blei-lab/edward/releases)
- [Edward papers, posters, and slides](https://github.com/edwardlib/papers)