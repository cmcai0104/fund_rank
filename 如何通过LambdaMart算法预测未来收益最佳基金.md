
# 一、XGBoost简介

XGBoost的全称是 **eXtreme Gradient Boosting**，由著名华裔机器学家陈天奇在2014年提出，其参与开发的MXNet是目前Amazon官方指定深度学习框架。

XGBoost是目前对结构化数据进行机器学习领域具有统治性地位的一种算法，也是目前**Kaggle Competition**中最为流行的方法，其流行度甚至超越神经网络。

当然**XGBoost**曾也存在一些缺陷（主要是其训练耗时很长，内存占用比较大），使其在工业领域的流行度远远不及学术圈。

2017年1月**Microsoft** 对**XGBoost**算法的性能进行改进，推出轻量级的**lightGBM**，希望牺牲模型少量准确性的前提下，提升算法性能，特别是连续特征数据,此外它还具有高准确性、支持GPU加速和大规模数据处理能力，取得了不错的成果，而且在accuracy方面基本不输**XGBoost**。所以，目前在工业界已经逐渐替代传统机器学习方法。

## 二、什么是**XGBoost**呢？

陈天奇在[XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)里是这么解释:
>A scalable end-to-end tree boosting system called XGBoost, which is used widely by data scientists to achieve state-of-the-art results on many machine learning challenges. We propose a novel sparsity-aware algorithm for sparse data and weighted quantile sketch for approximate tree learning. More importantly, we provide insights on cache access patterns, data compression and sharding to build a scalable tree boosting system. By combining these insights, XGBoost scales beyond billions of examples using far fewer resources than existing systems.

所以**XGBoost**是一种树的增强方法。那么，下面通过树和增强树逐步介绍**XGBoost**算法。

# 2.1 决策树

决策树是一种基本的分类与回归方法。决策树模型因呈树形结构而得名，在分类问题中，表示基于特征对目标进行分类的过程。

其具有以下优点：
1. 模型具有极强的可读性；
2. 速度快。

目前的决策树算法主要源自于**Quinlan**提出的**ID3**和**C4.5**算法，以及**Breiman**的**CART**算法。


## 2.1.1 ID3和C4.5算法

![image1](http://img4.imgtn.bdimg.com/it/u=2852250536,2959910718&fm=26&gp=0.jpg)
这就是一个**ID3**或者**C4.5**算法结构图。

而**C4.5**算法和**ID3**算法的区别是损失函数的选择，在**ID3**中选用**信息增益**作为树对哪个特征、特征的哪个位置进行分叉的指标，而在**C4.5**算法中采用**信息增益比**作为指标，其能有效解决训练数据集的经验熵大的时候，信息增益会偏大，反之亦然。

下面如何计算信息增益和信息增益比：
首先，假设$X$是一个取值有限个的离散随机变量，其概率分布为：$ P(X = x_{i}) = p_{i}, i = 1,2,...,n $
则，随机变量的熵定义为$H(X) = -\sum_{i=1}^{n} p_{i} \log{p_{i}}$.

然后，假设有随机变量$(X, Y),$ 其联合分布为$P(X=x_{i}, Y=y_{i}) = p_{ij}, i=1,2,...,n; j=1,2,...,m$. 称已知随机变量$X$的条件下，随机变量$Y$的不确定性为条件熵$H(Y|X),$ 其数学定义为：$H(Y|X) = \sum_{i=1}^{n} p_{i}H(Y|X=x_{i})$.

下面定义信息增益，即$g(D, A) = H(D) - H(D|A),$ 其中$D$表示数据集，而$A$表示特征。信息增益表示如果按特征$A$对数据集$D$进行分类后对数据集$D$熵降低了多少。

而信息增益比：$g_{R}(D, A) = \frac{g(D, A)}{H(D)}$


## 2.1.2 CART算法

![image2](http://ask.qcloudimg.com/http-save/yehe-2854634/xfd5zwqa8v.jpeg)

**注：该图竖更好，baidu没找到正确的图，将就下**

**CART树**相对于**ID3**和**C4.5**的主要区别有：
1. 二叉树
2. 左：Yes；右：No

CART树既可以做回归模型也可以做分类模型，是因为其模型即可以是回归形式：$f(x) = \sum_{m=1}^{M} c_{m}I(x\in R_{m})$, 其中$\{R_{m}, m=1,2,...,M \}$是对特征空间的一个划分，通过$\sum (y_{i} - f(x_{i}))^2$作为损失函数进行训练，找到使其最小的$\hat{c}_{m} = ave(y_{i}| x_{i}\in R_{m})$。

也可以是分类形式，此时采用**Gini index**。假设数据集有K个类，样本点属于第k类的概率为$p_k,$ 则概率分布的基尼指数定义为：$Gini(p) = \sum_{k=1}^{K}p_{k}(1-p_{k}).$ 则给定的样本集合$D$的基尼指数为$Gini(D) = 1-\sum_{k=1}^{K}(\frac{\| C_{k} \|}{\| D \|})^2$, $C_{k}$是属于第k类的样本子集，K是类的个数。然后，如果样本集合D根据特征A分为两类$D_1, D_2,$ 则在特征A的条件下，集合D的基尼指数定义为：$Gini(D, A) = \frac{|D_{1}|}{|D|}Gini(D_{1})+\frac{|D_{2}|}{|D|}Gini(D_{2})$.


注：
1. 由于具体树的生成过程和树的剪枝问题（防止过拟合）不是本文关注的内容，不在此详细叙述。
2. CART树和Dini指数是目前决策树算法的主流方式。

# 2.2 树的Ensemble

首先，什么树的Ensemble，如下图：

![](http://img2.imgtn.bdimg.com/it/u=2852626919,3032241439&fm=26&gp=0.jpg)

Ensemble其实是生成多棵树的，并结合在一起，目的是提升算法的精确性。

例如：对于要对某一分类问题训练分类器，如果训练一个分类器器准确性为70%，则分类器的准确性是70%；但是如果同时训练5个独立的分类器其准确性都是70%，那么把他们结合在一起准确性就能达到83.2%，训练11个模型准确性能达到99.9%。

Ensemble方法目前被广泛使用，例如：GBM算法，随机森林算法等。基本上超过50%的数据挖掘比赛都使用树的ensemble方法。且ensemble树可以还有以下优势：
1. 对输入不需要normalization（树的特点）
2. 发现特征间的更深层的关系
3. scalable

# 2.3 Boosting Tree

Ensemble是目前最为流行的提升算法精确性的方法。正如前面提到的，目前树ensemble的主要方式有两种：GBM和random forest。那么两种的区别是什么？GBM的全称是Gradient Boosting Tree，首先比较提升树和随机森林的区别，再讨论梯度的意思。

根据前面的叙述已经知道，ensemble是把很多模型集合到一起，而这两者的区别是这些小模型是否独立。

随机森林通过对统一问题独立生成不同的相互独立的树，构成森林；然后根据怎么多树同时对目标进行判断，对结果进行统计，判断出结果的概率。

而提升树方法恰恰相反。其构造的是一个决策树加法模型：$f_M(x) = \sum_{m=1}^{M} T(x; \theta_m),$ 其中$T(x; \theta_m)$表示一颗树，而$\theta_m$s是其参数，而M表示树的个数。

其产生的方法也是逐步的。首先假设$f_0 = 0,$ 第m步模型是$f_m(x) = f_{m-1}(x) + T(x; \theta_m),$ 通过估计$\hat{\theta}_m = argmin \sum_{i=1}^N L(y_i, f_{m-1}(x_i)+T(x_i; \theta_m))$

# 2.4 Gradient Boosting Tree

所谓梯度提升法是Freidman提出的意在解决非平方损失和指数损失函数时的函数优化问题。其核心是利用损失函数的负梯度在当前模型的值：
$$-[\frac{\partial L(y, f(x_i))}{\partial f(x_i)} ]_{f(x)=f_{m-1}(x)}$$
作为回归问题提升树算法中残差的近视值，从而拟合一个回归树。

# 三、MART
**MART** (Multiple Additive Regression Tree)
>MART is an implementation of the gradient tree boosting methods for predictive data mining.

 所以，*MART* 是一种输出是一系列回归树的线性组合的增强树模型。
 
 MART是一种集成学习算法，不同于经典的集成学习算法Adaboost利用前一轮学习器的误差来更新下一轮学习的样本权重，MART每次都拟合上一轮分类器产生的残差。

# 四、排序问题
排序目前在很多应用场景中占据核心地位，最直接的就是搜索引擎，当用户提交一个query，搜索引擎召回很多文档，然后根据文档与query以及用户的相关性对文档进行排序，所以排序算法直接决定搜索引擎的用户体验，这也是Google的看家本领。目前排序算法还在在线广告、协同过滤、多媒体检索等推荐领域发挥巨大作用。

LTR（Learning to Rank） 算法通常有三种手段，分别是：Pointwise、Pairwise 和 Listwise。Pointwise 和 Pairwise 类型的 LTR 算法，将排序问题转化为回归、分类或者有序分类问题。Listwise 类型的 LTR 算法则另辟蹊径，将用户查询（Query）所得的结果作为整体，作为训练用的实例（Instance）。

PointWise方法只考虑给定查询下，单个文档的绝对相关度，而不考虑其他文档和给定查询的相关度。即给定查询q的一个真实文档序列，我们只需要考虑单个文档$d_i$和该查询的相关程度$c_i$.

Pairwise方法考虑给定查询下，两个文档之间的相对相关度。即给定查询q的一个真实文档序列，我们只需要考虑任意两个相关度不同的文档之间的相对相关度：$d_i>d_j$. 相比Pointwise，Pairwise方法通过考虑两两文档之间的相对相关度来进行排序，有一定的进步。但是，Pairwise使用的这种基于两两文档之间相对相关度的损失函数，和真正衡量排序效果的一些指标之间，可能存在很大的不同，有时甚至是负相关. 另外，有的Pairwise方法没有考虑到排序结果前几名对整个排序的重要性，也没有考虑不同查询对应的文档集合的大小对查询结果的影响(但是有的Pairwise方法对这些进行了改进，比如IR SVM就是对Ranking SVM针对以上缺点进行改进得到的算法).

与Pointwise和Pairwise方法不同，Listwise方法直接考虑给定查询下的文档集合的整体序列，直接优化模型输出的文档序列，使得其尽可能接近真实文档序列. 所以比较好的克服了以上算法的缺陷。

# 4.1 RankNet

RankNet的核心是损失函数的定义，其模型可以是任意的可微函数，可以是神经网络，也可以是其它一般模型，这里采用boosted trees。所以我们不妨定义模型为$f(x).$ 下面分析如何通过定义损失函数反应模型的rank特点。
假设此时有两条样本分别为 $x_i$和 $x_j$，通过模型 $f$，它们的score分别为 $s_i$和 $s_j$。令$x_i \rightarrow x_j$表示样本$x_i$在未来的收益比$x_j$高，因此定义：
$$P_{ij}=P(x_i \rightarrow x_j)=\frac{1}{1+e^{-\sigma(s_i-s_j)}}$$
其中，$\sigma$是sigmoid函数的shape参数。

下面通过交叉熵损失函数，定义损失函数：
$$C_{ij}=-\bar{P}_{ij}\log P_{ij} - (1-\bar{P}_{ij})\log (1-P_{ij})$$
其中$\bar{P}_{ij}$是样本$x_i$在未来的收益比$x_j$高的已知概率。

如果定义：
$$ S_{ij} =\left\{
\begin{aligned}
& 1 & 样本i未来收益比j高; \\
& 0 & 样本i和j一样高;  \\
& -1 & 样本i未来收益比j低. 
\end{aligned}
\right.
$$
所以$\bar{P}_{ij} = \frac{1}{2}(1+S_{ij}).$
那么：
$$C_{ij}= \frac{1}{2}(1-S_{ij})\sigma(s_i - s_j) + log(1+e^{-\sigma(s_i-s_j)})$$
然后计算损失函数关于score（$s_i$）的梯度：
$$\lambda_{ij} = \frac{\partial C_{ij}}{\partial s_i} = \sigma(\frac{1}{2}(1-S_{ij})-\frac{1}{1+e^{\sigma(s_i - s_j)}}) = -\frac{\partial C_{ij}}{\partial s_i}$$

再根据链式法则，可以知道梯度下降法：
$$w_k \rightarrow w_k-\eta\frac{\partial C_{ij}}{\partial w_k} = w_k - \eta(\frac{\partial C_{ij}}{\partial s_i}\frac{\partial s_i}{\partial w_k}+\frac{\partial C_{ij}}{\partial s_j}\frac{\partial s_j}{\partial w_k})$$

# 4.2 度量方法
如何刻画结果的好坏是模型成败的关键。在排序领域刻画评分指标主要包括：
1. MRR(Mean Reciprocal Rank)，平均倒数排名
2. MAP(Mean Average Percision)，平均正确率均值
3. NDCG(Normalized Discounted Cumulative Gain)，标准折现累计增益
4. ERR(Expected Reciprocal Rank)，预期倒数排名

下面说明从数学上如何定义，使得它们能够刻画排序问题。
## 4.2.1 MRR
$$MRR = \frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{rank_i}$$
其中$|Q|$表示查询的数量，$rank_i$表示第一个相关结果的排序位置。
## 4.2.2 MAP
假设信息需求 $q_j$对应的所有相关文档集合为$d_1,...,d_{mj}, \ R_{jk}$是返回结果中直到遇到$d_k$后其所在位置前（含$d_k$）的所有文档的集合，则定义$MAP(Q)$如下：
$$MAP(Q) = \frac{1}{|Q|}\sum_{j=1}^{|Q|}\frac{1}{m_j}\sum_{k=1}^{m_j}Precision(R_{jk})$$

## 4.2.3 NDCG
$$DCG@T = \sum_{i=1}^{T}\frac{2^{l_i}-1}{log(i+1)}$$
其中 $T$表示结尾等级，如果我们只关注前几个目标，则可以取$T=10.$ $l_i$这里表示未来收益等级（在搜索算法中常表示结果与目标的相关性）。定义
$$NDCG@T = \frac{DCG@T}{maxDCG@T}$$
达到标准化的目的。
## 4.2.4 ERR
ERR旨在解决NDCG未考虑排在前面结果的影响的缺点。提出一种基于级联模型的评价指标。首先定义：
$$R(g) = \frac{2^g - 1}{2^{g_{max}}}$$
$g$表示文档的得分级别，$g_{max}$表示最大的分数级别。
然后定义：
$$ERR = \sum_{r=1}^{n}\frac{1}{r}\prod_{i=1}^{r-1}(1-R_i)R_r$$


通过上述可以发现MRR和MAP只能对二级的相关性进行评分（即：相关和不相关），而NDCG和ERR则可以对多级相关性进行评分。NDCG和ERR的也更加关注排名靠前的目标。但是它们都是不连续、不可微函数，所以要加入到用梯度下降求解模型中是不可行的。LambdaRank旨在解决这一问题。这里主要采用NDCG，目前xgboost还未支持ERR度量。

# 4.3 LambdaRank
先看一张论文原文中的图，如下所示。这是一组用二元等级相关性进行排序的链接地址，其中浅灰色代表链接与query不相关，深蓝色代表链接与query相关。 对于左边来说，总的pairwise误差为13，而右边总的pairwise误差为11。但是大多数情况下我们更期望能得到左边的结果。这说明最基本的pairwise误差计算方式并不能很好地模拟用户对搜索引擎的期望。右边黑色箭头代表RankNet计算出的梯度大小，红色箭头是期望的梯度大小。NDCG和ERR在计算误差时，排名越靠前权重越大，可以很好地解决RankNet计算误差时的缺点。但是NDCG和ERR均是不可导的函数，对于已知结果可以通过NDCG或者ERR评价,如何加入到RankNet的梯度计算中去呢？
![](https://img2018.cnblogs.com/blog/436630/201810/436630-20181008202412844-1605451333.png)
根据上面在迭代逼近参数的过程中，是通过$\lambda_{ij}，$即损失函数关于模型score的梯度，这里通过直接利用$\lambda_{ij}，$来避免梯度缺失的问题。此时定义:
$$\lambda_{ij} = \frac{-\sigma}{1+e^{\sigma(s_i-s_j)}}|\bigtriangleup_{NDCG}|$$
其中$|\bigtriangleup_{NDCG}|$表示交换$i$和$j$的顺序带来的NDCG的改变量。



# 4.4 LambdaMART

LambdaMART 是一种 Listwise 类型的 LTR 算法，出自微软的Chris Burges，是近期非常热门的算法，屡次出现在机器学习大赛，在Yahoo的Learning to Rank Challenge比赛中夺冠，据说Bing和Facebook采用的就是这个模型。

LambdaMART 是基于 LambdaRank 算法和 MART (Multiple Additive Regression Tree) 算法，将搜索引擎结果排序问题转化为回归决策树问题。

而传统的 LTR 算法（例如：RankNet，一种pairwise rank）由于损失函数梯度不存在，无法使用梯度下降法。而LambdaMART 使用一个特殊的 Lambda 值来代替上述梯度，也就是将 LambdaRank 算法与 MART 算法加和起来。通过Lambda把不适合用梯度下降求解的Rank问题转化为对概率交叉熵损失函数的优化问题，从而适用梯度下降法。