# 图神经网络`(GNN)`

# 一、为什么需要图神经网络？

随着深度学习的不断发展，在语音、图像、自然语言处理等方面都取得了很大的突破，但是这些数据都是结构化的数据。

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/heS6wRSHVMnLykfIRgjFaE5BibHMARNdE0WyMueqv8OiaSbb5Pj3BFGw0OpkDTtvjM1TlkNgGfpZxJnc2SuTiafSg/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1&tp=wxpic)

然而在现实世界很多事物都是非结构化。![图片](https://mmbiz.qpic.cn/mmbiz_jpg/hN1l83J6Ph9kPBhgZVzQ93qI5MJNfoibYHubKialQoTI1h58iaMBEX0ibJibibrzlJxMofvrF8iaqsNMpJ5jzaLAwOCgA/640?wx_fmt=jpeg&random=0.29603173730734134&wxfrom=5&wx_lazy=1&wx_co=1&tp=wxpic)

相较于结构化的数据，这种网络类型的非结构性的数据非常复杂：

* 图大小是任意的，其拓扑结构是非常复杂的，不像图像那样有空间局部性；

  > 图像的空间局部性是指相邻像素之间的相关性，这种相关性可以被用来设计更加高效的图像处理算法，从而提高程序的性能和效率。

  > 图的拓扑结构是指图中各个节点之间以及节点和边之间的关系所构成的一种结构

* 图没有固定的节点顺序
* 图经常是动态图，而且包含多模态的特征

## 二、图神经网络是怎么样的？

神经网络最基本的结构是全连接层`(MLP)`，特征矩阵乘以权重矩阵，图神经网络多了邻接矩阵。计算形式三个矩阵相乘外加一个非线性变化。

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/hN1l83J6Ph9kPBhgZVzQ93qI5MJNfoibYCm1BFofJI1wJgTZQsWCicXEcZkthNy4IX2dcDG1sY4r1duVLLXPuWhw/640?wx_fmt=jpeg&random=0.07862330424098896&wxfrom=5&wx_lazy=1&wx_co=1&tp=wxpic)

图神经网络的应用模型如图：![图片](https://mmbiz.qpic.cn/mmbiz_jpg/hN1l83J6Ph9kPBhgZVzQ93qI5MJNfoibYUmibuegKADjmfGWFpFB5uNPbELaPbqMWHhEAG6afVpelnib8NcG5PQzg/640?wx_fmt=jpeg&random=0.5882341607916157&wxfrom=5&wx_lazy=1&wx_co=1&tp=wxpic)

输入是一个图，经过多层图卷积等各种操作，最终得到各个节点的表示，以便于进行节点的分类等任务。

## 三、几个经典模型与发展

**1、Graph Convolution Networks`(GCN)`**

它首次将图像处理中的卷积操作简单的用到图结构数据处理中来，并且给出了具体的推导，这里面涉及到复杂的谱图理论。

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/hN1l83J6Ph9kPBhgZVzQ93qI5MJNfoibYN7yzvCKMJqueloTqPNYIQur485yoYxKdv5bw57ZuCTCYpum6HnGkrg/640?wx_fmt=jpeg&random=0.5068990008190379&wxfrom=5&wx_lazy=1&wx_co=1&tp=wxpic)

`GCN`的效果相比较于传统方法提升还是很明显的，这很有可能是得益于GCN善于编码图的结构信息，能够学习到更好的节点表示。

`GCN` 的缺点：

* GCN需要将整个图放到内存和显存
* GCN在训练时需要知道整个图的结构信息(包括待预测的节点), 这在现实某些任务中也不能实现(比如用今天训练的图模型预测明天的数据，那么明天的节点是拿不到的)。

**2、Graph Sample and Aggregate`(GraphSAGE)`**

`GraphSAGE`是一个`Inductive Learning`框架，具体实现中，训练时它仅仅保留训练样本到训练样本的边，然后包含**Sample**和**Aggregate**两大步骤，Sample是指如何对邻居的个数进行采样，Aggregate是指拿到邻居节点的embedding之后如何汇聚这些embedding以更新自己的embedding信息。

> 训练集$D={X_{tr},y_{tr},X_{un}}$，测试$X_{te}$，此时$X_{un},X_{te}$都是未标记的，但是在测试的时候没有使用到$X_{te}$， 测试集不会出现在训练集中，这种叫做`inductive semi-supervised learning`。
>
> 如果上述训练集中的$X_{un}$就是测试集$X_{te}$，由于训练的时候已经利用了测试集的特征信息，这就叫做`transductive semi-supervised learning`

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/hN1l83J6Ph9kPBhgZVzQ93qI5MJNfoibYrZnQ8pe3gWB12GmKHwHh9wBey7wlfQ7QTuLrRqHgKR3zALdMCVsTjQ/640?wx_fmt=jpeg&random=0.2858353758038412&wxfrom=5&wx_lazy=1&wx_co=1&tp=wxpic)

第一步，对邻居采样；

第二步，采样后的邻居embedding传到节点上来，并使用一个聚合函数聚合这些邻居信息以更新节点的embedding；

第三步，根据更新后的embedding预测节点的标签；

如何给一个新的节点生成embedding，算法图如下：

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/hN1l83J6Ph9kPBhgZVzQ93qI5MJNfoibY8Cm41zAWRMYDW9xGVMROVANiaZzlRZHoPZh4tFuVjl5PBfFTWf30Yicw/640?wx_fmt=jpeg&random=0.07282059403446528&wxfrom=5&wx_lazy=1&wx_co=1&tp=wxpic)

首先初始化输入图像中所有节点的特征向量，然后通过$N$函数得到节点$v$的邻域节点，再使用聚合函数聚合邻域节点的信息，最后先通过结合`K-1`层自身的embedding、聚合新得到的embedding和权重，然后通过非线性变换得到`K`层节点`v`的embedding。依次遍历每一层的每一个节点。

`GraphSAGE Sample`是通过采用定长抽样的方法：定义需要的邻居个数**S**，然后采用有放回的重采样/负采样方法达到**S**。

主要使用的聚合器：

* Mean Aggregator
* LSTM Aggregator
* Pooling Aggregator

如何学习聚合器的参数以及权重矩阵W呢？如果再有监督的情况下，可以使用每个节点的预测label和真实的label的交叉熵作为损失函数。无监督的情况，

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/hN1l83J6Ph9kPBhgZVzQ93qI5MJNfoibYnR40AibzN5QtoDmetRLrwq4fKAR0hSicUZqPzLjZZv2XQa4oCMrw4yCw/640?wx_fmt=jpeg&random=0.8498188263857156&wxfrom=5&wx_lazy=1&wx_co=1&tp=wxpic)

`GraphSAGE`的优点：

* 利用采样机制，可以很好地解决GCN训练时内存和显存的限制，即使对于未知的新节点，也能得到其表示；
* 聚合器和权重矩阵的参数对于所有节点都是共享的；
* 模型的参数与图的节点个数无关，可以处理更大的图；
* 既能处理监督任务也能处理非监督任务。

缺点：每个节点那么多邻居，`GraphSAGN`的采样没有考虑到不同邻居节点的重要性不同，而且聚合计算的时候邻居节点的重要性和当前节点也是不同的。

**3、Graph Attention Networks`(GAT)`**

为了解决上述没有考虑到不同邻居节点重要性的不同的问题，引入了`masked self-attention`机制，这样会根据每个邻居节点特征的不同来为其分配不同的权值。

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/hN1l83J6Ph9kPBhgZVzQ93qI5MJNfoibYiaxg0SwT1b81d8eJx5reteEc3mZwDrroKVYrSkF5dkWiayiccLttxAlxw/640?wx_fmt=jpeg&random=0.8844410123476205&wxfrom=5&wx_lazy=1&wx_co=1&tp=wxpic)

其中$a$采用了单层的前馈神经网络实现，计算过程如下（注意权重矩阵$W$对于所有的节点是共享的）：

![图片](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph9kPBhgZVzQ93qI5MJNfoibYOvZ0WES8qWRiawzIy34xfdvcMQyzrIQC3qNQlSI2poUDNPWdMWAnqRw/640?wx_fmt=png&random=0.8772910585259244&wxfrom=5&wx_lazy=1&wx_co=1&tp=wxpic)

计算完attention之后，就可以得到某个节点聚合其邻居节点信息的新的表示，计算过程如下：



![图片](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph9kPBhgZVzQ93qI5MJNfoibYZbiaicyQeNicYLnpJAEyQfibicyicWBmNa0LdAa3Iccoyo52LFrkN9ia2MWcg/640?wx_fmt=png&random=0.4895109532736497&wxfrom=5&wx_lazy=1&wx_co=1&tp=wxpic)

为了提高模型的拟合能力，还引入了多头的self-attention机制，即同时使用多个$W^k$计算self-attention，然后将计算的结果合并（连接或者求和）：

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/hN1l83J6Ph9kPBhgZVzQ93qI5MJNfoibYWUeOaMDrm735UND6WTN7fUhPrrZ7v81GhQu01IJicToKibiaetEqcV7IQ/640?wx_fmt=jpeg&random=0.7914936949267231&wxfrom=5&wx_lazy=1&wx_co=1&tp=wxpic)

此外，由于GAT结构的特性，GAT无需使用预先构建好的图，因此GAT既适用于`Transductive Learning`，又适用于`Inductive Learning`。

`GAT`的优点：

* 训练GCN无需了解到整个图的结构，只需要知道每个结点的邻居节点即可；

  > 知道每个节点的邻居节点不就和了解到整个图一样？

* 计算速度快。可以在每个节点上使用并行计算；
* 既可以用于`Transductive Learning`，又可以用于`Inductive Learning`，可以对未见过的图结构进行处理。

接下来根据具体任务类别来介绍一下流行的GNN模型和方法。

## 四、无监督的节点表示学习

标注数据的成本高，能够使用无监督学习方法来学习到节点的表示，能产生巨大的价值和意义。

**`Graph Auto-Encoder(GAE)`**：

**自编码器`(Auto-encoder)`**：利用反向传播算法使得输出值等于输入值的神经网络，它先将输入压缩成潜在空间表征，然后通过这种表征来重构输出。

主要由两部分组成：

* 编码器**：这部分能将输入压缩成潜在空间表征，可以用编码函数$h=f(x)$表示。

* 解码器**：这部分重构来自潜在空间表征的输入，可以用解码函数$r=g(h)$表示。

![img](https://pic1.zhimg.com/80/v2-ace24887b5ccf1696785bcc7b9abe218_720w.jpg)

自编码器的应用主要有两个方面，第一是**数据去噪**，第二是**为进行可视化而降维**。

![图片](https://mmbiz.qpic.cn/mmbiz_png/hN1l83J6Ph9kPBhgZVzQ93qI5MJNfoibYGZFVxZpDWhiapxrMib3LAiaib94rGvlQ7GwXA0GaP8x02SEG6jBoicKWTiaA/640?wx_fmt=png&random=0.2499754892248034&wxfrom=5&wx_lazy=1&wx_co=1&tp=wxpic)

输入图的邻接矩阵$A$和节点的特征矩阵$X$，通过编码器（图卷积网络）学习节点低维向量表示的均值$μ$和方差$σ$，然后解码器（链路预测）生成图。

编码器（Encoder）采用简单的两层GCN网络，解码器计算两点之间存在边的概率来重构图，损失函数包括生成图和原始图之间的距离度量，以及节点表示向量分布和正态分布的KL-散度两部分。

## 五、Graph pooling

`Graph pooling`是GNN中很流行的一种操作，目的是为了获取一整个图的表示，主要用于处理图级别的分类任务。

**Differentiable Pooling`(Diffpool)`**

在图级别的任务当中，当前的很多方法是将所有的节点嵌入进行全局池化，忽略了图中可能存在的任何层级结构，这对于图的分类任务来说尤其成问题，因为其目标是预测整个图的标签，针对这个问题，提出了用于图分类的可微池化操作模块——`DiffPool`，可以生成图的层级表示，并且可以以端到端的方式被各种图神经网络整合。

`DiffPool`的核心思想是通过一个可微池化操作模块去分层的聚合图节点，具体的，这个可微池化操作模块基于GNN上一层生成的节点嵌入$X$以及分配矩阵$S$，以端到端的方式分配给下一层的簇，然后将这些簇输入到GNN下一层，进而实现用分层的方式堆叠多个GNN层的想法。

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/hN1l83J6Ph9kPBhgZVzQ93qI5MJNfoibY2iamibrvWx77AGyLTPMr1D7IplIGibDbyo3oAMoIiaoHLdXuj7gzwJt7Rw/640?wx_fmt=jpeg&random=0.49115799795421156&wxfrom=5&wx_lazy=1&wx_co=1&tp=wxpic)

`DiffPool`的优点：

* 可以学习层次化的`pooling`策略；
* 可以学习到图的层次化表示；
* 可以以端到端的方式被各种图神经网络整合。

其局限性，分配矩阵需要很大的空间去存储，空间复杂度为$O(kV^2)$，$k$为池化层的层数，无法处理很大的图。