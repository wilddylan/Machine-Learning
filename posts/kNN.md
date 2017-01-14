先来一个简单的例子，我们如何来区分动作类电影与爱情类电影呢？动作片中存在很多的打斗镜头，爱情片中可能更多的是亲吻镜头，所以我们姑且通过这两种镜头的数量来预测这部电影的主题。简单的说，`k-近邻算法`采用了测量不同特征值之间的距离方法进行分类。

> 优点：精度高、对异常值不敏感、无数据输入假定
> 缺点：计算复杂度高、控件复杂度高
> 适用数据范围：数值型和标称型

首先我们来理解它的工作原理：

存在一个样本数据集（训练集），并且我们知道每一数据与目标变量的对应关系，输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较，然后算法提取样本集中最相近的分类标签，一般来说，我们只选择样本集中前k个最相似的数据，通常k为不大于20的整数，最后，选择k个最相似数据中出现次数最多的分类，作为新数据的分类。

现在我们回到文章开头所提到的电影的例子，根据下表如何确定未知的电影类型呢？

| 电影名称 | 打斗镜头 | 接吻镜头 | 电影类型 |
| ---- | ---- | ---- | ---- |
| 电影1  | 3    | 104  | 爱情   |
| 电影2  | 2    | 100  | 爱情   |
| 电影3  | 1    | 81   | 爱情   |
| 电影4  | 101  | 10   | 动作   |
| 电影5  | 99   | 5    | 动作   |
| 电影6  | 98   | 2    | 动作   |
| 电影7  | 18   | 90   | 未知？  |

该如何计算电影7的电影类型呢？计算电影7与样本集中其他电影的距离，然后我们假定k=3，看一下最近的3部电影是什么类型即可预测出电影7的电影类型。

#### 流程介绍

- 收集数据
- 准备数据：距离计算所需要的值，最好是结构化的数据格式
- 分析数据
- 测试算法：计算错误率
- 使用算法

#### 开始工作

###### 使用python导入数据

首先，创建名为kNN.py的Python模块，在构造算法之前，我们还需要编写一些通用的函数等，我们先写入一些简单的代码：

```python
from numpy import *
import operator

def createDataSet():
    group = array([
      [1.0, 1.1], 
      [1.0, 1.0],
      [0, 0],
      [0, 0.1]
    ])

    labels = ["A", "A", "B", "B"]

    return group, labels
```

然后将终端改变到代码文件目录，输入命令python进入到交互界面：

```python
>>> import kNN

>>> group, labels = kNN.createDataSet()

>>> group
array([[ 1. ,  1.1],
       [ 1. ,  1. ],
       [ 0. ,  0. ],
       [ 0. ,  0.1]])
>>> labels
['A', 'A', 'B', 'B']
```

这里有4组数据，每组数据有2个我们已知的特征值，上面的group矩阵每行为一条数据，对于每个数据点我们通常使用2个特征（所以特征的选择很重要），向量labels包含了每个数据点的标签信息，labels的维度等于矩阵的行数，这里我们将`[1, 1,1]`定为类A，`[0, 0.1]`定为类B，接下来我们进行算法的编写：

- 计算已知数据集中点到当前点之间的距离
- 按照距离递增次序排序
- 选取与当前点距离最小的k个点
- 确定前k个点所在类别的出现频率
- 返回前k个点中频次最高的类别作为预测类别

接着定义函数：

```python
# inX: 用于分类的输入向量
# dataSet：输入的训练集
# labels：标签向量
# k：选择近邻项目的个数
def classify0(inX, dataSet, labels, k) :
    dataSetSize = dataSet.shape[0]
    # 距离计算
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2 # python中, **2 代表2平方，**0.5代表开方
    sqDistances = sqDiffMat.sum(axis=1) # 加入axis=1以后就是将一个矩阵的每一行向量相加
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}

    # 选择距离最小的k个点
    for i in range(k) :
        voteILabel = labels[sortedDistIndicies[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
    
    # 排序
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
```

然后我们进行测试，重新打开python编译环境：

```python
>>> import kNN
>>> group, labels = kNN.createDataSet()
>>> kNN.classify0([0, 0], group, labels, 3)
'B'
>>> kNN.classify0([0.3, 0], group, labels, 3)
'B'
>>> kNN.classify0([0.8, 0.9], group, labels, 3)
'A'
```

我们看到，一个简单的分类器就这样搞定了。这时，我们来将电影数据进行样本写入：

```python
def createDataSet():
    group = array([
      [3, 104], 
      [2, 100],
      [1, 81],
      [101, 10],
      [99, 5],
      [98, 2]
    ])

    labels = ["love", "love", "love", "action", "action", "action"]

    return group, labels
```

```python
>>> import kNN
>>> group, labels = kNN.createDataSet()
>>> kNN.classify0([18, 90], group, labels, 3)
'love'
```

我们看到预测结果为爱情片。这是一个简单的分类器，当然，我们应该通过大量的测试，看预测结果与预期结果是否相同，进而得出错误率，完美的分类器错误率为0，最差的错误率为1，上边电影分类的例子已经可以使用了，但是貌似没有太大的作用，下边我们来看一个生活中的实例，使用k-近邻算法改进约会网站的效果，然后使用k-近邻算法改进手写识别系统。

#### 改进约会网站的配对效果

有个姑娘，一直在使用某交友网站，但是推荐来的人总是不尽人意，她总结了一番，曾交往过3中类型的人：

- 不喜欢的人
- 魅力一般的人
- 极具魅力的人

她觉得自己可以在周一~周五约会那些魅力一般的人，周末与那些极具魅力的人约会，因为她希望我们可以更好的帮助她将匹配的对象划分到确切的分类中，她还收集了一些约会网站未曾记录过的数据信息，她认为这些数据信息会有帮助。这些数据存放在代码文件夹下`datingTestSet2.txt`中，总共有1000行数据（妹的约了1000个人），主要包含以下特征：

- 每年获得的飞行常客里程数
- 玩视频游戏所消耗的时间百分比
- 每周消费的冰激凌公升数

我们看到，统计的东西都比较奇葩，我们先从文件中把这些数值解析出来，输出训练样本矩阵和分类标签向量。在kNN.py中继续书写代码：