# DCCA

## 一、算法的提出背景

### 1. CCA （Canonical Correlation Analysis, 典型成分分析）：

CCA算法是一种用于分析两个随机变量之间相关性的方法。它通过对两个随机变量进行线性映射，得到若干典型特征对，其目标是使得这些特征对之间的相关系数达到最大。这样，对两个原始随机变量之间相关性的分析，就可以转化为对这些特征对之间相关性的分析。

### 2. KCCA （Kernel CCA）：

CCA算法的缺点在于它只能挖掘线性相关性，而无法反映非线性的相关性。为了弥补这个缺点，提出了KCCA算法，它首先通过非线性映射，将两个随机变量映射到高维空间，然后在高维空间中使用CCA算法。

### 3. DCCA （Deep CCA）：

KCCA算法的局限性在于：

- 无参数的非线性映射限制了可能的表示空间；
- 推断时间随着训练样本数目的增加而快速增加。

为了克服以上局限性，提出了DCCA算法，用深度神经网络来拟合非线性映射。

## 二、问题的形式化

我们的目标是训练两个DNN，用他们分别对两个随机变量$x$和$y$进行非线性映射，使映射后的两个随机变量之间的相关性达到最大。

假设$x\in\R^p,y\in\R^q$，两个DNN所进行的映射分别为$f_x:\R^p\rightarrow\R^o$和$f_y:\R^q\rightarrow\R^o$，变换后的两个随机变量为$x'=f_x(x)\in\R^o$和$y'=f_y(y)\in\R^o$。$x',y'$之间的相关性记作$corr(x',y')$，它可以表示为对$x',y'$进行CCA后，所有典型特征对的相关系数之和。假设$x'$的方差，$y'$的方差和$x'$与$y'$之间的协方差分别为$\Sigma_{11},\Sigma_{22},\Sigma_{12}$，令$T=\Sigma_{11}^{-1/2}\Sigma_{12}\Sigma_{22}^{-1/2}$，则CCA后，所有特征对之间的相关系数之和就等于$T$的特征值之和，从而也就等于$T$的迹。因此，我们的目标就变成了选择两组参数$(\theta_1,\theta_2)$，使得：

$\mathop{\arg\max}\limits_{(\theta_1,\theta_2)}corr(f_x(x|\theta_1),f_y(y|\theta_2))=corr(x',y')=trace(T)$

对$(x',y')$的联合分布进行$n$次采样，得到$X',Y'\in\R^{o\times n}$，对其进行归一化，得到$\overline X=X'-\frac{1}{n}X' \bold 1,\overline Y=Y'-\frac{1}{n}Y' \bold 1$，则$\Sigma_{11},\Sigma_{22},\Sigma_{12}$可以估计如下，其中$r_1,r_2>0$为正则化系数，用于保证$\Sigma_{11},\Sigma_{22}$正定：

$\Sigma_{11}=\frac{1}{n-1}\overline X \overline X^T+r_1I,\Sigma_{22}=\frac{1}{n-1}\overline Y \overline Y^T+r_2I,\Sigma_{12}=\frac{1}{n-1}\overline X \overline Y^T$

## 三、求解方法

### 1. 梯度下降

我们使用梯度下降法来求出$(\theta_1,\theta_2)$的最优解，为此需要求出目标函数$corr(x',y')$对$(\theta_1,\theta_2)$的导数。按照链式法则：

$\frac{\partial corr(x',y')}{\partial\theta_1}=\frac{\partial corr(x',y')}{\partial x'}*\frac{\partial x'}{\partial\theta_1}$

其中，$\frac{\partial x'}{\partial\theta_1}$能够容易地通过反向传播求出，因此关键在于$\frac{\partial corr(x',y')}{\partial x'}$的求解。假设$T$的SVD分解结果为$T=UDV'$，则$\frac{\partial corr(x',y')}{\partial x'}$的计算公式为（具体推导过程见论文附录）：

$\frac{\partial corr(x',y')}{\partial x'}=\frac{1}{n-1}(2\nabla_{11}\overline X+\nabla_{12}\overline Y)$

$\nabla_{11}=-\frac{1}{2}\Sigma_{11}^{-1/2}UDU'\Sigma_{11}^{-1/2}$

$\nabla_{12}=\Sigma_{11}^{-1/2}UV'\Sigma_{22}^{-1/2}$

### 2. 初始化