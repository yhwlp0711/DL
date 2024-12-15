# Others

## 1.GQA(Grouped query attention)

在推理阶段，生成第 ``i+1`` 个 token 时，需要前``i``个token的 $K(W_k\times X)$ 和 $V(W_v\times X)$，以及第 ``i`` 个 token 的 $Q(W_q\times X)$。  
为了避免每次生成新 token 时重复计算 $K$ 和 $V$ ，将每个 token 的 $K$ 和 $V$ 保存下来，即 KVCache。

但是存储每个 token 的 $K、V$ 矩阵对内存占用较高，对于 `num_heads=8` 的多头注意力机制，可以将每两个 head 分为一组，共享相同的 $W_k$ 和 $W_v$ 矩阵，而每组内的 $W_q$ 矩阵保持独立。  
这样每组中的head生成相同的 $K$ 和 $V$，但具有不同的查询向量 $Q$。减少内存占用的同时（**也许**）也能减少矩阵乘法的次数 ($W_k\times X、W_v\times X、Q\times K^T$).

## 2.RoPE

传统的在Embedding上加入位置编码，而RoPE在Q、K矩阵上加入位置编码。

### 2.1假设词嵌入为二维  

对于 token $x_m$，$x_n$，有：  
$$x_m'=W_qx_me^{im\theta}=q_me^{im\theta}$$
$$x_n'=W_kx_ne^{in\theta}=k_ne^{in\theta}$$
其中 $\theta$ 是一个可学习的参数，$m$ 和 $n$ 分别是 token 的位置，$i$ 是虚数单位。
将 q_m 与 k_n 表示为复数，有：
$$q_m=q_m^1+iq_m^2$$
$$k_n=k_n^1+ik_n^2$$
而根据欧拉公式，有：
$$e^{i\theta}=cos\theta+isin\theta$$
则：
$$q_me^{im\theta}=(q_m^1+iq_m^2)(cosm\theta+isinm\theta)$$
$$q_me^{im\theta}=(q_m^1cosm\theta-q_m^2sinm\theta)+i(q_m^1sinm\theta+q_m^2cosm\theta)$$
再表示为向量形式：
$$q_me^{im\theta}=\begin{bmatrix}cosm\theta & -sinm\theta\\sinm\theta & cos\theta\end{bmatrix}\begin{bmatrix}q_m^1\\q_m^2\end{bmatrix}$$
同理：
$$k_ne^{in\theta}=\begin{bmatrix}cosn\theta & -sinn\theta\\sinn\theta & cos\theta\end{bmatrix}\begin{bmatrix}k_n^1\\k_n^2\end{bmatrix}$$
故：
$$x_m'^Tx_n'=(q_me^{im\theta})^T(k_ne^{in\theta})=\begin{bmatrix}q_m^1 & q_m^2\end{bmatrix}\begin{bmatrix}cosm\theta & -sinm\theta\\sinm\theta & cos\theta\end{bmatrix}^T\begin{bmatrix}cosn\theta & -sinn\theta\\sinn\theta & cos\theta\end{bmatrix}\begin{bmatrix}k_n^1\\k_n^2\end{bmatrix}$$
化简得：  
$$x_m'^Tx_n'=\begin{bmatrix}q_m^1 & q_m^2\end{bmatrix}\begin{bmatrix}cos((m-n)\theta) & -sin((m-n)\theta)\\sin((m-n)\theta) & cos((m-n)\theta)\end{bmatrix}\begin{bmatrix}k_n^1\\k_n^2\end{bmatrix}$$
直接从位置函数的角度考虑：  
$$g(x_m'^T,x_n',m-n)=Re[(W_qx_m)^T(W_kx_n)^*e^{i(m-n)\theta}]$$
其中 Re 表示取实部，$x^*$表示共轭，$m-n$表示位置差。
![img](./src/RoPE1.png)  
如图，在二维空间中，q 被旋转了 m 角度，k 被旋转了 n 角度，两者的点积就是他们在旋转后的位置的点积。  
$R(m)R(n)^T$即为先旋转 m 角度，再旋转 -n 角度，等价于旋转 m-n 角度。

### 2.2推广到多维

两两一组，如图所示：
![img](./src/RoPE.png)
其中 $\theta_i=1/10000^{2i/d}$。

## 3.SwiGLU(Switched Gated Linear Unit)

|激活函数|公式|特点|
|:-:|:-:|:-:|
|GLU|$GLU(x)=(xW_1+b_1)\times \sigma (xW_2+b_2)$|使用Sigmoid，计算复杂的较高|
|GELU|$GELU(x)=x\times \Phi(x)$|平滑激活，但无门控机制|
|SwiGLU|$SwiGLU(x)=Swish_\beta (xW_1+b_1) \times (xW_2+b_2)$|引入 Swish 和门控机制，性能优异，计算效率较高|

其中$\Phi (x)$为伯努利分布，$Swish_\beta(x)=x \times \sigma (\beta x)$，$\beta$ 为可学习参数，GLU与SwiGLU中都是矩阵逐元素相乘。

## 4.MOE(Mixture of Experts)

核心思想是稀疏激活（Sparse Activation），即并不是每次都激活所有专家，而是根据输入动态选择部分专家来参与计算。  
每个“专家”其实是一个较小的子模型（例如一个小型的神经网络），这些专家能够针对不同的数据分布或者任务表现出更好的性能。

## 5.Dense Transformer

即标准Transformer，每个位置都会和所有其他位置进行交互，这种全连接的方式会导致计算复杂度随序列长度的平方增长。
