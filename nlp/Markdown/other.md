# Others

## 1.GQA(Grouped query attention)

 在推理阶段，生成第 ``i+1`` 个 token 时，需要前``i``个token的 $K(W_k\times keys)$ 和 $V(W_v\times values)$，以及第 ``i`` 个 token 的 $Q(W_q\times queries)$。  
 为了避免每次生成新 token 时重复计算 $K$ 和 $V$ ，将每个 token 的 $K$ 和 $V$ 保存下来，即 KVCache。

但是存储每个 token 的 $K、V$ 矩阵对内存占用较高，对于 `num_heads=8` 的多头注意力机制，可以将每两个 head 分为一组，共享相同的 $W_k$ 和 $W_v$ 矩阵，而每组内的 $W_q$ 矩阵保持独立。  
这样每组中的head生成相同的 $K$ 和 $V$，但具有不同的查询向量 $Q$。减少内存占用的同时（**也许**）也能减少矩阵乘法的次数 ($W_k\times keys、W_v\times values、Q\times K^T$).

## 2.RoPE

## 3.SwiGLU
