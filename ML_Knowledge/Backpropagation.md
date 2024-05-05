## Backpropagation

Resources:
- Murphy - Probabilistic Machine Learning: An introduction; Chapter 13.3 (https://probml.github.io/pml-book/book1.html)

The **backpropagation** algorithm is used to compute the gradient of the loss function w.r.t. the parameters of the neural network in each layer.
This gradient is then typically passed to a gradient-based optimization algorithm.

Let $f = f_k\circ \ldots f_1\colon \mathbb{R}^n\to \mathbb{R}^m$ be a function. Via the chain-rule, it's derivative, or Jacobian, $J_{f,x}\in\mathbb{R}^{m\times n}$ can be expressed as

$`\begin{aligned}
J_{f,x} = J_{f_k,x_k} \circ \ldots J_{f_1,x_1}\in\mathbb{R}^{m\times n},
\end{aligned}`$ 
where $x_1\coloneq x $, $x_i \coloneq f_{i-1}(x_{i-1})$ and $J_{f_i,x_i}\in\mathbb{R}^{m_i\times m_{i-1}$ for $ i = 2, \ldots, k$.
There are two options to compute $J_{f,x}$.
One can either compute $J_{f,x}$ *column-wise* in a *forward* manner, i.e. from the right, by calculating

$`\begin{aligned}
J_{f,x}e_i = J_{f_k,x_k} \circ \ldots J_{f_1,x_1}\in\mathbb{R}^{m\times n}e_i,
\end{aligned}`$ 
for $i = 1,\ldots,m$ and the standard basis $e_1,\ldots,e_m$ of $\mathbb{R}^n$.
The second possibility is to compute $J_{f,x}$ *row-wise* in a *backward* manner, i.e. from the left, by calculating

$`\begin{aligned}
u^T_i J_{f,x} = u^T_i J_{f_k,x_k} \circ \ldots J_{f_1,x_1}\in\mathbb{R}^{m\times n}e^n_i,
\end{aligned}`$ 
for $i = 1,\ldots,n$ and the standard basis $u_1,\ldots,u_n$ of $\mathbb{R}^m$.

In case n > m, particularly if $f$ is scalar-valued, the backward calculation is *more efficient*.
E.g. assuming $m=1, k=3, n=m_1=m_2=m_3$, the cost of forward computation is $\mathcal{O}(n^3)$ and the cost of backward computation is $\mathcal{O}(n^2)$.

Since the loss function in deep-learning is typically scalar-valued, *back-propagation* is used to compute its gradient with respect to the weights (i.e. parameters) of the network, for a given input.
Note, of course the input itself has to fist be forward-propagated through the network (without differentiation) to start with.

Consider now $F = f_k\circ\ldots\circ f_1:\mathbb{R}^n \to \mathbb{R}$ to be a neural-network with loss function $L:\mathbb{R}^n\to\mathbb{R}$ and let
$\mathcal{L}\coloneq L\circ F$.
In contrast to the previous case, $F$ depends on learnable parameters $\theta$, as well as on the input $x$, i.e. $x_{i+1}=f_i(x_i, \theta_i)$.
Therefore, we obtain the expression
$`\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \theta_i} = \frac{\partial \mathcal{L}}{\partial x_k}\frac{\partial \mathcal{f_k(x_k,\theta_k)}}{\partial \theta_k}\circ\ldots\circ\frac{\partial \mathcal{f_1(x_1,\theta_1)}}{\partial \theta_1}
\end{aligned}`,$
which on can use to calculate the gradient of $\mathcal{L}$ w.r.t. the parameters via back-propagation.
First one has to do a forward-pass, i.e. for $i=1,\ldots,k$ we calculate $x_{i+1} \coloneq f_i(x_i, \theta_i)$.
For the back-propagation we set $u_{k+1}\coloneq 1$ and do the compute the following for $i=k,\ldots,1$:

$`\begin{aligned}
g_i & \coloneq u^T_{i+1}\frac{\partial f_i(x_i, \theta_i)}{\partial \theta_i}\\
u^T_i & \coloneq u^T_{i+1}\frac{\partial f_i(x_i, \theta_i)}{\partial x_i} = \frac{\partial x_k}{\partial x_k}\circ\cdots\circ \frac{\partial x_i}{\partial x_i}
\end{aligned}`$

This is the back-propagation algorithm for a simple multi-layer perceptron!
This concept can be extended to more general situations, where, instead of a simple feed-forward network, the underlying network has the structure of a *directed acyclic graph* (for more details see chapter 13.4 of the referenced book by Myrphy).
