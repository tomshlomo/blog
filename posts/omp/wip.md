---
title: "The greedy algorithms for sparse approximation no one talks about"
author: "Tom Shlomo"
date: "2024-01-30"
description: todo
format:
  html:
    toc: true
---

In the problem of sparse approximation, we are trying to approximate a given vector $y$ as a linear combination of few columns of a given matrix $A$.
It is useful in many applications, such as machine learning (e.g. feature selection), image processing (e.g. image denoising, inpainting, deblurring, compression), and signal processing (e.g. compressive sensing).

In this post, we will discuss greedy algorithms for solving the sparse approximation problem, focusing on the well-known Matching Pursuit (MP) and Orthogonal Matching Pursuit (OMP) algorithms, as well as a "new" algorithm that I call the Most Obvious Matching Pursuit (MOMP).

## Notation
In this post we will consider the following flavor of the sparse approximation problem:
\begin{equation}
    \begin{array}{ll}
        \underset{x}{\mbox{minimize}}  & \| Ax - y\|^2 \\
        \mbox{subject to} & \text{$x$ has at most $k$ non-zero entries}.
    \end{array}
    \label{e-opt-prob}
\end{equation}
Where

* $y$ is the $m$-vector we are trying to approximate,

* $A$ is the $m \times n$ dictionary matrix with columns $a_1, \ldots, a_n$,

* $x$ is the $n$-vector of coefficients,

When $S$ is an ordered subset of $1, \ldots, n$,
we denote by $A_S$ the matrix obtained by keeping only the columns of the matrix $A$ that are in $S$, and by $X_S$ the vector obtained by keeping only the elements of $x$ that are in $S$.

When $M$ is a matrix, $\text{Range}(M)$ denotes it's column space, and $M^H$ denotes it's conjugate transpose (if you don't care about complex data, you can think of it as the transpose).

<!-- Similarly, $I_S$ denotes the $n \times |S|$ matrix obtained by keeping only the columns of the identity matrix that are in $S$.
$I_S$ is a useful matrix, as it extracts the elements of a vector that are in $S$:
\begin{align}
x_S &= I_S x \\
A_S &= A I_S \\
\end{align}

Similarly, $I_S^H$ projects an $|S|$-vector to the space of vectors with support $S$, so
$\text{supp}(x) &= S$
if and only if
$x = I_S^H I_S x$. -->



<!-- Matching Pursuit (MP) is a popular and simple for this problem, which iteratively selects the column of $A$ that is most correlated with the current residual.
Orthogonal Matching Pursuit (OMP) is a
One of the most popular algorithms is Orthogonal Matching Pursuit (OMP), which is a greedy algorithm that iteratively selects the column of $A$ that is most correlated with the current residual. -->


## Support Recovery
If the support of $x$ is known and denoted by $S$, then since $A x = A_S x_S$, finding the optimal $x_S$ is a simple least squares problem:
$$
\underset{x_S}{\text{minimize}} \left\{ \| A_S x_S - y \|^2 \right\}.
$$
This means that in practice, the problem is to recover the support of $x$.
The following results will be useful in this context:

* An optimal support $S$ maximizes $y^H P_{A_S} y$,
where $P_{A_S}$ is the projection matrix onto $\text{Range}(A_S)$.
<!-- (The intution here is that with an optimal support, $y$ should be approximately in $\text{Range}(A_s)$, so projecting onto it should have little effect). -->

* An optimal $x_S$ is given by any solution to $A_S^H A_S x_S = A_S^H y$ (aka the normal equations).

::: {.callout-note icon=false collapse=true}
#### Proof [^1]
[^1]: While these are standard results for least squares problems,
I find that they are not always presented in a clear way.
I like proving such results without using gradients
(which obscure the intuition, in my opinion)
and without inverting matrices (which adds unnecessary caveats about rank and conditioning).

First, we rewrite $y$ as $y^{||} + y^{\perp}$, where $y^{||}$ is in $\text{Range}(A_S)$ and $y^{\perp}$ is orthogonal to it.
Since $A_S x_S$ is in $\text{Range}(A_S)$, we can apply the Pythagorean theorem (twice) to get:
\begin{align}
 \| A_S x_S - y \|^2 &=
\| \left(A_S x_S - y^{||}\right) - y^{\perp} \|^2
 \\&=
  \| A_S x_S - y^{||} \|^2 + \| y^{\perp} \|^2 
  \\&=
  \| A_S x_S - y^{||} \|^2 + \| y \|^2 - \| y^{||} \|^2 .
\end{align}
Minimizing across $x_S$, the first term vanishes (since $y^{||}$ is in $\text{Range}(A_S)$, there is by definition an $x_S$ such that $y^{||} = A_S x_S$).
\
Thus, an optimal $S$ maximizes $\| y^{||} \|^2$.
\
Since $P_{A_S}$ is a projection matrix, it is symmetric and idempotent, so
$$
\| y^{||} \|^2 = \| P_{A_S} y \|^2 = y^H P_{A_S}^H P_{A_S} y = y^H P_{A_S} y
$$
which proves the first item.

An optimal $x_S$ satisfies $y^{||} = A_S x_S$.
\
As the projection of $y$ onto $\text{Range}(A_S)$, $y^{||}$ is characterized by $A_S ^H \left(y - y^{||} \right) = 0$.
\
Combining the two, we get the second item.
:::

## Greedy Algorithms
Exactly Solving the sparse approximation problem is NP-hard. It turns out there isn't a significantly better way than brute force checking all $n \choose k$ possible supports.
Thus we often turn to greedy algorithms[^convex], which are faster but not guaranteed to find the optimal solution.

[^convex]: Convex relaxation (e.g. Basis Pursuit, LASSO) is another approach, but it is not the focus of this post.

# Matching Pursuit (MP)
MP is a simple and popular algorithm for the sparse approximation problem.
The algorithm is based on the following observation:
When $k=1$, the problem is equivalent to finding the column of $A$ that is most correlated with $y$:
\begin{align}
y^H P_{a_i} y &= y^H a_i (a_i^H a_i)^{-1} a_i^H y 
\\&=
\frac{| a_i^H y |^2}{\| a_i \|^2}.
\end{align}


MP is a greedy iterative algorithm.
We start with an empty support. At each iteration:
1. Greedily add the optimal column to the support.
2. Project $y$ onto the space orthogonal to the column 
(Or, equivalently, subtracts $a_i x_i$ from $y$, where $a_i$ is the selected column and $x_i$ is the coefficient that minimizes $\| a_i x_i - y \|^2$).

<!-- Note tht this is equivalent to subtracting the  of the selected column to $y$. -->
 <!-- and then project $y$ onto the the space orthogonal to that column.

In python, the algorithm can be implemented as follows:
```python
def mp(A: np.ndarray, y: np.ndarray, K: int) -> set[int]:
    S = set()
    A = A / np.linalg.norm(A, axis=0) // normalize columns
    while len(s) < k:
        i = np.argmax(np.abs(A.H @ y))
        S.add(i)
        x_i = A[:, i].H @ y
        y -= A[:, i] * x_i
``` -->

## Orthogonal Matching Pursuit (OMP)
OMP is a popular variant of MP.
While in MP we project $y$ onto the space orthogonal to the selected column in the current iteration, in OMP we project $y$ onto the space orthogonal to all the selected columns so far.

This is equivalent to subtracting from $y$ $A_S x_S$, where $S$ is the current support and $x_S$ minimizes $\| A_S x_S - y \|^2$.

## The Most Obvious Matching Pursuit (MOMP)
Start with $S=\emptyset$. At each iteration add the column $i$ that maximizes $y^H P_{A_{S \cup \left\{ i \right\}}} y$.
That's it, there are no $y$ updates here.

OMP tries to improve MP by optimizing the coefficients of all the selected columns at each iteration.
However, this coefficient optimization happens only after the new column is added. During the selection of the new column, like MP, OMP still selects the optimal column assuming $k=1$, affectively optimizing only the coefficient of the selected column.

In MOMP, we fix that, by simultaneously optimizing both the new column and the coefficients of all the selected columns. That is, we are affectively solving 
$$
\underset{S, x_S}{\text{minimize}} 
 \|
    A_S x_S - y 
\|^2 
$$
at each iteration, with the constraint that $S$ is the same as in the previous iteration, except for one additional column.

The algorithms MP and OMP are well known and widely used.
However, I have not seen the algorithm MOMP in the literature (I would be happy to be proven wrong), even though I really think it is the most obvious algorithm to try.

A possible reason for this is is compuation complexity: at each iteration, MOMP requires solving a least squares problem for each candidate column, as opposed to OMP which requires solving it only for the new column.
\
There is however a way to utilize the solution of the previous iteration to speed up the computation (by using incremental Cholesky/QR factorizations). I plan to discuss this in a future post.



<!-- In OMP, we select the best column assuming $k=1$, but then select the coefficients optimally for the whole support.
This algorithm can be seen as an improvemnt of OMP, since simulatenously optimiz
Both MP and OMP update $y$ at each iteration.


 that improves the solution by optimizing the coefficients of all the selected columns before subtracting their contribution from $y$.
OMP improves MP by optimizing the coefficients of all the selected columns before subtracting their contribution from $y$.
This subtraction is equivalent to projecting $y$ onto 


MP can be seen as a greedy algorithm that iteratively adds to the support the column of $A$ tha
Once the optimal column is found, can substract the subtract from $y$ its projection onto the column. 
the projection of $y$ onto the column space of the selected column.

\begin{align}
{\text{\argmin}} \left\{ \| A x - y \|^2 \right\}

\underset{|S| &= 1}{\text{\argmin}} \left\{ \| A x - y \|^2 \right\}
\\&=
\underset{|S| &= 1}{\text{\argmin}} \left\{ \| a_i x - y \|^2 \right\}
\end{align} -->
<!-- 
# Matching Pursuit (MP)
MP is a simple and popular algorithm for the sparse approximation problem.
The algorithm is based on the following observation:
When $k=1$, the problem is equivalent to finding the column of $A$ that is most correlated with $y$.
\begin{align}
\text{\min} \left\{ \| A x - y \|^2 \,|\, \text{supp}\left(x \right) \leq 1 \right\}
&=
\text{\min} \left\{ \| a_i x - y \|^2 \,|\, \text{supp}\left(x \right) \leq 1 \right\}
\end{align} -->



<!-- 
1/2 x'Qx + c'x + d

Qx + c = 0
x = - inv(Q) c

1/2 c' inv(Q) Q inv(Q) c - c' inv(Q) c + d

- 1/2 c' inv(Q) c + d


1/2 | Ax - y | ^2 = 1/2 (Ax - y)' (Ax - y) = 1/2 x' A' A x - y' A x + y' y / 2
min = -1/2 (- y' A) inv(A' A) (-A' y) + y' y / 2
    = 1/2 y' A inv(A' A) A' y + y' y / 2 -->
