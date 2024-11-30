---
title: "MUSIC as a sparse decomposition method"
author: "Tom Shlomo"
date: "2024-01-30"
description: A unique introduction to the MUSIC algorithm, as a general method to solve the multisnapshot sparse decomposition problem.
format:
  html:
    toc: true
---

MUSIC (MUltiple SIgnal Classification) is a widely used algorithm for estimating the directions of arrival (DOA) of waves recorded by an array of sensors.

While it is highly effective for this specific task, MUSIC is, in fact, a more general parameter estimation method. Unfortunately, conventional introductions to MUSIC often focus on equations tailored specifically for DOA estimation. These derivations, often filled with complex exponents and trigonometric identities, can overwhelm readers and obscure the fundamental principles underlying the algorithm.

Most derivations of MUSIC make the critical assumption that the autocorrelation matrix of the signals is available. In practice, however, only an estimate of this matrix is typically accessible—often derived from a limited number of samples. Furthermore, in many real-world scenarios, such as speech signals, the signals themselves are non-stationary, making the concept of an autocorrelation matrix less well-defined. Adding to these challenges, most explanations also assume that the noise is white, a condition that is rarely satisfied in practical applications.

Despite these limitations, MUSIC often performs remarkably well, even when these assumptions are violated. This suggests that there is an alternative and more general way to derive the algorithm.

In this post, I aim to address these issues by presenting MUSIC as a general method for approximately solving the multi-snapshot sparse decomposition problem. 

### A quick introduction to sparse decompositions
You obtained an $n$-dimensional vector $y$, and you know that it is a linear combination of several "atoms" from a given set $a_1, \dots, a_m$ known as the dictionary.
The goal is to decompose $y$ to it's atoms, that is, find the atoms that participate in the linear combination.
In matrix notation we can write this as a linear equation:
$$
y = Ax
$$
where $A$ is the (known) dictionary matrix, with columns $a_1, \dots, a_m$, and $x$ contains the (unknown) coefficient for each atom. The set of non-zero indices of $x$, which we also call the support, correspond to the atoms that participate in the linear combination.
\
It might be tempting to simply solve for $x$ as both $A$ and $y$ are known, but (at least for the interesting cases) $m > n$ and the system is under determined, that is, there are infinite ways to decompose $y$ as a linear combination of atoms.

In the setting of sparse decompositions, we add an additional prior to the problem: $y$ is composed of at most $k < n$ atoms, which means $x$ is $k$-sparse (has at most $k$ non zeros).

For example, 
in DOA estimation problems, we can use $y$ to represent a signal recorded by an array of $n$-sensors, $a_i$ the response of the array to a unit amplitude wave coming from the $i$'th direction, and $x_i$ the amplitude of the wave coming from the $i$'th direction.
$k$-sparsity of $x$ is equivalent to having at most $k$ waves active simultaneously, and
decomposing $y$ into it's atoms reveals their directions.

There are 2 important extensions to the basic sparse decomposition problem.
The first is increasing robustness to noise or modeling errors, by looking for an approximate sparse decomposition instead of an exact one.
\
For example, in machine learning, approximate sparse decomposition can be used for automatic feature selection in linear regression problems.
Here $y$ contains the training data labels, $A$ contains the training data features, $x$ is the coefficient of each feature, and $k$ is the number of features to select.

The second extension is the multisnapshot (aka joint sparsity) problem, where instead of observing a single data vector $y$, we get $p$ vectors $y_1, \dots, y_p$.
In matrix notation:
$$
Y = AX
$$
where
\begin{align*}
    Y &:= \begin{bmatrix} y_1 && \cdots && y_p \end{bmatrix}
\end{align*}
is the data matrix, and 
$X_{ij}$ is the (unknown) coefficient of atom $a_i$ in $y_j$.
Here, not only
the columns of $X$ are $k$-sparse, they also share the same support.
This means that the matrix $X$ is $k$-row-sparse, that is, has up to $k$ non-zero rows.
\
In DOA estimation, the multisnapshot problem can be obtained by observing the signals at $p$ different (usually consecutive) times.
\
In the feature selection for linear regression example, the multisnapshot problem is obtained when we have multiple labels to predict, and we want to select the same $k$ feature for each.

Solving sparse decomposition problems is in general a hard problem.
It turns out that you can't do much better than enumerating over all $m \choose k$ possibilities for the support, so in practice approximation methods are often used, e.g.
Matching Pursuit, Orthogonal Matching Pursuit, Basis Pursuit, and LASSO.
Sometimes, under additional assumptions, they provide some exactness guarantees.
Although usually not presented as such, MUSIC is also an approximation method for noisy multisnapshot sparse decomposition, with some guarantees  under additional assumptions.

### Solving the noiseless multisnapshot case
We will start by describing a method that can, under several assumptions, efficiently solve the noiseless joint sparsity problem.
MUSIC can be viewed as an extension of this method for the noisy case.

Let $S$ denote the (unknown) support of $X$.
We will denote by $X_S$ the sub-matrix of $X$ obtained by keeping only the rows in $S$, and by $A_S$ the sub-matrix of $A$ obtained by keeping only the columns in $S$.
Note that with this notation, we have
$$
Y = AX = A_S X_S.
$$

MUSIC is based on two assumptions:


1. \item $\text{rank} \left(X \right) = \left| S \right|$ (or equivalently, $\text{rank} \left(X_S \right) = \left| S \right|$, as the two matrices obviously have the same row space).

2. $a_i \in \text{Range} \left(A_S \right)$ if and only if $i \in S$.

::: {.callout-note appearance="simple"}
## Reminder
Our goal is to find $S$ from $Y$.
:::


Assumption 1 implies that
$$
\text{Range}(Y) 
=
\text{Range}(A_S X_S)
=
\text{Range}(A_S),
$$
so we can get $\text{Range}(A_S)$ from $Y$.
Assumption 2 means that once we have $\text{Range}(A_s)$, we can reconstruct $S$ simply by checking which atoms are in it.
The implied algorithm is simple: given the observation $Y$ and the dictionary $A$, set $S$ as:
$$
S = \left\{ i \mid a_i \in \text{Range}(Y) \right\}.
$$
<!-- 
1. Calculate $\text{Range}(Y)$.

2. $S=\emptyset$,

3. for each $i$, if $a_i \in \text{Range}(Y)$, add $i$ to $S$. -->

Although correct and efficient, this is a terrible algorithm. Calculating the range of a matrix is numerically unstable, and even the slightest perturbation (e.g. a roundoff error) can change it drastically.
But before we continue to the more noise-robust MUSIC, let's discuss the implications of our two assumptions.

Assumption 2 means that to build an atom from a linear combination of other atoms, you need more than $\left| S \right|$ atoms.
This is related to something called the 
[spark](https://en.wikipedia.org/wiki/Spark_(mathematics))
 of $A$.
We won't get into it here, but conditions on the dictionary spark are elementary in basically every sparse decomposition method.
For certain  dictionaries, it can be shown that assumption 2 holds for any $S$ of size less than $n$.
Specifically, it holds for the dictionary in DOA estimation [^1].

[^1]: With linear, equally spaced array of sensors, if the usual anti-aliasing conditions hold: the spacing between the sensors is smaller than half the wavelength, and no 2 directions lie on the same cone who's axis contains the array.

Assumption 1 is more restrictive. It means that no row of $X_S$ is a linear combination of the other rows.
A necessary (but not sufficient) condition is $\left| S \right| \leq p$.
\
In the DOA estimation, each rows of $X_S$ contains the samples of a different source. If the sources are uncorrelated (e.g. different speakers) and $\left| S \right| \leq p$, it is very unlikely that one is a linear combination of the others.
If the sources are correlated, this doesn't hold, and MUSIC can not be applied. This happens, for example, when one source is an echo of another, due to multi-path propagation.

### MUSIC
The method above relies on the equation
$$
\text{Range}(Y) = \text{Range}(A_S)
$$ {#eq-range}
which is true if $Y=AX$, but in practice the best we can hope for is $Y=AX+W$, where the noise term $W$ is very small compared to $AX$.
Unfortunately, no matter how small $W$ is,
due to the discontinuity of $\text{Range}$,  @eq-range won't even hold approximately. 
In fact, if $p \geq n$, we will almost surely have $\text{Range}(Y) = \mathbb{R}^n$, 
and the algorithm above would just yield $S=\left\{1, \dots,  m \right\}$.

MUSIC makes 2 modifications the the algorithm above.
\
First, we replace $Y$ with $\tilde{Y}$, a rank-$\left| S \right|$ approximation of $Y$.

::: {.callout-note appearance="simple"}
## Note
$\left| S \right|$ is assumed known is MUSIC. It can be avoided, sometimes, using model selection methods.
:::

Since $AX$ has rank $\left| S \right|$,
taking a rank $\left| S \right|$ approximation of $Y$ has a denoising effect[^2]. 
Indeed, unlike $\text{Range} \left( Y \right)$, $\text{Range} \left( \tilde{Y} \right)$
is a good estimate for 
$\text{Range} \left( A_S \right)$ when $W$ is small,
but it is not exact:
almost surely, none of the atoms would lie exactly in it.
So the second modification soften the requirement that $a_i \in \text{Range} \left( \tilde{Y} \right)$ to add $i$ to $S$.
Instead, we will require that $a_i$ is "almost in" $\text{Range} \left( \tilde{Y} \right)$, by checking if it looses little magnitude when projected onto it:
$$
c_i := \frac{\| \text{Proj}_{\text{Range} \left( \tilde{Y} \right)}(a_i) \|^2}
{\| a_i \|^2 }
\text{ is close to 1}
\implies
\text{ add $i$ to $S$}
$$
(what "is close" means exactly differs between implementations. 
When the atoms can be ordered, like in DOA estimation, it is common to use a peak selection algorithm).

[^2]: $\text{Range} \left( A_S \right)$ is sometimes called the signal subspace, and the subspace orthogonal to it the noise subspace.

As we said above, $\tilde{Y}$ is a rank-$\left| S \right|$ approximation to $Y$.
In MUSIC, we use the best rank-$\left| S \right|$ approximation in the least squares sense,
which is given by the truncated singular value decomposition (SVD) of $Y$.
Note that we don't really need to calculate $\tilde{Y}$ itself.
Since the first $\left| S \right|$ left singular vectors of $Y$ form an orthonormal basis for $\text{Range} \left( \tilde{Y} \right)$, we have:
$$
\label{music_final}
c_i = \frac{\| U^H a_i\|^2}{\| a_i\|^2}.
$$
where the columns of $U$ are the first $\left| S \right|$ left singular vectors.

To wrap things up, a few notes to connect the above to the "usual" MUSIC derivation:

* The left singular vectors of $Y$ are the eigenvectors of $p^{-1} YY^T$, which, in a stochastic setting, can be viewed as an estimate of the autocorrelation matrix.

* The usual MUSIC formula use the last $n-\left| S \right|$ left singular vectors (which we stack to the columns of the matrix $\bar{U}$) instead of the first $\left| S \right|$.
From the Pythagorean theorem
$$
\| a_i \| ^2 = \|U^H a_i \|^2 + \| \bar{U}^H a_i \|^2,
$$
so we can write $c_i$ as follows:
$$
c_i = 1 -\frac{ 
\| \bar{U}^H a_i \|^2 
}{\| a_i\|^2}.
$$

* In MUSIC for DOA/spectral estimation,
it is common to plot $\frac{1}{1-c_i}$, and call it the "pseudo-spectrum".
The 1-over-1-minus transform maps numbers close to 1 to very large numbers, which often results in very beautiful and pointy (but somewhat misleading) plots.