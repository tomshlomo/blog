---
title: "Bilinear Muon"
author: "Tom Shlomo"
date: "2026-06-14"
format:
  html:
    toc: true
---

## 1. The Flatland Trap

For years, the deep learning community has relied on optimizers like Adam and AdamW. While effective, these methods suffer from a fundamental geometric blind spot: they view weight matrices as flat, one-dimensional arrays of parameters. By maintaining per-parameter moments, Adam scales each weight independently, completely ignoring the structural reality that these numbers are spatially arranged into matrices. It optimizes in "Flatland," blind to the fact that these matrices operate on high-dimensional vector spaces.

## 2. The Muon Leap

The recently introduced Muon optimizer breaks free from Flatland by treating weight matrices as what they actually are: linear operators. Instead of scaling parameters individually, Muon operates on the spectrum of the matrix.

Specifically, Muon computes the steepest descent step bounded by the spectral norm ($\|\cdot\|_2$). Let $\mathcal{L}$ be the loss function and $G = \nabla_W \mathcal{L}$ be the gradient with respect to a linear operator $W$. Muon solves:


$$\min_{\Delta W} \langle G, \Delta W \rangle \quad \text{s.t.} \quad \|\Delta W\|_2 \le \epsilon$$


The analytical solution to this problem is $\Delta W = -\epsilon \cdot \text{Orth}(G)$, where $\text{Orth}(G)$ is the orthogonalized gradient—an operation that flattens all non-zero singular values of $G$ to exactly $1$. By applying orthogonalization, Muon forces the update to be well-conditioned, making the learning process invariant to the dominant singular values of the gradient. This proves that respecting the metric space of linear operators yields faster, more stable training.

## 3. The Bilinear Reality

However, we can take the Muon idea one step further when applying it to Transformer attention layers. In the pre-softmax attention mechanism, the query matrix ($W_Q$) and key matrix ($W_K$) do not evaluate data independently.

Let $x_i$ and $x_j$ be the token embeddings (vectors in $\mathbb{R}^d$) for tokens $i$ and $j$. The attention mechanism evaluates their interaction not as isolated linear projections, but as a bilinear transformation. A bilinear transformation is a function $B: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ that is linear in both arguments. Every such bilinear transformation can be uniquely associated with a matrix $M \in \mathbb{R}^{d \times d}$ such that:


$$B(x_i, x_j) = x_i M x_j^T$$

In the case of self-attention, the associated matrix is $W_{QK} = W_Q W_K^T$.

The critical realization is that **we are actually searching over the space of bilinear transformations**, not independent linear operators. This space has its own norms and structures that are fundamentally different from the ones we would get if we looked at each matrix individually.

For instance, the supremum norm of a bilinear transformation measures its maximum possible output for bounded inputs:


$$\|B\|_\infty = \sup_{\substack{\|x\|_2 \le 1 \\ \|y\|_2 \le 1}} |B(x, y)|$$


When we translate this norm back to its associated matrix $M$, it corresponds exactly to the spectral norm of the composite matrix ($\|M\|_2$).

Similarly, we can define the Euclidean energy of the bilinear form natively as the expected squared output under isotropic random inputs. If $x$ and $y$ are drawn from distributions with zero mean and identity covariance ($\mathbb{E}[x^T x] = I$, $\mathbb{E}[y^T y] = I$), the expected energy is:


$$\mathbb{E}\left[|B(x, y)|^2\right]$$


When mapped back to the associated matrix $M$, this geometric expectation corresponds exactly to the squared Frobenius norm of the matrix ($\|M\|_F^2$).

By treating $W_Q$ and $W_K$ as completely independent linear operators, current optimizers are mathematically blind to the true norm and geometric structure of the bilinear form that actually dictates the forward pass. To optimize attention correctly, we must optimize natively in the space of bilinear transformations.

## 4. The Low-Rank Constraint

In practice, we do not search over the entire ambient space of bilinear forms. Neural networks use bilinear forms of the structure $x_i W_Q W_K^T x_j^T$, which correspond to a strictly low-rank subset. Specifically, the intermediate projection space has a dimension $d_k$ (the attention head dimension), which is typically much smaller than the hidden dimension $d$ (e.g., $d_k = 128$ while $d = 4096$). This limits the rank of the associated matrix to at most $d_k$.

This architectural constraint is highly deliberate and provides three major benefits:

1. **Computation:** It allows for significantly faster evaluation of the bilinear form in both the forward and backward passes due to matrix associativity (computing $(x_i W_Q) (x_j W_K)^T$ is vastly cheaper than multiplying by a full $d \times d$ matrix).
2. **Memory:** It requires storing only $2 d \cdot d_k$ parameters instead of the $d^2$ parameters required for a full-rank bilinear form.
3. **Generalization:** By drastically reducing the parameter count, it restricts the hypothesis space, acting as an implicit regularization that promotes smoother, more generalizable representations.

To optimize within this subset, our goal is to find an update $\Delta W_{QK}$ that maximizes descent along the composite gradient $G_{QK}$ (where $G_{QK} = \nabla_{W_{QK}} \mathcal{L}$), subject to both a continuous norm bound and the strict structural constraint of the architecture.

The steepest descent problem in its pure form is therefore:


$$\min_{\Delta W_{QK}} \langle G_{QK}, \Delta W_{QK} \rangle$$

$$\text{s.t.} \quad \|\Delta W_{QK}\| \le \epsilon$$

$$\text{rank}(W_{QK} + \Delta W_{QK}) \le d_k$$

How do we actually enforce this rank constraint? By the mathematical definition of matrix rank, any $d \times d$ matrix has a rank of at most $d_k$ if and only if it can be perfectly factored into the product of two $d \times d_k$ matrices. Let us call these new ideal factors $A$ and $B$:


$$W_{QK} + \Delta W_{QK} = A B^T$$

Since our current weights $W_Q$ and $W_K$ are fixed, we can reparameterize our search. Instead of searching for the ambient step $\Delta W_{QK}$, we search for the difference between these new ideal factors and our current weights:


$$\Delta W_Q = A - W_Q$$

$$\Delta W_K = B - W_K$$

Substituting these back into our factorization gives:


$$W_{QK} + \Delta W_{QK} = (W_Q + \Delta W_Q)(W_K + \Delta W_K)^T$$

Expanding the right side yields:


$$W_{QK} + \Delta W_{QK} = W_Q W_K^T + \Delta W_Q W_K^T + W_Q \Delta W_K^T + \Delta W_Q \Delta W_K^T$$

Since we defined our current state as $W_{QK} = W_Q W_K^T$, we can subtract it from both sides to isolate the exact step taken in the ambient space of bilinear forms:


$$\Delta W_{QK} = \Delta W_Q W_K^T + W_Q \Delta W_K^T + \Delta W_Q \Delta W_K^T$$

This is a profound geometric realization. Searching over the abstract, non-convex rank constraint in the ambient space is **mathematically identical** to searching over the unconstrained factors $\Delta W_Q$ and $\Delta W_K$ and applying this exact polynomial expansion. There is no rank-$d_k$ matrix reachable in the ambient space that cannot be reached natively through the factors. The architectural constraint forces the algebraic form.

## 5. The Tangent Approximation

By moving to the factorized view, we have translated an abstract rank constraint into a concrete algebraic equation. By construction, any update applied via the factors trivially satisfies the rank $d_k$ constraint.

However, substituting this exact $\Delta W_{QK}$ into our steepest descent objective introduces a significant mathematical hurdle: the quadratic term $\Delta W_Q \Delta W_K^T$. Because this bilinear coupling transforms the norm constraint into a non-convex set, the quadratic term will prove exceptionally difficult to handle analytically in the coming sections. There are two approaches we can take here to move forward.

The first approach is to keep the quadratic term and employ control-theoretic techniques, such as Linear Matrix Inequality (LMI) or Semidefinite Programming (SDP) relaxations, to bound the non-convex objective. While mathematically intriguing, these iterative solvers are not well-suited for the massive parallelism of modern GPUs, so we will not pursue them here.

The second approach is to ignore the quadratic term. Assuming our step size bound $\epsilon$ is sufficiently small, the quadratic term $\Delta W_Q \Delta W_K^T$ (which scales with $\epsilon^2$) is negligible compared to the linear terms. Dropping it yields the approximation:


$$\Delta W_{QK} \approx \Delta W_Q W_K^T + W_Q \Delta W_K^T$$

Geometrically, this can be interpreted as constraining our step to lie strictly on the tangent space of the low-rank manifold at our current weights. As an aside, if you are interested in how this exact geometric perspective is being used to rethink neural network architectures, this connects nicely to some great recent works on modular manifolds from [Thinking Machines](https://thinkingmachines.ai/blog/modular-manifolds/) and [Modula Systems](https://www.google.com/search?q=https://docs.modula.systems//).

This first-order tangent approximation captures the optimal descent direction while converting the intractable non-convex constraint into a strictly linear operator, unlocking exact, hardware-friendly analytical solutions.

### A Beautiful Consequence

Using this tangent approximation has a powerful implication for our objective function. If we substitute our linearized step into the descent objective $\langle G_{QK}, \Delta W_{QK} \rangle$, we get:


$$\langle G_{QK}, \Delta W_Q W_K^T + W_Q \Delta W_K^T \rangle = \langle G_{QK}, \Delta W_Q W_K^T \rangle + \langle G_{QK}, W_Q \Delta W_K^T \rangle$$

Using the cyclic properties of the Frobenius inner product (trace), we can isolate the parameter updates:


$$\langle G_{QK}, \Delta W_Q W_K^T \rangle = \text{tr}(G_{QK}^T \Delta W_Q W_K^T) = \text{tr}(W_K^T G_{QK}^T \Delta W_Q) = \text{tr}((G_{QK} W_K)^T \Delta W_Q) = \langle G_{QK} W_K, \Delta W_Q \rangle$$

Similarly, for the second term:


$$\langle G_{QK}, W_Q \Delta W_K^T \rangle = \text{tr}(G_{QK}^T W_Q \Delta W_K^T) = \text{tr}((G_{QK}^T W_Q)^T \Delta W_K) = \langle G_{QK}^T W_Q, \Delta W_K \rangle$$

By the chain rule, $G_Q = \nabla_{W_Q} \mathcal{L} = G_{QK} W_K$ and $G_K = \nabla_{W_K} \mathcal{L} = G_{QK}^T W_Q$. Substituting these gives us the simplified objective:

$$ \langle G_Q, \Delta W_Q \rangle + \langle G_K, \Delta W_K \rangle$$

This is beautiful—we don't need to ever form the large $d \times d$ gradient $G_{QK}$. The downstream gradients of $W_Q$ and $W_K$ (which we would effortlessly get from a naive PyTorch implementation) are sufficient to optimize natively in the space of bilinear transformations.

## 6. The $L^2$ Crown Jewel: Analytical Frobenius Optimization

Now let's commit to a norm and derive the update. We will begin with the $L^2$ norm. We derive the update using the Lagrangian form:

$$\max_{\Delta W_Q, \Delta W_K} \quad \langle G_Q, \Delta W_Q \rangle + \langle G_K, \Delta W_K \rangle - \frac{1}{2\eta} \| \Delta W_Q W_K^T + W_Q \Delta W_K^T \|_F^2$$

Taking partial derivatives with respect to $\Delta W_Q$ and $\Delta W_K$ and setting them to zero yields the following coupled optimality conditions:

1. $\Delta W_Q (W_K^T W_K) + W_Q (\Delta W_K^T W_K) = \eta G_Q$
2. $\Delta W_K (W_Q^T W_Q) + W_K (\Delta W_Q^T W_Q) = \eta G_K$

This system consists of $2 d \cdot d_k$ unknowns defined by $2 d \cdot d_k$ linear equations. At first glance, this is a well-determined square system. However, the system is fundamentally underdetermined: it possesses a non-trivial null space. For any arbitrary $d_k \times d_k$ matrix $S$, consider the update:


$$\Delta W_Q = W_Q S, \quad \Delta W_K = -W_K S^T$$

If we substitute these into the homogeneous part of the first optimality condition, we get:


$$(W_Q S)(W_K^T W_K) + W_Q ((-W_K S^T)^T W_K) = W_Q S (W_K^T W_K) + W_Q (S W_K^T W_K) = W_Q S W_K^T W_K - W_Q S W_K^T W_K = 0$$

The same substitution holds for the second condition. This proves that there is an entire affine subspace of optimal solutions.

### Exploiting the Redundancy

At first glance this null space looks like a defect: there is no unique pair $(\Delta W_Q, \Delta W_K)$ for the optimizer to return. But look at what the gauge mode actually does to the bilinear step. Substituting $\Delta W_Q = W_Q S$ and $\Delta W_K = -W_K S^T$ into the tangent update gives:

$$\Delta W_Q W_K^T + W_Q \Delta W_K^T = W_Q S W_K^T - W_Q S W_K^T = 0$$

Every direction in the null space leaves the actual update to $W_{QK}$ completely unchanged. The freedom is not in *what* step we take in the space of bilinear forms—it is only in *how we split* that single step between the two factors. Since the loss (to first order) only ever sees the product $W_{QK}$, we should stop trying to solve for $\Delta W_Q$ and $\Delta W_K$ directly, and instead solve for the one quantity that is actually pinned down: the tangent step itself,

$$T := \Delta W_Q W_K^T + W_Q \Delta W_K^T$$

This is how we turn the singularity to our advantage. The two optimality conditions are coupled and $2 d \cdot d_k$-dimensional only because they are written in the redundant factors. Rewritten in terms of $T$, they collapse into something trivial. Since $\Delta W_Q (W_K^T W_K) + W_Q (\Delta W_K^T W_K) = T W_K$ (and symmetrically for the second), the conditions are simply:

$$T W_K = \eta\, G_Q, \qquad T^T W_Q = \eta\, G_K$$

We can now read off $T$ in closed form, with no linear solve at all. Because $T$ lies in the tangent space, it satisfies the projection identity $T = P_Q T + T P_K - P_Q T P_K$, where

$$P_Q = W_Q (W_Q^T W_Q)^{-1} W_Q^T, \qquad P_K = W_K (W_K^T W_K)^{-1} W_K^T$$

are the orthogonal projectors onto the column spaces of the two factors. The conditions above supply each piece directly—$W_Q^T T = \eta\, G_K^T$ from the second and $T W_K = \eta\, G_Q$ from the first—yielding:

$$T = \eta\left[\, W_Q (W_Q^T W_Q)^{-1} G_K^T + G_Q (W_K^T W_K)^{-1} W_K^T - W_Q (W_Q^T W_Q)^{-1}(W_Q^T G_Q)(W_K^T W_K)^{-1} W_K^T \,\right]$$

Every inverse here is a tiny $d_k \times d_k$ matrix, and we never form the $d \times d$ gradient $G_{QK}$—only the cheap factor gradients $G_Q$ and $G_K$ appear.

All that remains is to hand the optimizer *some* factorization of $T$. This last choice is precisely the leftover gauge freedom, and by the argument above it does not affect the update to $W_{QK}$. A simple, immediate split is:

$$\Delta W_Q = \eta\, G_Q (W_K^T W_K)^{-1}$$

$$\Delta W_K = \eta\,(I - P_K)\, G_K (W_Q^T W_Q)^{-1}$$

(If the conditioning of the factors matters across many steps, one can instead solve for the minimum-norm, balanced split—but the resulting step in the space of bilinear forms is identical.)

This gives the exact, closed-form steepest-descent step on the tangent space. By recognizing that the under-determinacy is pure factorization redundancy rather than a genuine ambiguity, we reduced a massive, ill-posed coupled system to a handful of $d_k \times d_k$ operations—no Sylvester equation, no $d \times d$ inversion, nothing to iterate.
