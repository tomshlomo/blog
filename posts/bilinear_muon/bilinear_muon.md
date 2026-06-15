---
title: "Bilinear Muon"
description: "Adam, and even Muon, optimize attention's query and key matrices as if they were independent. Treating them as the single bilinear form they jointly define yields a family of Muon-style update rules."
author: "Tom Shlomo"
date: "2026-06-14"
format:
  html:
    toc: true
---

## Introduction

Optimizers have been getting steadily more geometry-aware. Adam treats every weight as an independent scalar. Muon takes a step further by treating a weight matrix as a linear operator and taking its steepest-descent step in the spectral norm. This post pushes one step further still, into the geometry that attention actually computes in.

The key observation is that the query and key matrices $W_Q$ and $W_K$ are never used on their own. What an attention score sees is the bilinear form $x_i\, W_Q W_K^T x_j^T$, governed by the single matrix $W_{QK} = W_Q W_K^T$. So instead of optimizing $W_Q$ and $W_K$ as two independent operators, we should run Muon on the bilinear form they jointly define. That one shift in perspective is the heart of the post.

Carried out properly, it leads not to a single optimizer but to a family of them. Working in the space of low-rank bilinear forms, I derive the steepest-descent update under two different norms. The Frobenius ($L^2$) norm gives a clean closed form I call Bilinear SGD, in which each factor's gradient is preconditioned by the geometry of the other factor. The spectral ($L^\infty$) norm, the one the bilinear picture really calls for, has no exact closed form, so I trade exactness for GPU-friendliness through a sequence of progressively cheaper approximations. These produce two practical Muon variants, preconditioned Muon and cross-scaled Muon, each in a winner-take-all and a symmetric flavor. A summary table at the end lays them all out by the design choices that separate them: the space they descend in, the norm that bounds the step, and the approximations that buy a closed form. Much of this overlaps with Tilde Research's [Compositional Muon](https://blog.tilderesearch.com/blog/compositional-muon), published a few days earlier; the [Related Work](#related-work) section spells out what coincides and what is new.

A caveat up front. This is a theory-first post. I derive these update rules but have not yet benchmarked them, so I make no claim about which one wins in practice. The aim is to show that the bilinear view is the natural one for attention, and to map out the design space it opens up.

## 1. Adam Treats Weight Matrices as Flat Arrays

For years, the deep learning community has relied on optimizers like Adam and AdamW. While effective, these methods suffer from a fundamental geometric blind spot: they view weight matrices as flat, one-dimensional arrays of parameters. By maintaining per-parameter moments, Adam scales each weight independently, completely ignoring the structural reality that these numbers are spatially arranged into matrices. It optimizes in "Flatland," blind to the fact that these matrices operate on high-dimensional vector spaces.

## 2. Muon: Steepest Descent Under the Spectral Norm

The recently introduced Muon optimizer breaks free from Flatland by treating weight matrices as what they actually are: linear operators. Instead of scaling parameters individually, Muon operates on the spectrum of the matrix.

Specifically, Muon computes the steepest descent step bounded by the spectral norm ($\|\cdot\|_2$). Let $\mathcal{L}$ be the loss function and $G = \nabla_W \mathcal{L}$ be the gradient with respect to a linear operator $W$. Muon solves:


$$\min_{\Delta W} \langle G, \Delta W \rangle \quad \text{s.t.} \quad \|\Delta W\|_2 \le \epsilon$$ {#eq-muon}


The analytical solution to this problem is $\Delta W = -\epsilon \cdot \text{Orth}(G)$, where $\text{Orth}(G)$ is the orthogonalized gradient, an operation that flattens all non-zero singular values of $G$ to exactly $1$. By applying orthogonalization, Muon forces the update to be well-conditioned, making the learning process invariant to the dominant singular values of the gradient. Muon's empirical track record suggests that respecting the geometry of linear operators yields faster, more stable training.

## 3. Attention as a Bilinear Form

However, we can take the Muon idea one step further when applying it to Transformer attention layers. In the pre-softmax attention mechanism, the query matrix ($W_Q$) and key matrix ($W_K$) do not evaluate data independently.

Let $x_i$ and $x_j$ be the token embeddings (vectors in $\mathbb{R}^d$) for tokens $i$ and $j$. The attention mechanism evaluates their interaction not as isolated linear projections, but as a bilinear form. A bilinear form is a function $B: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ that is linear in both arguments. Every such bilinear form can be uniquely associated with a matrix $M \in \mathbb{R}^{d \times d}$ such that:


$$B(x_i, x_j) = x_i M x_j^T$$ {#eq-bilinear}

In the case of self-attention, the associated matrix is $W_{QK} = W_Q W_K^T$.

The critical realization is that **we are actually searching over the space of bilinear forms**, not independent linear operators. This space has its own norms and structures that are fundamentally different from the ones we would get if we looked at each matrix individually.

For instance, the supremum norm of a bilinear form measures its maximum possible output for bounded inputs:


$$\|B\|_\infty = \sup_{\substack{\|x\|_2 \le 1 \\ \|y\|_2 \le 1}} |B(x, y)|$$ {#eq-supnorm}


When we translate this norm back to its associated matrix $M$, it corresponds exactly to the spectral norm of the composite matrix ($\|M\|_2$).

Similarly, we can define the Euclidean energy of the bilinear form natively as the expected squared output under isotropic random inputs. If $x$ and $y$ are drawn from distributions with zero mean and identity covariance ($\mathbb{E}[x^T x] = I$, $\mathbb{E}[y^T y] = I$), the expected energy is:


$$\mathbb{E}\left[|B(x, y)|^2\right]$$


When mapped back to the associated matrix $M$, this geometric expectation corresponds exactly to the squared Frobenius norm of the matrix ($\|M\|_F^2$).

By treating $W_Q$ and $W_K$ as completely independent linear operators, current optimizers are mathematically blind to the true norm and geometric structure of the bilinear form that actually dictates the forward pass. To optimize attention correctly, we must optimize natively in the space of bilinear forms.

## 4. The Low-Rank Constraint

In practice, we do not search over the entire ambient space of bilinear forms. Neural networks use bilinear forms of the structure $x_i W_Q W_K^T x_j^T$, which correspond to a strictly low-rank subset. Specifically, the intermediate projection space has a dimension $d_k$ (the attention head dimension), which is typically much smaller than the hidden dimension $d$ (e.g., $d_k = 128$ while $d = 4096$). This limits the rank of the associated matrix to at most $d_k$.

This architectural constraint is highly deliberate and provides three major benefits:

1. **Computation:** It allows for significantly faster evaluation of the bilinear form in both the forward and backward passes due to matrix associativity (computing $(x_i W_Q) (x_j W_K)^T$ is vastly cheaper than multiplying by a full $d \times d$ matrix).
2. **Memory:** It requires storing only $2 d \cdot d_k$ parameters instead of the $d^2$ parameters required for a full-rank bilinear form.
3. **Generalization:** By drastically reducing the parameter count, it restricts the hypothesis space, acting as an implicit regularization that promotes smoother, more generalizable representations.

To optimize within this subset, our goal is to find an update $\Delta W_{QK}$ that maximizes descent along the composite gradient $G_{QK}$ (where $G_{QK} = \nabla_{W_{QK}} \mathcal{L}$), subject to both a continuous norm bound and the strict structural constraint of the architecture.

The steepest descent problem in its pure form is therefore:


$$
\begin{aligned}
\min_{\Delta W_{QK}} \quad & \langle G_{QK}, \Delta W_{QK} \rangle \\
\text{s.t.} \quad & \|\Delta W_{QK}\| \le \epsilon \\
& \text{rank}(W_{QK} + \Delta W_{QK}) \le d_k
\end{aligned}
$$ {#eq-rank-sd}

How do we actually enforce this rank constraint? By the mathematical definition of matrix rank, any $d \times d$ matrix has a rank of at most $d_k$ if and only if it can be perfectly factored into the product of two $d \times d_k$ matrices. Let us call these new ideal factors $A$ and $B$:


$$W_{QK} + \Delta W_{QK} = A B^T$$ {#eq-factor}

Since our current weights $W_Q$ and $W_K$ are fixed, we can reparameterize our search. Instead of searching for the ambient step $\Delta W_{QK}$, we search for the difference between these new ideal factors and our current weights:


$$
\Delta W_Q = A - W_Q, \qquad \Delta W_K = B - W_K
$$ {#eq-reparam}

Substituting these back into our factorization gives:


$$W_{QK} + \Delta W_{QK} = (W_Q + \Delta W_Q)(W_K + \Delta W_K)^T$$

Expanding the right side yields:


$$W_{QK} + \Delta W_{QK} = W_Q W_K^T + \Delta W_Q W_K^T + W_Q \Delta W_K^T + \Delta W_Q \Delta W_K^T$$ {#eq-expand}

Since we defined our current state as $W_{QK} = W_Q W_K^T$, we can subtract it from both sides to isolate the exact step taken in the ambient space of bilinear forms:


$$\Delta W_{QK} = \Delta W_Q W_K^T + W_Q \Delta W_K^T + \Delta W_Q \Delta W_K^T$$ {#eq-exact-step}

This is a useful geometric fact. Searching over the abstract, non-convex rank constraint of @eq-rank-sd in the ambient space is **mathematically identical** to searching over the unconstrained factors $\Delta W_Q$ and $\Delta W_K$ and applying this exact polynomial expansion. There is no rank-$d_k$ matrix reachable in the ambient space that cannot be reached natively through the factors. The architectural constraint forces the algebraic form.

## 5. The Tangent Approximation

By moving to the factorized view, we have translated an abstract rank constraint into a concrete algebraic equation. By construction, any update applied via the factors trivially satisfies the rank $d_k$ constraint.

However, substituting this exact $\Delta W_{QK}$ into our steepest descent objective introduces a significant mathematical hurdle: the quadratic term $\Delta W_Q \Delta W_K^T$. Because this bilinear coupling transforms the norm constraint into a non-convex set, the quadratic term will prove exceptionally difficult to handle analytically in the coming sections. There are two approaches we can take here to move forward.

The first approach is to keep the quadratic term and employ control-theoretic techniques, such as Linear Matrix Inequality (LMI) or Semidefinite Programming (SDP) relaxations, to bound the non-convex objective. While mathematically intriguing, these iterative solvers are not well-suited for the massive parallelism of modern GPUs, so we will not pursue them here.

The second approach is to ignore the quadratic term in @eq-exact-step. Assuming our step size bound $\epsilon$ is sufficiently small, the quadratic term $\Delta W_Q \Delta W_K^T$ (which scales with $\epsilon^2$) is negligible compared to the linear terms. Dropping it yields the approximation:


$$\Delta W_{QK} \approx \Delta W_Q W_K^T + W_Q \Delta W_K^T$$ {#eq-tangent}

Geometrically, this is the statement that our step lies strictly on the *tangent space* of the low-rank manifold at our current weights, a perspective we make precise, and then exploit, below.

This first-order tangent approximation captures the optimal descent direction while converting the intractable non-convex constraint into a strictly linear operator, unlocking exact, hardware-friendly analytical solutions.

### Reducing the Objective to the Factor Gradients

Using this tangent approximation has a powerful implication for our objective function. If we substitute our linearized step into the descent objective $\langle G_{QK}, \Delta W_{QK} \rangle$, we get:


$$\langle G_{QK}, \Delta W_Q W_K^T + W_Q \Delta W_K^T \rangle = \langle G_{QK}, \Delta W_Q W_K^T \rangle + \langle G_{QK}, W_Q \Delta W_K^T \rangle$$

Using the cyclic properties of the Frobenius inner product (trace), we can isolate the parameter updates:


$$\langle G_{QK}, \Delta W_Q W_K^T \rangle = \text{tr}(G_{QK}^T \Delta W_Q W_K^T) = \text{tr}(W_K^T G_{QK}^T \Delta W_Q) = \text{tr}((G_{QK} W_K)^T \Delta W_Q) = \langle G_{QK} W_K, \Delta W_Q \rangle$$

Similarly, for the second term:


$$\langle G_{QK}, W_Q \Delta W_K^T \rangle = \text{tr}(G_{QK}^T W_Q \Delta W_K^T) = \text{tr}((G_{QK}^T W_Q)^T \Delta W_K) = \langle G_{QK}^T W_Q, \Delta W_K \rangle$$

By the chain rule, $G_Q = \nabla_{W_Q} \mathcal{L} = G_{QK} W_K$ and $G_K = \nabla_{W_K} \mathcal{L} = G_{QK}^T W_Q$. Substituting these gives us the simplified steepest descent rule:

$$
\begin{aligned}
\min_{\Delta W_Q, \Delta W_K} \quad & \langle G_Q, \Delta W_Q \rangle + \langle G_K, \Delta W_K \rangle \\
\text{s.t.} \quad & \|\Delta W_Q W_K^T + W_Q \Delta W_K^T\| \le \epsilon
\end{aligned}
$$ {#eq-factor-sd}

This is the key simplification. We don't need to ever form the large $d \times d$ gradient $G_{QK}$. The downstream gradients of $W_Q$ and $W_K$ (which we would effortlessly get from a naive PyTorch implementation) are sufficient to optimize natively in the space of bilinear forms.

### An Explicit Characterization of the Tangent Space

There is a clean geometric reading of the linearized step. The tangent approximation @eq-tangent says precisely that $\Delta W_{QK}$ is constrained to the **tangent space** of the rank-$d_k$ manifold at the current point $W_{QK} = W_Q W_K^T$.

This tangent space admits an equivalent characterization stated directly in the ambient step $\Delta W_{QK}$, with no reference to the factors:[^tangent-proof]

$$\mathcal{T} = \{\, \Delta W_Q W_K^T + W_Q \Delta W_K^T \,\} = \{\, Z : (I - P_Q)\, Z\, (I - P_K) = 0 \,\}$$ {#eq-tangent-space}

where $P_Q$ and $P_K$ are the orthogonal projectors onto the column spaces of $W_Q$ and $W_K$. Read as a bilinear form $x_i\, Z\, x_j^T$, the constraint says the update may create *no* new interaction between a query direction invisible to $W_Q$ ($x_i \perp \operatorname{col}(W_Q)$) and a key direction invisible to $W_K$ ($x_j \perp \operatorname{col}(W_K)$). Reaching that doubly-invisible corner is precisely the second-order move carried by the quadratic term $\Delta W_Q \Delta W_K^T$ we dropped.

This hands us a second, fully equivalent way to write the steepest-descent step, now in the single ambient variable $\Delta W_{QK}$, with the tangent constraint stated explicitly:

$$\min_{\Delta W_{QK}} \;\langle G_{QK}, \Delta W_{QK} \rangle \quad \text{s.t.} \quad \|\Delta W_{QK}\| \le \epsilon, \quad (I - P_Q)\, \Delta W_{QK}\, (I - P_K) = 0$$ {#eq-geom-sd}

Although this uses the large $d \times d$ variable, the norm bound is now a clean norm of a *single* matrix rather than of a coupled sum of factor steps, which is what makes the upcoming derivations tractable. And despite being written with the ambient gradient $G_{QK}$, the resulting update rules will need neither $G_{QK}$ nor any other $d \times d$ matrix: everything reduces to the cheap factor gradients $G_Q$, $G_K$ and small $d_k \times d_k$ operations, as we will see.

This tangent-space viewpoint is not just a convenience. Optimizing along the tangent space of a structured weight manifold is the organizing principle behind recent work on modular manifolds (see for example this [blog post](https://thinkingmachines.ai/blog/modular-manifolds/) or the [modula library](https://docs.modula.systems/)). Here it falls out automatically from considering the space of low-rank bilinear forms in attention.


[^tangent-proof]: Proof. Here $W_Q^+$ and $W_K^+$ denote the Moore-Penrose pseudo-inverses, so that $P_Q = W_Q W_Q^+$ and $P_K = W_K W_K^+$. Both inclusions are direct. For "$\subseteq$", note that $(I - P_Q) W_Q = 0$ and $W_K^T (I - P_K) = 0$, so applying the double projection to any factorized step annihilates both of its terms. For "$\supseteq$", take any $Z$ with $(I - P_Q)\, Z\, (I - P_K) = 0$ and split it twice, once on the left by $P_Q$, then the remainder on the right by $P_K$:

    $$Z = P_Q Z + (I - P_Q) Z = P_Q Z + (I - P_Q) Z P_K + \underbrace{(I - P_Q)\, Z\, (I - P_K)}_{=\,0}$$

    The two surviving terms are each *already* in factorized form,

    $$P_Q Z = W_Q \underbrace{W_Q^+ Z}_{\Delta W_K^T}, \qquad (I - P_Q) Z P_K = \underbrace{(I - P_Q) Z\, (W_K^+)^T}_{\Delta W_Q}\, W_K^T,$$

    so $Z = \Delta W_Q W_K^T + W_Q \Delta W_K^T \in \mathcal{T}$. Whenever the orthogonality condition holds, projecting onto the two factor subspaces hands you the factors directly.


## 6. The $L^2$ Case: Analytical Frobenius Optimization

Now let's commit to a norm and derive the update, starting with the $L^2$ (Frobenius) norm. Rather than the hard norm-ball constraint of @eq-factor-sd, we use the equivalent **penalty** (Lagrangian) form: for a fixed norm the two share the same optimal *direction* and differ only in how the step is scaled, and the penalty form is far easier to differentiate. The natural first attempt is to optimize the factors directly:

$$\min_{\Delta W_Q, \Delta W_K} \quad \langle G_Q, \Delta W_Q \rangle + \langle G_K, \Delta W_K \rangle + \frac{1}{2\eta} \| \Delta W_Q W_K^T + W_Q \Delta W_K^T \|_F^2$$ {#eq-l2-factor}

Taking partial derivatives with respect to $\Delta W_Q$ and $\Delta W_K$ and setting them to zero yields the following coupled optimality conditions:

1. $\Delta W_Q (W_K^T W_K) + W_Q (\Delta W_K^T W_K) = -\eta G_Q$
2. $\Delta W_K (W_Q^T W_Q) + W_K (\Delta W_Q^T W_Q) = -\eta G_K$

This system consists of $2 d \cdot d_k$ unknowns defined by $2 d \cdot d_k$ linear equations. At first glance, this is a well-determined square system. However, the system is fundamentally underdetermined: it possesses a non-trivial null space. For any arbitrary $d_k \times d_k$ matrix $S$, consider the update:

$$\Delta W_Q = W_Q S, \quad \Delta W_K = -W_K S^T$$ {#eq-gauge}

If we substitute these into the homogeneous part of the first optimality condition, we get:

$$(W_Q S)(W_K^T W_K) + W_Q ((-W_K S^T)^T W_K) = W_Q S (W_K^T W_K) + W_Q (S W_K^T W_K) = W_Q S W_K^T W_K - W_Q S W_K^T W_K = 0$$

The same substitution holds for the second condition. This proves that there is an entire affine subspace of optimal solutions.

### Switching to the Gauge-Free Variable

While an under-determinacy is not in itself a problem (a linear solver handles it fine, for example by returning the minimum-norm solution), it often signals that a more insightful re-parameterization is available. That is the case here. The objective depends on the factors only through the tangent step $\Delta W_Q W_K^T + W_Q \Delta W_K^T$ (the penalty is its squared norm, and as shown above the linear part is $\langle G_{QK}, \cdot\rangle$ of it), so any direction that leaves this step fixed leaves the objective unchanged. And there are many: for any $d_k \times d_k$ matrix $S$, substituting $\Delta W_Q = W_Q S$ and $\Delta W_K = -W_K S^T$ gives

$$\Delta W_Q W_K^T + W_Q \Delta W_K^T = W_Q S W_K^T - W_Q S W_K^T = 0$$

The non-uniqueness, then, is not in *what* step we take in the space of bilinear forms, only in *how we split* it between the two factors. The factors are simply the wrong variable to solve for.

This is exactly why we built the geometric formulation @eq-geom-sd: it is written directly in the one object that *is* pinned down, $\Delta W_{QK}$, and never refers to the split. Committing to the $L^2$ norm, its penalty form reads:

$$\min_{\Delta W_{QK}} \;\langle G_{QK}, \Delta W_{QK} \rangle + \frac{1}{2\eta}\|\Delta W_{QK}\|_F^2 \quad \text{s.t.} \quad (I - P_Q)\, \Delta W_{QK}\, (I - P_K) = 0$$ {#eq-l2-geom}

This is no longer underdetermined: the objective is strictly convex on the tangent subspace, so it has a unique minimizer.

Steepest descent with a Frobenius penalty over a linear subspace is nothing but the orthogonal projection of the unconstrained gradient step onto that subspace. The projector onto the tangent space is

$$\mathcal{P}_{\mathcal{T}}(X) = P_Q X + X P_K - P_Q X P_K,$$ {#eq-proj}

so the solution is $\Delta W_{QK} = -\eta\, \mathcal{P}_{\mathcal{T}}(G_{QK})$. Using the chain-rule identities $G_{QK} W_K = G_Q$ and $W_Q^T G_{QK} = G_K^T$ to eliminate every appearance of the ambient gradient, and writing $W_Q^+, W_K^+$ for the Moore-Penrose pseudo-inverses, this becomes:

$$\Delta W_{QK} = -\eta\left[\, (W_Q^+)^T G_K^T + G_Q W_K^+ - (W_Q^+)^T (W_Q^T G_Q)\, W_K^+ \,\right]$$ {#eq-l2-sol}

Each pseudo-inverse hides only a tiny $d_k \times d_k$ inversion, and, just as promised, only the cheap factor gradients $G_Q$ and $G_K$ appear. The $d \times d$ gradient $G_{QK}$ never has to be formed.

All that remains is to hand the optimizer *some* factorization of this step: a pair $(\Delta W_Q, \Delta W_K)$ with $\Delta W_Q W_K^T + W_Q \Delta W_K^T = \Delta W_{QK}$. This is precisely the gauge freedom we met earlier, the null space of the factor system, resurfacing exactly where it is harmless: any valid split produces the same $\Delta W_{QK}$. A simple, immediate choice is:

$$
\Delta W_Q = -\eta\, G_Q (W_K^T W_K)^{-1}, \qquad \Delta W_K = -\eta\,(I - P_K)\, G_K (W_Q^T W_Q)^{-1}
$$ {#eq-l2-split}

These rules deserve a name: we'll call this update **Bilinear SGD**, the $L^2$ steepest-descent step in the bilinear geometry, the direct sibling of ordinary SGD in flat parameter space (which is itself nothing but $L^2$ steepest descent). It is worth pausing on what the rules say. Vanilla SGD would update each factor with its own gradient, $\Delta W_Q = -\eta\, G_Q$ and $\Delta W_K = -\eta\, G_K$, treating the two matrices as independent. Our rules instead right-multiply each factor gradient by the inverse Gram matrix of the *other* factor: $(W_K^T W_K)^{-1}$ for the query step, $(W_Q^T W_Q)^{-1}$ for the key step. The bilinear geometry couples them: the natural scale of a step in $W_Q$ is set by the size of $W_K$, and vice versa, so whichever factor currently carries more of the product's magnitude moves proportionally less.

This coupling is also what makes the step insensitive to the factorization's scale ambiguity. The reparameterization $W_Q \to W_Q R,\ W_K \to W_K R^{-T}$ leaves the product $W_{QK}$, and hence the loss, unchanged, yet it would alter the bare SGD step. Here the induced update $\Delta W_{QK} = -\eta\, \mathcal{P}_{\mathcal{T}}(G_{QK})$ depends on the factors only through their column spaces, which the gauge preserves, so it is invariant by construction. SGD descends on the two matrices, while this descends on the bilinear form they define.

Stepping back: by linearizing onto the tangent space and solving for the gauge-invariant step $\Delta W_{QK}$ rather than the redundant factors, the entire $L^2$ derivation collapsed to a handful of $d_k \times d_k$ operations, with no ill-posed coupled solve and no $d \times d$ matrix ever formed.

Frobenius was only the first norm we could have chosen. The geometric formulation, a linear objective minimized over the tangent space under a norm bound, is norm-agnostic, and only the inner solve changes. Next we take the norm the bilinear picture really calls for: the sup norm, which (as we saw earlier) is the spectral norm of the form. There the clean projection no longer applies, but the steepest-descent step still does, and it leads us back to Muon.


## 7. The $L^\infty$ Case: From the Exact Problem to Closed-Form Restrictions

Under the sup norm the constraint bounds the spectral norm of the bilinear step, $\|\Delta W_{QK}\|_2 \le \epsilon$, and the steepest-descent problem on the tangent space becomes

$$\min_{\Delta W_Q,\, \Delta W_K} \;\langle G_Q, \Delta W_Q\rangle + \langle G_K, \Delta W_K\rangle \quad \text{s.t.}\quad \big\|\, \Delta W_Q W_K^T + W_Q \Delta W_K^T \,\big\|_2 \le \epsilon.$$ {#eq-linf}

This is still convex, a linear objective over a spectral-norm ball, but the two factors now sit inside a *single* spectral norm, with no inner product to diagonalize and no orthogonal projection to read off. Instead of one clean solution we get a frontier: a nested family of constraint sets

$$\mathcal{F}_t^{\times} \;\subseteq\; \mathcal{F}_t \;\subseteq\; \mathcal{F}_\triangle \;\subseteq\; \mathcal{F} := \{(\Delta W_Q, \Delta W_K) : \|\Delta W_{QK}\|_2 \le \epsilon\},$$ {#eq-nested}

each a strictly smaller (more conservative) *inner* approximation of the true feasible set $\mathcal{F}$, and each cheaper to optimize over. Since every set sits inside $\mathcal{F}$, any step we produce is automatically a valid $\epsilon$-bounded step. We are only trading away some of the budget in return for a closed form.

### I. The exact convex problem

Problem @eq-linf is convex, the exact, tightest step on the tangent space, but the two factors are locked inside a single spectral norm, so there is no closed form. To use it in practice we would need a GPU-friendly iterative solver, for example an ADMM scheme built on iterative singular-value-thresholding approximations rather than exact SVDs. Solving such a problem from scratch at every step and every head looks hopeless, with each inner iteration pure overhead. But there is a hand-wavy way around this that I find appealing. The weights barely move between optimizer steps, so the solution of @eq-linf drifts only slowly; this is exactly the regime where a *warm-started* solver, initialized from the previous step's solution, needs almost no work to catch up. Concretely, one could promote the solver's auxiliary and dual variables to persistent optimizer state, right alongside the momentum buffers that every optimizer already carries per parameter, and advance them by a *single* ADMM iteration per step. The inner solve and the outer optimizer then co-evolve over training rather than nesting one inside the other, at a cost of one iteration per step. This single-loop, warm-started pattern is a recurring trick wherever an inner subproblem sits inside an outer optimization, and it seems like the most promising route to the exact step; I leave a real treatment to future work. The remaining restrictions instead give up this exactness in exchange for a closed form.

### II. The triangle-inequality restriction: preconditioned Muon

Split the single spectral norm into two. For any budget share $t \in [0,1]$,

$$\mathcal{F} \;\supseteq\; \underbrace{\{\|\Delta W_Q W_K^T\|_2 + \|W_Q \Delta W_K^T\|_2 \le \epsilon\}}_{\mathcal{F}_\triangle} \;\supseteq\; \underbrace{\{\|\Delta W_Q W_K^T\|_2 \le t\epsilon\} \cap \{\|W_Q \Delta W_K^T\|_2 \le (1-t)\epsilon\}}_{\mathcal{F}_t},$$ {#eq-tri-sets}

where the first inclusion is the triangle inequality, and the second holds because the two shares sum to $\epsilon$. Over $\mathcal{F}_t$ the factors decouple, so we can solve for each separately. Take the query factor:

$$\min_{\Delta W_Q}\;\langle G_Q, \Delta W_Q\rangle \quad \text{s.t.}\quad \|\Delta W_Q W_K^T\|_2 \le t\epsilon.$$ {#eq-tri-query}

This is almost the Muon problem @eq-muon, a linear form minimized over a spectral ball, except the constraint bounds $\Delta W_Q W_K^T$, a linear image of $\Delta W_Q$, rather than $\Delta W_Q$ itself. To bring it into Muon form, change variables to $Z = \Delta W_Q W_K^T$. Picking any left inverse $L$ of $W_K$,[^rank-deficiency] the problem becomes

$$\min_{Z}\;\langle G_Q L, Z\rangle \quad \text{s.t.}\quad \|Z\|_2 \le t\epsilon,\quad Z P_K = Z.$$ {#eq-tri-z}

We have traded the linear term inside the norm for an ordinary spectral ball plus the column-space constraint $Z P_K = Z$. Now let's use the freedom in $L$. Drop that constraint for a moment: minimizing $\langle G_Q L, Z\rangle$ over the bare ball is exactly Muon, with solution $Z = -t\epsilon\,\mathrm{Orth}(G_Q L)$, which depends on $L$. If we can pick $L$ so that this $Z$ already satisfies $Z P_K = Z$, the dropped constraint cost us nothing and we are done. The pseudo-inverse $L = W_K^+$ does exactly that: the rows of $W_K^+$ lie in $\operatorname{col}(W_K)$, so the rows of $G_Q W_K^+$ do too, and since $\mathrm{Orth}$ leaves the row space unchanged, $\mathrm{Orth}(G_Q W_K^+)\,P_K = \mathrm{Orth}(G_Q W_K^+)$. Recovering $\Delta W_Q = Z L^T$ with $L = W_K^+$ then gives

$$\Delta W_Q = -\,t\epsilon\;\mathrm{Orth}\!\big(G_Q W_K^+\big)\,(W_K^T)^+,$$ {#eq-tri-sol}

and symmetrically $\Delta W_K = -(1-t)\epsilon\,\mathrm{Orth}(G_K W_Q^+)(W_Q^T)^+$. This is a Muon step sandwiched by pseudo-inverses of the *other* factor, the spectral analog of the Gram-matrix preconditioning from the Frobenius case. It is fully closed-form: a fixed number of orthogonalizations and small $d_k \times d_k$ inverses, with no inner loop.

What should $t$ be? The natural answer is to pick the split that buys the most descent. A spectral-ball steepest step of radius $r$ against a gradient $G$ decreases the objective by $r\,\|G\|_*$, where $\|\cdot\|_*$ is the nuclear norm (the sum of singular values, dual to the spectral norm). The query step has radius $t\epsilon$ against the preconditioned gradient $G_Q W_K^+$ and the key step radius $(1-t)\epsilon$ against $G_K W_Q^+$, so the total descent over $\mathcal{F}_t$ is

$$D(t) = t\epsilon\,\|G_Q W_K^+\|_* \;+\; (1-t)\epsilon\,\|G_K W_Q^+\|_*.$$ {#eq-tri-Dt}

This is *linear* in $t$, with slope $\epsilon\big(\|G_Q W_K^+\|_* - \|G_K W_Q^+\|_*\big)$, so the maximum over $[0,1]$ sits at an endpoint:

$$t^\star = \begin{cases} 1, & \|G_Q W_K^+\|_* \ge \|G_K W_Q^+\|_*, \\[2pt] 0, & \text{otherwise.} \end{cases}$$ {#eq-tstar}

The optimal split is **winner-take-all**: it spends the entire budget on whichever factor has the larger preconditioned nuclear norm and leaves the other untouched. This is not just the best choice among our boxes $\mathcal{F}_t$: it turns out to be optimal over the entire set we restricted to, the triangle superset $\mathcal{F}_\triangle$.[^triangle-opt]

I find this surprising: taken literally, the rule says we should update only one of $W_Q$ and $W_K$ on each step. My intuition is that this won't actually work well in practice (the inter-step dynamics of freezing a different factor each step seem likely to hurt), but I haven't tested it, and it would be interesting to race this winner-take-all split against a plain $t = \tfrac12$ split that updates both factors evenly. For the rest of this post I'll stick with the latter. I'll call the two variants **winner-take-all** and **symmetric preconditioned Muon**.

[^triangle-opt]: This is one instance of a general fact about problems coupled only through a shared budget. For two blocks,
$$V = \min_{x,\,y}\ f(x) + g(y) \quad\text{s.t.}\quad a(x) + b(y) \le \epsilon \qquad (a, b \ge 0),$$
define each block's *value function* (the best objective reachable on a given budget) as $v_f(s) = \inf_{a(x) \le s} f(x)$ and $v_g(s) = \inf_{b(y) \le s} g(y)$. Then solving the coupled problem is the same as optimally splitting the budget:
$$V = \inf_{0 \le s \le \epsilon}\ v_f(s) + v_g(\epsilon - s).$$
Both directions are immediate: any split $s$ gives a feasible pair, so $V \le v_f(s) + v_g(\epsilon - s)$. Conversely a feasible $(x, y)$ has $b(y) \le \epsilon - a(x)$, so taking $s = a(x)$ gives $f(x) + g(y) \ge v_f(s) + v_g(\epsilon - s)$. Our share $t$ is just $s = t\epsilon$, so optimizing $t$ optimizes over the entire set $\mathcal{F}_\triangle$, not merely one box $\mathcal{F}_t$. What makes the answer one-sided is that here the value functions are *linear*: spectral/nuclear duality gives $v_f(s) = -s\,\|G_Q W_K^+\|_*$ and $v_g(s) = -s\,\|G_K W_Q^+\|_*$, so $v_f(s) + v_g(\epsilon - s)$ is linear in $s$ and minimized at an endpoint. The sub-multiplicative restriction (III) is the same problem with $v_f(s) = -s\,\|G_Q\|_*/\|W_K\|_2$ and $v_g(s) = -s\,\|G_K\|_*/\|W_Q\|_2$, still linear, so the one-factor conclusion carries over unchanged.

[^rank-deficiency]: A left inverse exists exactly when $W_K$ has full column rank, which we assume throughout. While this typically holds for a thin $d \times d_k$ matrix with $d \gg d_k$, the caveat hints that the method may become unstable for almost-low-rank weights, where the pseudo-inverse blows up. This is worth exploring in the future, perhaps with added regularization.

### III. The sub-multiplicative restriction: cross-scaled Muon

We can tighten the constraint once more using sub-multiplicativity, with $\|\Delta W_Q W_K^T\|_2 \le \|\Delta W_Q\|_2\,\|W_K\|_2$ and $\|W_Q \Delta W_K^T\|_2 \le \|W_Q\|_2\,\|\Delta W_K\|_2$, and then split the budget as before:

$$\mathcal{F}_\triangle \;\supseteq\; \underbrace{\{\|\Delta W_Q\|_2\|W_K\|_2 + \|W_Q\|_2\|\Delta W_K\|_2 \le \epsilon\}}_{\mathcal{F}_\triangle^{\times}} \;\supseteq\; \underbrace{\{\|\Delta W_Q\|_2\|W_K\|_2 \le t\epsilon\} \cap \{\|W_Q\|_2\|\Delta W_K\|_2 \le (1-t)\epsilon\}}_{\mathcal{F}_t^{\times}},$$ {#eq-sub-sets}

with the same two reasons as in the triangle case: the first inclusion is sub-multiplicativity, the second holds because the two shares sum to $\epsilon$.

Over $\mathcal{F}_t^{\times}$ the factors decouple, and the query is now bounded through $\|\Delta W_Q\|_2$ directly. Contrast @eq-tri-query, where the bound sat on the linear image $\Delta W_Q W_K^T$:

$$\min_{\Delta W_Q}\;\langle G_Q, \Delta W_Q\rangle \quad \text{s.t.}\quad \|\Delta W_Q\|_2 \le \frac{t\epsilon}{\|W_K\|_2}.$$ {#eq-sub-query}

This is exactly the Muon problem @eq-muon, a linear form over a spectral ball of the factor itself, with no change of variables or pseudo-inverse. Its solution is a plain Muon step, rescaled by the partner's spectral norm,

$$\Delta W_Q = -\,\frac{t\epsilon}{\|W_K\|_2}\,\mathrm{Orth}(G_Q), \qquad \Delta W_K = -\,\frac{(1-t)\epsilon}{\|W_Q\|_2}\,\mathrm{Orth}(G_K).$$ {#eq-cross}

Set against plain Muon run separately on each factor, the only difference is the cross-scaling: the $W_Q$ step is divided by $\|W_K\|_2$ and the $W_K$ step by $\|W_Q\|_2$. Read as a prescription, it suggests that a Muon update on attention factors should, at the very least, scale each factor's step by the spectral norm of its partner.

The split $t$ enters exactly as before. The achievable descent is

$$D(t) = t\epsilon\,\frac{\|G_Q\|_*}{\|W_K\|_2} \;+\; (1-t)\epsilon\,\frac{\|G_K\|_*}{\|W_Q\|_2},$$ {#eq-sub-Dt}

linear in $t$ once again, with only the per-factor rates changed from the triangle case @eq-tri-Dt, from the preconditioned nuclear norms to the ratios $\|G_Q\|_*/\|W_K\|_2$ and $\|G_K\|_*/\|W_Q\|_2$. So the same budget-split argument applies, and the optimum is again **winner-take-all**: the whole budget on whichever of $W_Q$, $W_K$ has the larger ratio. As in the triangle case, this gives two variants, **winner-take-all** and **symmetric cross-scaled Muon**, where the symmetric $t = \tfrac12$ choice updates both factors with an ordinary Muon step, each damped by the spectral size of its partner.

It is worth setting all three preconditioned steps side by side, because they share one skeleton: each factor's gradient is reshaped by the geometry of its *partner*, never its own. Bilinear SGD, the Frobenius step of the previous section, multiplies $G_Q$ by the partner's inverse Gram matrix $(W_K^T W_K)^{-1}$. Preconditioned Muon applies that exact same matrix, since $W_K^+ (W_K^T)^+ = (W_K^T W_K)^{-1}$, but splits it apart and inserts an orthogonalization into the gap, giving $\mathrm{Orth}(G_Q W_K^+)(W_K^T)^+$. That single inserted orthogonalization is all that separates the $L^2$ step from the $L^\infty$ one. Cross-scaled Muon then discards the matrix altogether and keeps only its scale, dividing a plain Muon step by the scalar $\|W_K\|_2$. The cross-coupling that the bilinear geometry demands survives in every case, carried first by a full matrix metric, then by that metric wrapped around an orthogonalization, and finally by a single number.

### Computing the norms

The update rules above require the spectral norm $\|W\|_2$ and the nuclear norm $\|G\|_*$. Both would naively call for an SVD, but each can be computed cheaply, to reasonable accuracy, in a GPU-friendly way.

The spectral norm is the largest singular value, which a few steps of **power iteration** recover to ample accuracy. Each step is just a couple of matrix-vector products.

The nuclear norm is the sum of singular values, and it falls out of the orthogonalization Muon already performs. With $\mathrm{Orth}(M) = UV^T$ for the SVD $M = U\Sigma V^T$,

$$\|M\|_* = \operatorname{tr}(\Sigma) = \operatorname{tr}\!\big(\mathrm{Orth}(M)^T M\big) = \langle \mathrm{Orth}(M),\, M\rangle.$$ {#eq-nuclear}

So once the Newton-Schulz iteration has produced $\mathrm{Orth}(M)$, the very step Muon computes anyway, the nuclear norm is one Frobenius inner product away.

## 8. Summary

Each optimizer in this post comes down to a single steepest-descent step, fixed by three choices: the *space* it descends in, the *norm* that bounds the step, and the *approximations* that buy a closed form. @tbl-variants lines them up.

| Name | Update rule | Space | Norm | Approximations |
|---|---|---|---|---|
| SGD | $\Delta W = -\eta\,G$ | flattened params | $L^2$ | none |
| Muon | $\Delta W = -\epsilon\,\mathrm{Orth}(G)$ | linear operators | $L^\infty$ | none |
| Bilinear SGD | $\Delta W_Q = -\eta\,G_Q (W_K^T W_K)^{-1}$<br>$\Delta W_K = -\eta\,(I-P_K)\,G_K (W_Q^T W_Q)^{-1}$ | low-rank bilinear forms | $L^2$ | tangent space |
| Winner-take-all preconditioned Muon | if $\lVert G_Q W_K^+\rVert_* \ge \lVert G_K W_Q^+\rVert_*$:<br>$\Delta W_Q = -\epsilon\,\mathrm{Orth}(G_Q W_K^+)(W_K^T)^+$<br>$\Delta W_K = 0$<br>otherwise:<br>$\Delta W_Q = 0$<br>$\Delta W_K = -\epsilon\,\mathrm{Orth}(G_K W_Q^+)(W_Q^T)^+$ | low-rank bilinear forms | $L^\infty$ | tangent space, triangle inequality |
| Symmetric preconditioned Muon | $\Delta W_Q = -\tfrac{\epsilon}{2}\,\mathrm{Orth}(G_Q W_K^+)(W_K^T)^+$<br>$\Delta W_K = -\tfrac{\epsilon}{2}\,\mathrm{Orth}(G_K W_Q^+)(W_Q^T)^+$ | low-rank bilinear forms | $L^\infty$ | tangent space, triangle inequality, symmetric split ($t=\tfrac12$) |
| Winner-take-all cross-scaled Muon | if $\lVert G_Q\rVert_*/\lVert W_K\rVert_2 \ge \lVert G_K\rVert_*/\lVert W_Q\rVert_2$:<br>$\Delta W_Q = -\epsilon\,\mathrm{Orth}(G_Q)/\lVert W_K\rVert_2$<br>$\Delta W_K = 0$<br>otherwise:<br>$\Delta W_Q = 0$<br>$\Delta W_K = -\epsilon\,\mathrm{Orth}(G_K)/\lVert W_Q\rVert_2$ | low-rank bilinear forms | $L^\infty$ | tangent space, triangle inequality, sub-multiplicative norm |
| Symmetric cross-scaled Muon | $\Delta W_Q = -\tfrac{\epsilon}{2}\,\mathrm{Orth}(G_Q)/\lVert W_K\rVert_2$<br>$\Delta W_K = -\tfrac{\epsilon}{2}\,\mathrm{Orth}(G_K)/\lVert W_Q\rVert_2$ | low-rank bilinear forms | $L^\infty$ | tangent space, triangle inequality, sub-multiplicative norm, symmetric split ($t=\tfrac12$) |

: Every optimizer in this post as a single steepest-descent step. The $L^2$ rows are written in penalty (learning-rate $\eta$) form, the $L^\infty$ rows in trust-region (radius $\epsilon$) form, and $W^+$ denotes the Moore-Penrose pseudo-inverse. {#tbl-variants}

## 9. Related Work

A few days before I posted this, Tilde Research published [Compositional Muon](https://blog.tilderesearch.com/blog/compositional-muon), which arrives at the same central idea from a similar starting point: the loss sees $W_Q$ and $W_K$ only through their product, so steepest descent should be run on the composed map rather than on the two factors independently. I came across it only while writing this up, and the overlap is substantial enough that I want to be upfront about it.

The correspondence runs deep. Their derivation passes through the same first-order expansion, the same observation that the achievable steps form the tangent space to the rank-$d_k$ manifold, and the same gauge freedom in how a step is split between the factors. More concretely, two of the practical rules I land on are theirs as well. My symmetric *preconditioned Muon*, $\Delta W_Q = -\tfrac{\epsilon}{2}\,\mathrm{Orth}(G_Q W_K^+)(W_K^T)^+$, is algebraically identical to their half-split rule $\Delta W_Q = -\tfrac{\epsilon}{2}\,\mathrm{msign}(G_Q C_K^{-1})\,C_K^{-1}$ (with $C_K = (W_K^T W_K)^{1/2}$, the two coincide because $\mathrm{Orth}(G_Q W_K^+)(W_K^T)^+ = \mathrm{Orth}(G_Q C_K^{-1})\,C_K^{-1}$), and my *cross-scaled Muon* is their isotropic approximation up to whether the partner is measured by its spectral norm or its RMS scale. They also go further than I do here: they treat the $W_O W_V$ pathway, work out how the gauge interacts with momentum, and, most importantly, actually run the thing, reporting consistent gains over Muon at scale and a new state of the art on the nanoGPT speedrun.

Two pieces of this post have no counterpart in theirs. The first is the $L^2$ side entirely: Compositional Muon works only under the spectral norm, so there is nothing there like *Bilinear SGD*, the closed-form Frobenius step I get from orthogonal projection onto the tangent space. The second is the budget split. Their half-split fixes the $t=\tfrac12$ allocation from the start and notes only that it is arbitrary; I keep $t$ free, show the achievable descent is linear in it, and conclude that the optimum is **winner-take-all**, spending the whole budget on a single factor each step. And this is not merely the best split among the boxes $\mathcal{F}_t$: as I prove, it is optimal over the entire sum-of-norms set $\mathcal{F}_\triangle$ that the triangle inequality hands us, so no allocation of any kind does better under that relaxation. So their two rules sit inside my taxonomy as the symmetric $L^\infty$ entries, with the $L^2$ row and the winner-take-all row genuinely new here. The rest of what this post adds is framing: the explicit bilinear-form view, under which the two relevant norms (the sup norm as the spectral norm of the form, the energy as its Frobenius norm) fall out naturally, and the organization of the $L^\infty$ variants as a single nested chain of inner approximations (exact $\subseteq$ triangle $\subseteq$ sub-multiplicative), each trading approximation budget for GPU-friendliness.

## 10. Future Work

The most obvious next step is to actually run these optimizers and measure how they perform. The analysis here says nothing about which variant wins in practice, or whether the winner-take-all split really does hurt as much as I suspect.

A second direction is to look for better solutions to the exact sup-norm problem on the tangent space (@eq-linf). The closed-form variants in this post all lean on the triangle inequality, which is a fairly crude inner approximation. I suspect there is room for a method that stays GPU-friendly yet gives up far less of the budget, perhaps through a dual decomposition of the coupled spectral norm that keeps each inner step cheap, or through the single-loop, warm-started ADMM scheme sketched in [section 7](#i-the-exact-convex-problem) that folds the solver state into the optimizer and takes one iteration per step. This is the direction I plan to explore next.

A third direction is regularization. Decoupled weight decay (AdamW) is a near-universal improvement over plain Adam, which motivates adding a regularization term to the steepest-descent problems above. That turns each update into a proximal step, a shrinkage operation, but now one that acts on the bilinear form itself (say, Frobenius of $W_{QK}$) rather than on the raw factor parameters, which is arguably the more natural object to penalize.