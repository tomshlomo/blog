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

Geometrically, this is the statement that our step lies strictly on the *tangent space* of the low-rank manifold at our current weights—a perspective we make precise, and then exploit, below.

This first-order tangent approximation captures the optimal descent direction while converting the intractable non-convex constraint into a strictly linear operator, unlocking exact, hardware-friendly analytical solutions.

### A Beautiful Consequence

Using this tangent approximation has a powerful implication for our objective function. If we substitute our linearized step into the descent objective $\langle G_{QK}, \Delta W_{QK} \rangle$, we get:


$$\langle G_{QK}, \Delta W_Q W_K^T + W_Q \Delta W_K^T \rangle = \langle G_{QK}, \Delta W_Q W_K^T \rangle + \langle G_{QK}, W_Q \Delta W_K^T \rangle$$

Using the cyclic properties of the Frobenius inner product (trace), we can isolate the parameter updates:


$$\langle G_{QK}, \Delta W_Q W_K^T \rangle = \text{tr}(G_{QK}^T \Delta W_Q W_K^T) = \text{tr}(W_K^T G_{QK}^T \Delta W_Q) = \text{tr}((G_{QK} W_K)^T \Delta W_Q) = \langle G_{QK} W_K, \Delta W_Q \rangle$$

Similarly, for the second term:


$$\langle G_{QK}, W_Q \Delta W_K^T \rangle = \text{tr}(G_{QK}^T W_Q \Delta W_K^T) = \text{tr}((G_{QK}^T W_Q)^T \Delta W_K) = \langle G_{QK}^T W_Q, \Delta W_K \rangle$$

By the chain rule, $G_Q = \nabla_{W_Q} \mathcal{L} = G_{QK} W_K$ and $G_K = \nabla_{W_K} \mathcal{L} = G_{QK}^T W_Q$. Substituting these gives us the simplified steepest descent rule:

$$\min_{\Delta W_Q, \Delta W_K} \langle G_Q, \Delta W_Q \rangle + \langle G_K, \Delta W_K \rangle$$

$$\text{s.t.} \quad \|\Delta W_Q W_K^T + W_Q \Delta W_K^T\| \le \epsilon$$

This is beautiful—we don't need to ever form the large $d \times d$ gradient $G_{QK}$. The downstream gradients of $W_Q$ and $W_K$ (which we would effortlessly get from a naive PyTorch implementation) are sufficient to optimize natively in the space of bilinear transformations.

### The Tangent Space, Made Explicit

There is a clean geometric reading of the linearized step. The approximation

$$\Delta W_{QK} \approx \Delta W_Q W_K^T + W_Q \Delta W_K^T$$

says precisely that $\Delta W_{QK}$ is constrained to the **tangent space** of the rank-$d_k$ manifold at the current point $W_{QK} = W_Q W_K^T$.

This tangent space admits an equivalent characterization stated directly in the ambient step $\Delta W_{QK}$, with no reference to the factors:[^tangent-proof]

$$\mathcal{T} = \{\, \Delta W_Q W_K^T + W_Q \Delta W_K^T \,\} = \{\, Z : (I - P_Q)\, Z\, (I - P_K) = 0 \,\}$$

where $P_Q$ and $P_K$ are the orthogonal projectors onto the column spaces of $W_Q$ and $W_K$. Read as a bilinear form $x_i\, Z\, x_j^T$, the constraint says the update may create *no* new interaction between a query direction invisible to $W_Q$ ($x_i \perp \operatorname{col}(W_Q)$) and a key direction invisible to $W_K$ ($x_j \perp \operatorname{col}(W_K)$); reaching that doubly-invisible corner is precisely the second-order move carried by the quadratic term $\Delta W_Q \Delta W_K^T$ we dropped.

This hands us a second, fully equivalent way to write the steepest-descent step—now in the single ambient variable $\Delta W_{QK}$, with the tangent constraint stated explicitly:

$$\min_{\Delta W_{QK}} \;\langle G_{QK}, \Delta W_{QK} \rangle \quad \text{s.t.} \quad \|\Delta W_{QK}\| \le \epsilon, \quad (I - P_Q)\, \Delta W_{QK}\, (I - P_K) = 0$$

Although this uses the large $d \times d$ variable, the norm bound is now a clean norm of a *single* matrix rather than of a coupled sum of factor steps—which is what makes the upcoming derivations tractable. And despite being written with the ambient gradient $G_{QK}$, the resulting update rules will need neither $G_{QK}$ nor any other $d \times d$ matrix: everything reduces to the cheap factor gradients $G_Q$, $G_K$ and small $d_k \times d_k$ operations, as we will see.

This tangent-space viewpoint is not just a convenience. Optimizing along the tangent space of a structured weight manifold is the organizing principle behind recent work on modular manifolds (see for example this [blog post](https://thinkingmachines.ai/blog/modular-manifolds/) or the [modula library](https://docs.modula.systems/)); here it falls out automatically from considering the space of low-rank bilinear transformations in attention.


[^tangent-proof]: Proof. Here $W_Q^+$ and $W_K^+$ denote the Moore–Penrose pseudo-inverses, so that $P_Q = W_Q W_Q^+$ and $P_K = W_K W_K^+$. Both inclusions are direct. For "$\subseteq$", note that $(I - P_Q) W_Q = 0$ and $W_K^T (I - P_K) = 0$, so applying the double projection to any factorized step annihilates both of its terms. For "$\supseteq$", take any $Z$ with $(I - P_Q)\, Z\, (I - P_K) = 0$ and split it twice—once on the left by $P_Q$, then the remainder on the right by $P_K$:

    $$Z = P_Q Z + (I - P_Q) Z = P_Q Z + (I - P_Q) Z P_K + \underbrace{(I - P_Q)\, Z\, (I - P_K)}_{=\,0}$$

    The two surviving terms are each *already* in factorized form,

    $$P_Q Z = W_Q \underbrace{W_Q^+ Z}_{\Delta W_K^T}, \qquad (I - P_Q) Z P_K = \underbrace{(I - P_Q) Z\, (W_K^+)^T}_{\Delta W_Q}\, W_K^T,$$

    so $Z = \Delta W_Q W_K^T + W_Q \Delta W_K^T \in \mathcal{T}$. Whenever the orthogonality condition holds, projecting onto the two factor subspaces hands you the factors directly.


## 6. The $L^2$ Crown Jewel: Analytical Frobenius Optimization

Now let's commit to a norm and derive the update, starting with the $L^2$ (Frobenius) norm. Rather than the hard norm-ball constraint, we use the equivalent **penalty** (Lagrangian) form: for a fixed norm the two share the same optimal *direction* and differ only in how the step is scaled, and the penalty form is far easier to differentiate. The natural first attempt is to optimize the factors directly:

$$\max_{\Delta W_Q, \Delta W_K} \quad \langle G_Q, \Delta W_Q \rangle + \langle G_K, \Delta W_K \rangle - \frac{1}{2\eta} \| \Delta W_Q W_K^T + W_Q \Delta W_K^T \|_F^2$$

Taking partial derivatives with respect to $\Delta W_Q$ and $\Delta W_K$ and setting them to zero yields the following coupled optimality conditions:

1. $\Delta W_Q (W_K^T W_K) + W_Q (\Delta W_K^T W_K) = \eta G_Q$
2. $\Delta W_K (W_Q^T W_Q) + W_K (\Delta W_Q^T W_Q) = \eta G_K$

This system consists of $2 d \cdot d_k$ unknowns defined by $2 d \cdot d_k$ linear equations. At first glance, this is a well-determined square system. However, the system is fundamentally underdetermined: it possesses a non-trivial null space. For any arbitrary $d_k \times d_k$ matrix $S$, consider the update:

$$\Delta W_Q = W_Q S, \quad \Delta W_K = -W_K S^T$$

If we substitute these into the homogeneous part of the first optimality condition, we get:

$$(W_Q S)(W_K^T W_K) + W_Q ((-W_K S^T)^T W_K) = W_Q S (W_K^T W_K) + W_Q (S W_K^T W_K) = W_Q S W_K^T W_K - W_Q S W_K^T W_K = 0$$

The same substitution holds for the second condition. This proves that there is an entire affine subspace of optimal solutions.

### Switching to the Gauge-Free Variable

While an under-determinacy is not in itself a problem—a linear solver handles it fine, for example by returning the minimum-norm solution—it often signals that a more insightful re-parameterization is available. That is the case here. The objective depends on the factors only through the tangent step $\Delta W_Q W_K^T + W_Q \Delta W_K^T$ (the penalty is its squared norm, and as shown above the linear part is $\langle G_{QK}, \cdot\rangle$ of it), so any direction that leaves this step fixed leaves the objective unchanged. And there are many—for any $d_k \times d_k$ matrix $S$, substituting $\Delta W_Q = W_Q S$ and $\Delta W_K = -W_K S^T$ gives

$$\Delta W_Q W_K^T + W_Q \Delta W_K^T = W_Q S W_K^T - W_Q S W_K^T = 0$$

The non-uniqueness, then, is not in *what* step we take in the space of bilinear forms—only in *how we split* it between the two factors. The factors are simply the wrong variable to solve for.

This is exactly why we built the geometric formulation in the previous section: it is written directly in the one object that *is* pinned down, $\Delta W_{QK}$, and never refers to the split. Committing to the $L^2$ norm, its penalty form reads:

$$\max_{\Delta W_{QK}} \;\langle G_{QK}, \Delta W_{QK} \rangle - \frac{1}{2\eta}\|\Delta W_{QK}\|_F^2 \quad \text{s.t.} \quad (I - P_Q)\, \Delta W_{QK}\, (I - P_K) = 0$$

This is no longer underdetermined: the objective is strictly convex on the tangent subspace, so it has a unique minimizer.

Steepest descent with a Frobenius penalty over a linear subspace is nothing but the orthogonal projection of the unconstrained gradient step onto that subspace. The projector onto the tangent space is

$$\mathcal{P}_{\mathcal{T}}(X) = P_Q X + X P_K - P_Q X P_K,$$

so the solution is $\Delta W_{QK} = \eta\, \mathcal{P}_{\mathcal{T}}(G_{QK})$. Using the chain-rule identities $G_{QK} W_K = G_Q$ and $W_Q^T G_{QK} = G_K^T$ to eliminate every appearance of the ambient gradient—and writing $W_Q^+, W_K^+$ for the Moore–Penrose pseudo-inverses—this becomes:

$$\Delta W_{QK} = \eta\left[\, (W_Q^+)^T G_K^T + G_Q W_K^+ - (W_Q^+)^T (W_Q^T G_Q)\, W_K^+ \,\right]$$

Each pseudo-inverse hides only a tiny $d_k \times d_k$ inversion, and—just as promised—only the cheap factor gradients $G_Q$ and $G_K$ appear; the $d \times d$ gradient $G_{QK}$ never has to be formed.

All that remains is to hand the optimizer *some* factorization of this step: a pair $(\Delta W_Q, \Delta W_K)$ with $\Delta W_Q W_K^T + W_Q \Delta W_K^T = \Delta W_{QK}$. This is precisely the gauge freedom we met earlier—the null space of the factor system—resurfacing exactly where it is harmless: any valid split produces the same $\Delta W_{QK}$. A simple, immediate choice is:

$$\Delta W_Q = \eta\, G_Q (W_K^T W_K)^{-1}$$

$$\Delta W_K = \eta\,(I - P_K)\, G_K (W_Q^T W_Q)^{-1}$$

It is worth pausing on what these rules say. Vanilla SGD would update each factor with its own gradient, $\Delta W_Q = \eta\, G_Q$ and $\Delta W_K = \eta\, G_K$, treating the two matrices as independent. Our rules instead right-multiply each factor gradient by the inverse Gram matrix of the *other* factor—$(W_K^T W_K)^{-1}$ for the query step, $(W_Q^T W_Q)^{-1}$ for the key step. The bilinear geometry couples them: the natural scale of a step in $W_Q$ is set by the size of $W_K$, and vice versa, so whichever factor currently carries more of the product's magnitude moves proportionally less.

This coupling is also what makes the step insensitive to the factorization's scale ambiguity. The reparameterization $W_Q \to W_Q R,\ W_K \to W_K R^{-T}$ leaves the product $W_{QK}$—and hence the loss—unchanged, yet it would alter the bare SGD step; here the induced update $\Delta W_{QK} = \eta\, \mathcal{P}_{\mathcal{T}}(G_{QK})$ depends on the factors only through their column spaces, which the gauge preserves, so it is invariant by construction. SGD descends on the two matrices; this descends on the bilinear form they define.

Stepping back: by linearizing onto the tangent space and solving for the gauge-invariant step $\Delta W_{QK}$ rather than the redundant factors, the entire $L^2$ derivation collapsed to a handful of $d_k \times d_k$ operations—no ill-posed coupled solve, and no $d \times d$ matrix ever formed.

Frobenius was only the first norm we could have chosen. The geometric formulation—a linear objective minimized over the tangent space under a norm bound—is norm-agnostic; only the inner solve changes. Next we take the norm the bilinear picture really calls for: the sup norm, which (as we saw earlier) is the spectral norm of the form. There the clean projection no longer applies, but the steepest-descent step still does—and it leads us back to Muon.


## 7. The $L^\infty$ Pareto Frontier: A Spectrum of Tractability

If we instead choose the supremum norm to bound the spectral behavior of the composite matrix ($\| \Delta W_{QK} \|_2 \le \epsilon$), the rigid $L^\infty$ geometry shatters into a spectrum of approximations. We can map these from the most mathematically precise (and intractable) down to the most hardware-friendly.

### I. The Exact Convex Formulation

Because the step $\Delta W_Q W_K^T + W_Q \Delta W_K^T$ is strictly linear, bounding its spectral norm defines a perfectly convex set.
While this convex problem can be solved directly (e.g., using a Factorized Alternating Direction Method of Multipliers, or ADMM), iteratively projecting onto the spectral ball still requires repeated Singular Value Decompositions in the inner loop, creating significant computational friction.

### II. The Triangle Inequality Relaxation

To bypass iterative solvers, we relax the bound using the triangle inequality:


$$\| \Delta W_Q W_K^T + W_Q \Delta W_K^T \|_2 \le \| \Delta W_Q W_K^T \|_2 + \| W_Q \Delta W_K^T \|_2 \le \epsilon$$

By allocating an $\epsilon$ budget to each term, the problem decouples cleanly. For the query matrix, the steepest descent maps through the pseudo-inverse:


$$\Delta W_Q \propto - \text{Orth}(G_Q W_K^+) (W_K^T)^+$$

Interestingly, if we try to analytically optimize the allocation of $\epsilon$ between the two matrices by parameterizing it as $t\epsilon$ and $(1-t)\epsilon$, the $L^\infty$ geometry forces a "bang-bang" solution: the optimal path allocates 100% of the budget to the steepest matrix and freezes the other. In practice, a fixed symmetric split avoids this extreme coordinate descent while preserving the mathematically beautiful Riemannian preconditioning of the pseudo-inverse.

### III. The Sub-Multiplicative Relaxation (Cross-Scaled Muon)

The final, most tractable point on the frontier applies the sub-multiplicative property of the spectral norm to the previous bounds:


$$\| \Delta W_Q W_K^T \|_2 \le \| \Delta W_Q \|_2 \| W_K \|_2$$

This completely decouples the norms into scalars. The optimization resolves exactly into standard, independent Muon steps—but with a critical, theoretically justified correction. The step size for each matrix must be inversely proportional to the scalar spectral norm of the *other* matrix:


$$\Delta W_Q \propto - \left( \frac{1}{\| W_K \|_2} \right) \text{Orth}(G_Q)$$

This final relaxation recovers the distributed speed of the standard Muon optimizer while mathematically re-injecting the vital cross-matrix coupling that respects the true bilinear geometry of attention.