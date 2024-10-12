
```python
def solve_x_S(A: np.ndarray, y: np.ndarray, S: list[int]) -> np.ndarray:
    """
    Returns an $x_S$ that minimizes $\| A_S x_S - y \|^2$
    """
    A_S = A[:, S]
    return np.linalg.solve(A_S.H @ A_S, A_S.H @ y)

def project_y_to_range_A_S(A: np.ndarray, y: np.ndarray, S: list[int]) -> np.ndarray::
    """
    Returns $P_{A_S} y$
    """
    A_S = A[:, S]
    return A_S @ solve_x(A: np.ndarray, y: np.ndarray, S: list[int]) -> np.ndarray:

def score_S(A: np.ndarray, y: np.ndarray, S: list[int]) -> float:
    """
    Returns $y^H P_{A_S} y$
    """
    return np.real(y.H @ project_to_range(A, y, S))  # real is needed only because of numerical errors
```

## Matching Pursuit
and at each iteration we increase the support by one and update $y$.

and at each iteration we 

 and the original $y$. Then, at each iteration:

The algorithm is based on the following observation:
When $k=1$, the problem is equivalent to finding the column of $A$ that is most correlated with $y$:
\begin{align}
y^H P_{a_i} y &= y^H a_i (a_i^H a_i)^{-1} a_i^H y 
\\&=
\frac{| a_i^H y |^2}{\| a_i \|^2}.
\end{align}


MP is a greedy iterative algorithm. At each iteration we increase the support by one and update $y$.
We start with an empty support and the original $y$. Then, at each iteration:

1. Add the column $i$ that maximizes $y^H P_{a_i} y$ to the support.

2. Update $y$ by projecting it onto the space orthogonal to $a_i$. 

In non efficient Python, it looks like this:
```python
def solve_x_S(A, y, S):
        A_S = A[:, S]
        return np.linalg.solve(A_S.H @ A_S, A_S.H @ y)

def project_to_range_A_S(A, y, S):
    A_S = A[:, S]
    return A_S @ solve_x(A, y, S)

def score_S(A, y, S):
    return y.H @ project_to_range(A, y, S)

def mp(A: np.ndarray, y: np.ndarray, k: int) -> list[int]:
    S = []
    while len(S) < k:
        i = max(range(n), key=lambda i: score(A, y, [i]))
        S.add(i)
        y -= project_to_range(A, y, S)
```


def mp(A: np.ndarray, y: np.ndarray, k: int) -> list[int]:
    S = []
    A = A / np.linalg.norm(A, axis=0) # normalize columns so that the correlation is the same as the inner product
    m, n = A.shape

        

    def correlation(i):
        return np.abs(A[:, i].H @ y) / np.linalg.norm(A[:, i])

    while len(s) < k:
        i = max(range(n), key=correlation)
        S.add(i)

        for i in range(n):
            x[i] = np.abs(A[:, i].H @ y) ** 2 / np.linalg.norm(A[:, i]) ** 2

        x = A.H @ y
        i = np.argmax(np.abs(x))
        S.add(i)
        y -= A[:, i] * x[i]  # A[:, i] * x[i] is the projection of y onto the column a_i, so this is the projection of y onto the space orthogonal to a_i
```


<!-- (Or, equivalently, subtracts $a_i x_i$ from $y$, where $a_i$ is the selected column and $x_i$ is the coefficient that minimizes $\| a_i x_i - y \|^2$). -->

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
