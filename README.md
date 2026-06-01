# Eigen Solver for Symmetric & Hermitian Matrices

A focused Python utility designed to compute the **eigenvalues** and **eigenvectors** of real symmetric or Hermitian matrices efficiently. Powered by NumPy's optimized numerical routines, this script ensures high precision and performance for structural engineering, quantum mechanics, and machine learning workflows (like PCA).

## 🧠 Matrix Theory Behind `np.linalg.eigh`

Unlike standard solvers that handle any arbitrary square matrix, this implementation utilizes `np.linalg.eigh`. 

* **Why?** For **Symmetric matrices** ($M = M^T$) or **Hermitian matrices** ($M = M^\dagger$), the eigenvalues are guaranteed to be **purely real numbers**, and the eigenvectors are **orthogonal**. 
* **The Benefit:** `np.linalg.eigh` uses specialized algorithms (like LAPACK routines) which are significantly faster, more mathematically stable, and automatically sort the eigenvalues in **ascending order**.

---

## 🚀 Features

* Fully optimized using **NumPy (`linalg.eigh`)**.
* Guarantees sorted eigenvalues ($λ_1 \le λ_2 \le \dots \le λ_n$).
* Column-bound eigenvector outputs (Each column corresponds to a specific eigenvalue).
* Handles both real symmetric and complex Hermitian systems safely.

---

## 💻 Code & Implementation

### Core Function (`eigen_solver.py`)

```python
import numpy as np

def eigen_solve(M):
    """
    Returns (eigenvalues, eigenvectors) of a real symmetric
    or Hermitian matrix M using numpy.linalg.eigh.
    
    Parameters:
    M (numpy.ndarray): A square symmetric or Hermitian matrix.
    
    Returns:
    eigenvalues (ndarray): Sorted eigenvalues in ascending order.
    eigenvectors (ndarray): The normalized eigenvectors, where column v[:, i] 
                            is the eigenvector corresponding to eigenvalue[i].
    """
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    return eigenvalues, eigenvectors
