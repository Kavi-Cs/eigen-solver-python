def eigen_solve(M):
    """
    Returns (eigenvalues, eigenvectors) of a real symmetric
    or Hermitian matrix M using numpy.linalg.eigh.
    Eigenvalues are returned in ascending order.
    Each column of the returned array is an eigenvector.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    return eigenvalues, eigenvectors

print("eigen_solve() function defined.")
