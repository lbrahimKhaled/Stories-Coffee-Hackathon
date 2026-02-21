import numpy as np
import numpy.linalg as la


def standardize_matrix(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (X - mu) / sigma, mu, sigma


def ridge_fit(X, y, alpha=1.0):
    X1 = np.c_[np.ones(len(X)), X]
    I = np.eye(X1.shape[1])
    I[0, 0] = 0.0
    return la.solve(X1.T @ X1 + alpha * I, X1.T @ y)


def ridge_predict(X, beta):
    X1 = np.c_[np.ones(len(X)), X]
    return X1 @ beta


def loocv_rmse(X, y, alpha=1.0):
    if len(y) < 2:
        return np.nan

    preds = []
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        beta = ridge_fit(X[mask], y[mask], alpha=alpha)
        preds.append(ridge_predict(X[i : i + 1], beta)[0])

    preds = np.array(preds)
    return float(np.sqrt(np.mean((preds - y) ** 2)))


def fit_ridge_with_loocv(X, y, alphas):
    if len(y) == 0:
        raise ValueError("Cannot train ridge model with zero samples.")

    if len(y) < 2:
        alpha = float(alphas[0])
        return ridge_fit(X, y, alpha=alpha), alpha, np.nan

    rmses = [loocv_rmse(X, y, alpha=a) for a in alphas]
    best_idx = int(np.nanargmin(rmses))
    best_alpha = float(alphas[best_idx])
    beta = ridge_fit(X, y, alpha=best_alpha)
    return beta, best_alpha, float(rmses[best_idx])
