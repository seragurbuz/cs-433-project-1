import numpy as np


def _mse_loss(y, tx, w):
    """Compute the course's MSE convention: half the mean squared error."""
    error = y - tx.dot(w)
    return 0.5 * np.mean(error**2)


def _sigmoid(scores):
    """Compute sigmoid without overflowing for large negative scores."""
    return np.exp(-np.logaddexp(0, -scores))


def _logistic_loss(y, tx, w):
    """Compute average binary negative log likelihood for labels in {0, 1}."""
    scores = tx.dot(w)
    # logaddexp evaluates log(1 + exp(scores)) without overflowing.
    return np.mean(np.logaddexp(0, scores) - y * scores)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Fit linear regression with full-batch gradient descent.

    Starting from initial_w, take max_iters steps of size gamma. Return
    (w, loss), where loss is 0.5 * mean((y - tx @ w)**2) at the final w.
    """
    w = initial_w.copy()
    for _ in range(max_iters):
        error = y - tx.dot(w)
        gradient = -tx.T.dot(error) / len(y)
        w = w - gamma * gradient

    return w, _mse_loss(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Fit linear regression with stochastic gradient descent, batch size 1.

    Each of max_iters updates samples one row uniformly with replacement
    and takes a step of size gamma. Return (w, loss), evaluating the final
    MSE on all rows. Use np.random.seed before calling for reproducibility.
    """
    w = initial_w.copy()
    for _ in range(max_iters):
        # Equivalent to the exercise's batch_iter with batch_size=1.
        index = np.random.randint(len(y))
        error = y[index] - tx[index].dot(w)
        gradient = -tx[index] * error
        w = w - gamma * gradient

    return w, _mse_loss(y, tx, w)


def least_squares(y, tx):
    """Solve the least-squares normal equations and return (w, MSE).

    The MSE includes the course's factor of 0.5. As in the exercise,
    tx.T @ tx must be nonsingular for np.linalg.solve.
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, _mse_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Minimize MSE + lambda_ * ||w||**2 using the normal equations.

    Regularize every coefficient, including any intercept. Return (w, MSE)
    with the penalty excluded from the reported loss.
    """
    regularizer = 2 * len(y) * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + regularizer
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, _mse_loss(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Fit logistic regression by gradient descent for labels in {0, 1}.

    Starting from initial_w, take max_iters steps of size gamma on the
    average negative log likelihood. Return (w, loss) at the final w.
    """
    w = initial_w.copy()
    for _ in range(max_iters):
        predictions = _sigmoid(tx.dot(w))
        gradient = tx.T.dot(predictions - y) / len(y)
        w = w - gamma * gradient

    return w, _logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Fit logistic regression with an L2 penalty for labels in {0, 1}.

    Take max_iters steps of size gamma from initial_w, minimizing average
    negative log likelihood + lambda_ * ||w||**2. Regularize all weights,
    including any intercept. Return (w, loss), excluding the penalty.
    """
    w = initial_w.copy()
    for _ in range(max_iters):
        predictions = _sigmoid(tx.dot(w))
        gradient = tx.T.dot(predictions - y) / len(y) + 2 * lambda_ * w
        w = w - gamma * gradient

    return w, _logistic_loss(y, tx, w)
