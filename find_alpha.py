def find_alpha(gamma, accept_rate_total, tol=1e-8, max_iter=100):
    """
    Solve for alpha in (0,1) such that
        (1 - alpha^(gamma+1)) / (1 - alpha) == gamma * accept_rate_total
    using the bisection method.
    """
    def f(alpha):
        # avoid division by zero at alpha=1
        return (1 - alpha**(gamma+1)) / (1 - alpha) -1 - gamma * accept_rate_total

    # initial bracket [low, high]
    low, high = 0.0, 1.0 - 1e-15
    f_low, f_high = f(low), f(high)

    if f_low * f_high > 0:
        raise ValueError(
            "f(0) and f(1) have the same sign; no guaranteed root in (0,1). "
            f"f(0)={f_low}, f(1-)={f_high}"
        )

    for i in range(max_iter):
        mid = (low + high) / 2
        f_mid = f(mid)

        # Check for convergence
        if abs(f_mid) < tol or (high - low)/2 < tol:
            return mid

        # Narrow the bracket
        if f_low * f_mid <= 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid

    # return best estimate after max_iter
    return (low + high) / 2
gamma = 16
accept_rate_total = 0.4558
accept_rate_per_token = find_alpha(gamma, accept_rate_total)
print(f"accept_rate_per_token: {accept_rate_per_token}")