import argparse
import pandas as pd

def find_alpha(gamma, total_accepted, tol=1e-8, max_iter=100):
    """
    Solve for alpha in (0,1) such that
        (1 - alpha^(gamma+1)) / (1 - alpha) == gamma * total_accepted
    using the bisection method.
    """
    def f(alpha):
        return (1 - alpha**(gamma+1)) / (1 - alpha) - 1 - gamma * total_accepted

    low, high = 0.0, 1.0 - 1e-15
    f_low, f_high = f(low), f(high)
    if f_low * f_high > 0:
        raise ValueError(
            f"No root bracketed in (0,1): f(0)={f_low}, f(1-)={f_high}"
        )

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = f(mid)
        if abs(f_mid) < tol or (high - low)/2 < tol:
            return mid
        if f_low * f_mid <= 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid

    return 0.5 * (low + high)

def main():
    parser = argparse.ArgumentParser(
        description="Read CSV and compute alpha for each row"
    )
    parser.add_argument(
        "--csv", type=str, required=True,
        help="Path to input CSV with columns prefix_len,draft_budget,gamma,task,accept_rate_total"
    )
    args = parser.parse_args()

    # Load the data
    df = pd.read_csv(args.csv)

    # Compute and print alpha for each row
    print("prefix_len,draft_budget,gamma,accept_rate_total,alpha")
    for _, row in df.iterrows():
        gamma = int(row["gamma"])
        total_accepted = float(row["accept_rate_total"])
        alpha = find_alpha(gamma, total_accepted)
        print(f"{int(row['prefix_len'])},{int(row['draft_budget'])},{gamma},{total_accepted:.4f},{alpha:.6f}")

if __name__ == "__main__":
    main()
