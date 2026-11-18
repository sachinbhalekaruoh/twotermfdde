import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


def solve_quartic_w(a, b, c, alpha):
    """
    Solves equation (13) for w = v^alpha > 0:
    c^2 * w^4 + 2*c*cos(alpha*pi/2)*w^3 + (1 - 2*a*c*cos(alpha*pi))*w^2
    - 2*a*cos(alpha*pi/2)*w + (a^2 - b^2) = 0
    Returns sorted positive real roots w = v^alpha.
    """
    c_half = np.cos(alpha * np.pi / 2.0)
    c_full = np.cos(alpha * np.pi)
    coeffs = [
        c**2,
        2.0 * c * c_half,
        1.0 - 2.0 * a * c * c_full,
        -2.0 * a * c_half,
        a**2 - b**2,
    ]
    roots = np.roots(coeffs)
    pos_roots = []
    for r in roots:
        if abs(r.imag) < 1e-7 and r.real > 1e-9:
            w = r.real
            lhs = (w * c_half + c * (w**2) * c_full - a) ** 2 + (
                w * np.sin(alpha * np.pi / 2.0) + c * (w**2) * np.sin(alpha * np.pi)
            ) ** 2
            if abs(lhs - b**2) < 1e-4 * (1.0 + b**2):
                pos_roots.append(w)
    pos_roots = sorted(pos_roots)
    unique_roots = []
    for w in pos_roots:
        if not unique_roots or abs(w - unique_roots[-1]) > 1e-5:
            unique_roots.append(w)
    return unique_roots


def get_critical_tau(w, a, b, c, alpha, n=0):
    """
    Computes valid critical delay tau+(n) or tau-(n) using equations (11), (12), (14), (15).
    """
    v = w ** (1.0 / alpha)
    cos_val = (w * np.cos(alpha * np.pi / 2.0) + c * (w**2) * np.cos(alpha * np.pi) - a) / b
    sin_val = -(w * np.sin(alpha * np.pi / 2.0) + c * (w**2) * np.sin(alpha * np.pi)) / b
    cos_val = np.clip(cos_val, -1.0, 1.0)
    theta = np.arccos(cos_val)
    if sin_val >= -1e-7:
        return (2.0 * n * np.pi + theta) / v
    n_eff = max(1, n) if n == 0 else n
    return (2.0 * n_eff * np.pi - theta) / v


def compute_tangent_curve(b, alpha, c_vals, branch="minus", sign_a1="pos"):
    """
    Computes exact common tangent curves Gamma_5 through Gamma_10 directly from
    the parametric boundary dP (Eqs. 23-25), avoiding inner loop tangencies and
    reproducing the exact benchmark values in Section 8.
    """
    S = np.sin(alpha * np.pi / 2.0)
    C = np.cos(alpha * np.pi / 2.0)
    sin_ap = np.sin(alpha * np.pi)
    cos_ap = np.cos(alpha * np.pi)
    csc_ap = 1.0 / sin_ap
    sign_Q = -1.0 if branch == "minus" else 1.0

    pts_a1, pts_c = [], []

    for c in c_vals:
        if abs(c) < 1e-6:
            continue

        def tangency_residual(phi):
            x = np.cos(phi)
            y = np.sin(phi)
            delta = S**2 + 4.0 * b * c * y * sin_ap
            if delta <= 1e-12:
                return np.nan
            Q = np.sqrt(delta)
            w = csc_ap * (-S + sign_Q * Q) / (2.0 * c)
            if w <= 1e-9:
                return np.nan
            denom_w = 2.0 * c * sin_ap * w + S
            if abs(denom_w) < 1e-10:
                return np.nan
            dw_dy = b / denom_w
            dx_dy = ((C + 2.0 * c * cos_ap * w) * dw_dy) / b
            return x * dx_dy + y

        phis = np.linspace(-np.pi + 1e-4, np.pi - 1e-4, 600)
        vals = [tangency_residual(p) for p in phis]
        cands = []
        for i in range(len(phis) - 1):
            if np.isnan(vals[i]) or np.isnan(vals[i + 1]):
                continue
            if vals[i] * vals[i + 1] <= 0 and abs(vals[i] - vals[i + 1]) < 5.0:
                try:
                    p_root = brentq(tangency_residual, phis[i], phis[i + 1])
                    x = np.cos(p_root)
                    y = np.sin(p_root)
                    delta = S**2 + 4.0 * b * c * y * sin_ap
                    Q = np.sqrt(max(0.0, delta))
                    w = csc_ap * (-S + sign_Q * Q) / (2.0 * c)
                    if w <= 1e-9:
                        continue
                    a = C * w + c * cos_ap * (w**2) - b * x
                    a1 = a + b
                    cands.append(a1)
                except ValueError:
                    pass

        if not cands:
            continue

        if sign_a1 == "pos":
            # Gamma_5, Gamma_6, Gamma_8, Gamma_10 (c < 0, a1 > -1/(4c))
            valid = [val for val in cands if val > max(0.0, -0.25 / c)]
            if valid:
                pts_a1.append(max(valid))
                pts_c.append(c)
        else:
            # Gamma_7 (b > 0, c > 0, a1 <= 0) and Gamma_9 (b < 0, c > 0, a1 <= 2b)
            if b > 0:
                valid = [val for val in cands if val <= 0.0]
            else:
                valid = [val for val in cands if val <= 2.0 * b + 1e-4]
            if valid:
                pts_a1.append(min(valid))
                pts_c.append(c)

    if pts_c:
        idx = np.argsort(pts_c)
        return np.array(pts_a1)[idx], np.array(pts_c)[idx]
    return np.array([]), np.array([])


def compute_switch_curve(b, alpha, c_vals, a1_bounds_func, mode="SS"):
    """
    Computes delay-dependent bifurcation curves Gamma_11 through Gamma_18
    by solving the matching critical delay equations from Section 7.
    """
    pts_a1, pts_c = [], []
    for c in c_vals:
        a1_min, a1_max = a1_bounds_func(c)
        if a1_min >= a1_max:
            continue

        def residual(a1):
            a = a1 - b
            roots = solve_quartic_w(a, b, c, alpha)
            if len(roots) < 2:
                return np.nan
            w1, w2 = roots[:2]
            v2 = w2 ** (1.0 / alpha)
            tau1 = get_critical_tau(w1, a, b, c, alpha, n=0)
            tau2 = get_critical_tau(w2, a, b, c, alpha, n=0)
            if mode == "SS":
                return (tau1 - tau2) - (2.0 * np.pi / v2)
            return tau1 - tau2

        # Cluster sample points near the outer boundary where SS/SSR gap is narrow
        t = np.linspace(0.0, 1.0, 140)
        grid = a1_min + (a1_max - a1_min) * (1.0 - (1.0 - t) ** 2)
        res_vals = [residual(val) for val in grid]
        for i in range(len(grid) - 1):
            if np.isnan(res_vals[i]) or np.isnan(res_vals[i + 1]):
                continue
            if res_vals[i] * res_vals[i + 1] <= 0:
                try:
                    a1_root = brentq(residual, grid[i], grid[i + 1])
                    pts_a1.append(a1_root)
                    pts_c.append(c)
                    break
                except ValueError:
                    pass
    return np.array(pts_a1), np.array(pts_c)


def plot_bifurcation_diagram(alpha, b, a1_lim=(-4, 4), c_lim=(-4, 4)):
    """
    Generates the stability and bifurcation diagram in the (a1, c)-plane
    corresponding to Figures 17, 18, 19, or 20 in the paper.
    """
    if not (0 < alpha < 1) or b == 0:
        raise ValueError("Require 0 < alpha < 1 and b != 0.")

    a1_min, a1_max = a1_lim
    c_min, c_max = c_lim

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axhline(0, color="black", linewidth=1.2)
    ax.axvline(0, color="black", linewidth=1.2)

    c_pos = np.linspace(0.05, c_max, 240)
    c_neg = np.linspace(c_min, -0.05, 240)

    # 1. Non-delayed boundaries Gamma_0, Gamma_1, Gamma_2
    c_g1_pos = np.linspace(0.06, c_max, 250)
    ax.plot(-0.25 / c_g1_pos, c_g1_pos, "k--", linewidth=1.2, label=r"$\Gamma_1: a_1 c = -1/4$")
    c_g1_neg = np.linspace(c_min, -0.06, 250)
    ax.plot(-0.25 / c_g1_neg, c_g1_neg, "k--", linewidth=1.2)

    k2 = -(np.tan(alpha * np.pi / 2.0) ** 2 + 1.0) / 4.0
    ax.plot(k2 / c_g1_neg, c_g1_neg, color="purple", linewidth=2.0, label=r"$\Gamma_2$")

    if b < 0:
        ax.plot(-0.25 / c_g1_neg + b, c_g1_neg, color="brown", linestyle=":", linewidth=1.5, label=r"$\Gamma_0$")
        ax.axvline(2 * b, color="gray", linestyle="-.", linewidth=1.2, label=r"$a_1 = 2b$")

    # 2. Case-specific Tangent (Gamma_5 - Gamma_10) and Delay-Switch (Gamma_11 - Gamma_18) curves
    if b > 0 and alpha < 0.5:
        title = f"Figure 17 Regime: $b = {b} > 0$, $\\alpha = {alpha} \\in (0, 1/2)$"
        a1_5, c_5 = compute_tangent_curve(b, alpha, c_neg, branch="minus", sign_a1="pos")
        ax.plot(a1_5, c_5, color="crimson", linewidth=2.0, label=r"$\Gamma_5$ (Tangent)")

        def bound_11(c):
            hi = np.interp(c, c_5, a1_5) - 1e-5 if len(c_5) > 1 else min(a1_max, -0.25 / c + 2 * b)
            return (k2 / c + 1e-4, hi)

        a1_11, c_11 = compute_switch_curve(b, alpha, np.linspace(c_min, -0.08, 90), bound_11, mode="SS")
        ax.plot(a1_11, c_11, color="royalblue", linewidth=1.8, label=r"$\Gamma_{11}$ (SS/SSR)")

    elif b < 0 and alpha < 0.5:
        title = f"Figure 18 Regime: $b = {b} < 0$, $\\alpha = {alpha} \\in (0, 1/2)$"
        a1_6, c_6 = compute_tangent_curve(b, alpha, c_neg, branch="minus", sign_a1="pos")
        ax.plot(a1_6, c_6, color="crimson", linewidth=2.0, label=r"$\Gamma_6$ (Tangent)")

        def bound_12(c):
            hi = np.interp(c, c_6, a1_6) - 1e-5 if len(c_6) > 1 else min(a1_max, k2 / c + 1.5)
            return (k2 / c + 1e-4, hi)

        a1_12, c_12 = compute_switch_curve(b, alpha, np.linspace(c_min, -0.08, 90), bound_12, mode="SS")
        ax.plot(a1_12, c_12, color="royalblue", linewidth=1.8, label=r"$\Gamma_{12}$ (SS/SSR)")

        a1_13, c_13 = compute_switch_curve(
            b, alpha, np.linspace(c_min, -0.08, 90),
            lambda c: (max(1e-4, -0.25 / c + b), k2 / c - 1e-4), mode="IS"
        )
        ax.plot(a1_13, c_13, color="darkorange", linewidth=1.8, label=r"$\Gamma_{13}$ (IS/U)")

    elif b > 0 and alpha > 0.5:
        title = f"Figure 19 Regime: $b = {b} > 0$, $\\alpha = {alpha} \\in (1/2, 1)$"
        a1_7, c_7 = compute_tangent_curve(b, alpha, c_pos, branch="plus", sign_a1="neg")
        ax.plot(a1_7, c_7, color="crimson", linewidth=2.0, label=r"$\Gamma_7$ (Tangent)")
        a1_8, c_8 = compute_tangent_curve(b, alpha, c_neg, branch="minus", sign_a1="pos")
        ax.plot(a1_8, c_8, color="darkred", linewidth=2.0, label=r"$\Gamma_8$ (Tangent)")

        def bound_14(c):
            lo = np.interp(c, c_7, a1_7) + 1e-5 if (len(c_7) > 1 and c >= c_7[0]) else a1_min
            return (lo, -1e-4)

        a1_14, c_14 = compute_switch_curve(b, alpha, np.linspace(0.2, c_max, 90), bound_14, mode="SS")
        ax.plot(a1_14, c_14, color="royalblue", linewidth=1.8, label=r"$\Gamma_{14}$ (SS/SSR)")

        def bound_15(c):
            hi = np.interp(c, c_8, a1_8) - 1e-5 if len(c_8) > 1 else min(a1_max, k2 / c + 3.0)
            return (k2 / c + 1e-4, hi)

        a1_15, c_15 = compute_switch_curve(b, alpha, np.linspace(c_min, -0.08, 90), bound_15, mode="SS")
        ax.plot(a1_15, c_15, color="teal", linewidth=1.8, label=r"$\Gamma_{15}$ (SS/SSR)")

    else:
        title = f"Figure 20 Regime: $b = {b} < 0$, $\\alpha = {alpha} \\in (1/2, 1)$"
        a1_9, c_9 = compute_tangent_curve(b, alpha, c_pos, branch="plus", sign_a1="neg")
        ax.plot(a1_9, c_9, color="crimson", linewidth=2.0, label=r"$\Gamma_9$ (Tangent)")
        a1_10, c_10 = compute_tangent_curve(b, alpha, c_neg, branch="minus", sign_a1="pos")
        ax.plot(a1_10, c_10, color="darkred", linewidth=2.0, label=r"$\Gamma_{10}$ (Tangent)")

        def bound_16(c):
            lo = np.interp(c, c_9, a1_9) + 1e-5 if (len(c_9) > 1 and c >= c_9[0]) else a1_min
            return (lo, -1e-4)

        a1_16, c_16 = compute_switch_curve(b, alpha, np.linspace(0.2, c_max, 90), bound_16, mode="SS")
        ax.plot(a1_16, c_16, color="royalblue", linewidth=1.8, label=r"$\Gamma_{16}$ (SS/SSR)")

        def bound_17(c):
            hi = np.interp(c, c_10, a1_10) - 1e-5 if len(c_10) > 1 else min(a1_max, k2 / c + 2.5)
            return (k2 / c + 1e-4, hi)

        a1_17, c_17 = compute_switch_curve(b, alpha, np.linspace(c_min, -0.08, 90), bound_17, mode="SS")
        ax.plot(a1_17, c_17, color="teal", linewidth=1.8, label=r"$\Gamma_{17}$ (SS/SSR)")

        a1_18, c_18 = compute_switch_curve(
            b, alpha, np.linspace(c_min, -0.08, 90),
            lambda c: (1e-4, k2 / c - 1e-4), mode="IS"
        )
        ax.plot(a1_18, c_18, color="darkorange", linewidth=1.8, label=r"$\Gamma_{18}$ (IS/U)")

    ax.set_xlim(a1_lim)
    ax.set_ylim(c_lim)
    ax.set_xlabel(r"$a_1 = a + b$", fontsize=12)
    ax.set_ylabel(r"$c$", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()


def main():
    # Figure 17: b > 0, 0 < alpha < 1/2
   # plot_bifurcation_diagram(alpha=0.3, b=1.0, a1_lim=(-4, 4), c_lim=(-4, 4))

    # Figure 18: b < 0, 0 < alpha < 1/2
    # plot_bifurcation_diagram(alpha=0.45, b=-1.0, a1_lim=(-4, 4), c_lim=(-4, 4))

    # Figure 19: b > 0, 1/2 < alpha < 1
    # plot_bifurcation_diagram(alpha=0.8, b=1.0, a1_lim=(-4, 12), c_lim=(-4, 6))

    # Figure 20: b < 0, 1/2 < alpha < 1
     plot_bifurcation_diagram(alpha=0.8, b=-1.0, a1_lim=(-4, 6), c_lim=(-4, 10))


if __name__ == "__main__":
    main()
