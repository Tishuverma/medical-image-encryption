import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SHA-256 DERIVED PARAMETERS
# ============================================================

n1 = 0.408828404860659
n2 = 0.984613543227155
n3 = 0.312967540926650
n4 = 0.363484199195558
n5 = 0.686281646560643
n6 = 0.00004990719571655964

x0 = n1
y0 = n2
z0 = n3

alpha = n4
beta = n5

actual_l = 1.5 + (3.8 - 1.5) * n6

print("Actual SHA-derived parameters:")
print("x0    =", x0)
print("y0    =", y0)
print("z0    =", z0)
print("alpha =", alpha)
print("beta  =", beta)
print("l     =", actual_l)


# ============================================================
# 3D CSLCM
# ============================================================

def cslcm_step(x, y, z, l):

    x_new = (
        l * x * (1 - x)
        + beta * np.sin(y) * z**2
        + alpha * y**3 * np.cos(z)
    )

    y_new = (
        l * y * (1 - y)
        + beta * np.sin(z) * x**2
        + alpha * z**3 * np.cos(x)
    )

    z_new = (
        l * z * (1 - z)
        + beta * np.sin(x) * y**2
        + alpha * x**3 * np.cos(y)
    )

    return x_new, y_new, z_new


# ============================================================
# JACOBIAN
# ============================================================

def jacobian(x, y, z, l):

    J = np.array([

        [
            l * (1 - 2*x),
            beta * np.cos(y) * z**2
            + 3 * alpha * y**2 * np.cos(z),
            2 * beta * np.sin(y) * z
            - alpha * y**3 * np.sin(z)
        ],

        [
            2 * beta * np.sin(z) * x
            - alpha * z**3 * np.sin(x),
            l * (1 - 2*y),
            beta * np.cos(z) * x**2
            + 3 * alpha * z**2 * np.cos(x)
        ],

        [
            beta * np.cos(x) * y**2
            + 3 * alpha * x**2 * np.cos(y),
            2 * beta * np.sin(x) * y
            - alpha * x**3 * np.sin(y),
            l * (1 - 2*z)
        ]

    ])

    return J


# ============================================================
# LARGEST LYAPUNOV EXPONENT
# ============================================================

def largest_lyapunov(l, transient=500, iterations=2000):

    x = x0
    y = y0
    z = z0

    # Remove transient behavior
    for _ in range(transient):

        x, y, z = cslcm_step(x, y, z, l)

        if not np.all(np.isfinite([x, y, z])):
            return np.nan

    # Initial perturbation
    v = np.array([1.0, 0.0, 0.0])

    total = 0.0

    for _ in range(iterations):

        # Stop safely if the state becomes invalid
        if not np.all(np.isfinite([x, y, z])):
            return np.nan

        J = jacobian(x, y, z, l)

        v = J @ v

        norm = np.linalg.norm(v)

        if not np.isfinite(norm) or norm == 0:
            return np.nan

        total += np.log(norm)

        v = v / norm

        x, y, z = cslcm_step(x, y, z, l)

    return total / iterations


# ============================================================
# LYAPUNOV SWEEP
# ============================================================

l_values = np.linspace(2.00, 2.02, 1000)

lyapunov_values = []

print("\nCalculating Lyapunov exponent sweep...")

for l in l_values:

    le1 = largest_lyapunov(l)

    lyapunov_values.append(le1)


lyapunov_values = np.array(lyapunov_values)


# ============================================================
# FIND MAXIMUM CHAOS
# ============================================================

valid = np.isfinite(lyapunov_values)

if np.any(valid):

    valid_indices = np.where(valid)[0]

    max_index = valid_indices[
        np.argmax(lyapunov_values[valid])
    ]

    maximum_l = l_values[max_index]
    maximum_le1 = lyapunov_values[max_index]

    print("\nMaximum LE1 found in tested interval:")
    print("l   =", maximum_l)
    print("LE1 =", maximum_le1)

else:

    print("\nNo valid Lyapunov values were obtained.")


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    l_values,
    lyapunov_values,
    linewidth=1
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.axvline(
    actual_l,
    linestyle="--",
    linewidth=1,
    label="Actual SHA-derived l"
)

if np.any(valid):

    plt.scatter(
        maximum_l,
        maximum_le1,
        s=50,
        label="Maximum LE1"
    )

plt.xlabel("Control parameter l")
plt.ylabel("Largest Lyapunov Exponent (LE1)")
plt.title("Largest Lyapunov Exponent vs Control Parameter")
plt.legend()
plt.grid(True)

plt.savefig(
    "cslcm_lyapunov_sweep.png",
    dpi=300,
    bbox_inches="tight"
)
plt.xlim(1.5, 3.8)
plt.show()

# ============================================================
# CHECK LE1 AT l = 2.0
# ============================================================

test_l = 2.0

test_le1 = largest_lyapunov(test_l)

print("\nLyapunov exponent at l = 2.0:")
print("l   =", test_l)
print("LE1 =", test_le1)

# ============================================================
# CHECK ACTUAL SHA-DERIVED l
# ============================================================

actual_le1 = largest_lyapunov(actual_l)

print("\nLyapunov exponent at actual SHA-derived l:")
print("l   =", actual_l)
print("LE1 =", actual_le1)