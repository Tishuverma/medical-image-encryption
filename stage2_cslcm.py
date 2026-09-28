import math
import numpy as np


# ============================================================
# 1. VALUES GENERATED FROM STAGE 1
# ============================================================
# These are the EXACT n1-n6 values obtained from SHA-256.
# Do NOT generate new values here.

n1 = 0.408828404860659
n2 = 0.984613543227155
n3 = 0.312967540926650
n4 = 0.363484199195558
n5 = 0.686281646560643

# B6 = 54873542
# n6 = B6 / (2^40 - 1)
n6 = 54873542 / ((2 ** 40) - 1)


# ============================================================
# 2. MAP n1-n6 TO 3D CSLCM PARAMETERS
# ============================================================

x0 = n1
y0 = n2
z0 = n3

alpha = n4
beta = n5

# n6 is in [0,1].
# CSLCM requires l in [1.5, 3.8].
l = 1.5 + (3.8 - 1.5) * n6


# ============================================================
# 3. NUMBER OF CHAOTIC VALUES TO GENERATE
# ============================================================
# Change this according to your requirement.
#
# For an image of size M x N:
#
#     SEQUENCE_LENGTH = M * N
#
# For testing, we use 10000 values.

SEQUENCE_LENGTH = 10000


# ============================================================
# 4. TRANSIENT ITERATIONS
# ============================================================
# These initial iterations are discarded to reduce transient
# effects before storing the final chaotic sequences.
#
# This is an implementation choice. The paper describes
# removing transient effects but does not specify one fixed
# universal T in the relevant section.

TRANSIENT_ITERATIONS = 0


# ============================================================
# 5. 3D CSLCM FUNCTION
# ============================================================

def generate_cslcm_sequences(
    x0,
    y0,
    z0,
    alpha,
    beta,
    l,
    sequence_length,
    transient_iterations
):
    """
    Generate the three chaotic sequences produced by
    the 3D Cubic-Sine Logistic Coupled Map (CSLCM).

    Returns:
        x_sequence
        y_sequence
        z_sequence
    """

    # --------------------------------------------------------
    # Current state
    # --------------------------------------------------------

    x = x0
    y = y0
    z = z0

    # --------------------------------------------------------
    # Total iterations
    # --------------------------------------------------------

    total_iterations = (
        transient_iterations + sequence_length
    )

    # --------------------------------------------------------
    # Arrays to store the three sequences
    # --------------------------------------------------------

    x_sequence = np.empty(sequence_length, dtype=np.float64)
    y_sequence = np.empty(sequence_length, dtype=np.float64)
    z_sequence = np.empty(sequence_length, dtype=np.float64)

    sequence_index = 0

    # ========================================================
    # ITERATE CSLCM
    # ========================================================

    for i in range(total_iterations):

        # ----------------------------------------------------
        # 3D CSLCM equations
        # ----------------------------------------------------

        x_next = (
            l * x * (1 - x)
            + beta * math.sin(y) * (z ** 2)
            + alpha * (y ** 3) * math.cos(z)
        )

        y_next = (
            l * y * (1 - y)
            + beta * math.sin(z) * (x ** 2)
            + alpha * (z ** 3) * math.cos(x)
        )

        z_next = (
            l * z * (1 - z)
            + beta * math.sin(x) * (y ** 2)
            + alpha * (x ** 3) * math.cos(y)
        )

        # ----------------------------------------------------
        # Update current state
        # ----------------------------------------------------

        x = x_next
        y = y_next
        z = z_next

        # ----------------------------------------------------
        # Discard transient iterations
        # ----------------------------------------------------

        if i >= transient_iterations:

            x_sequence[sequence_index] = x
            y_sequence[sequence_index] = y
            z_sequence[sequence_index] = z

            sequence_index += 1

    return (
        x_sequence,
        y_sequence,
        z_sequence
    )


# ============================================================
# 6. PRINT PARAMETERS
# ============================================================

print("=" * 70)
print("3D CSLCM KEY / PARAMETER GENERATION")
print("=" * 70)

print("\nOriginal n-values:")

print(f"n1 = {n1:.15f}")
print(f"n2 = {n2:.15f}")
print(f"n3 = {n3:.15f}")
print(f"n4 = {n4:.15f}")
print(f"n5 = {n5:.15f}")
print(f"n6 = {n6:.17f}")


print("\n" + "-" * 70)
print("3D CSLCM PARAMETERS")
print("-" * 70)

print(f"x0    = {x0:.15f}")
print(f"y0    = {y0:.15f}")
print(f"z0    = {z0:.15f}")
print(f"alpha = {alpha:.15f}")
print(f"beta  = {beta:.15f}")
print(f"l     = {l:.15f}")


# ============================================================
# 7. GENERATE THREE CHAOTIC SEQUENCES
# ============================================================

X_sequence, Y_sequence, Z_sequence = generate_cslcm_sequences(
    x0=x0,
    y0=y0,
    z0=z0,
    alpha=alpha,
    beta=beta,
    l=l,
    sequence_length=SEQUENCE_LENGTH,
    transient_iterations=TRANSIENT_ITERATIONS
)


# ============================================================
# 8. VERIFY SEQUENCE LENGTH
# ============================================================

print("\n" + "=" * 70)
print("SEQUENCE INFORMATION")
print("=" * 70)

print(f"\nRequired sequence length = {SEQUENCE_LENGTH}")

print(f"X sequence length = {len(X_sequence)}")
print(f"Y sequence length = {len(Y_sequence)}")
print(f"Z sequence length = {len(Z_sequence)}")


# ============================================================
# 9. PRINT FIRST 10 VALUES OF EACH SEQUENCE
# ============================================================

print("\n" + "=" * 70)
print("FIRST 10 VALUES OF X SEQUENCE")
print("=" * 70)

for i in range(min(10, SEQUENCE_LENGTH)):
    print(f"X[{i}] = {X_sequence[i]:.15f}")


print("\n" + "=" * 70)
print("FIRST 10 VALUES OF Y SEQUENCE")
print("=" * 70)

for i in range(min(10, SEQUENCE_LENGTH)):
    print(f"Y[{i}] = {Y_sequence[i]:.15f}")


print("\n" + "=" * 70)
print("FIRST 10 VALUES OF Z SEQUENCE")
print("=" * 70)

for i in range(min(10, SEQUENCE_LENGTH)):
    print(f"Z[{i}] = {Z_sequence[i]:.15f}")


# ============================================================
# 10. SAVE THE THREE SEQUENCES
# ============================================================
# They are saved in NumPy .npy format so that they can be
# directly loaded later for chaotic-property analysis or
# image encryption.

np.save("cslcm_x_sequence.npy", X_sequence)
np.save("cslcm_y_sequence.npy", Y_sequence)
np.save("cslcm_z_sequence.npy", Z_sequence)


# ============================================================
# 11. ALSO SAVE ALL THREE IN ONE FILE
# ============================================================

np.savez(
    "cslcm_sequences.npz",
    X=X_sequence,
    Y=Y_sequence,
    Z=Z_sequence
)


# ============================================================
# 12. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("CSLCM GENERATION COMPLETE")
print("=" * 70)

print("\nThree chaotic sequences generated:")

print("1. X sequence")
print("2. Y sequence")
print("3. Z sequence")

print("\nFiles created:")

print("cslcm_x_sequence.npy")
print("cslcm_y_sequence.npy")
print("cslcm_z_sequence.npy")
print("cslcm_sequences.npz")

print("\n" + "=" * 70)