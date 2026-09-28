import numpy as np

# Load CSLCM sequences generated in Stage 2
X = np.load("cslcm_x_sequence.npy")
Y = np.load("cslcm_y_sequence.npy")
Z = np.load("cslcm_z_sequence.npy")

# Check that all three sequences have the same length
if not (len(X) == len(Y) == len(Z)):
    raise ValueError("X, Y and Z must have the same length")

# -------------------------------------------------
# Combine X, Y, Z by reversible interleaving
# K = [X1, Y1, Z1, X2, Y2, Z2, ...]
# -------------------------------------------------

K = np.empty(3 * len(X), dtype=np.float64)

K[0::3] = X
K[1::3] = Y
K[2::3] = Z

# Save final single sequence
np.save("final_sequence.npy", K)

print("X length:", len(X))
print("Y length:", len(Y))
print("Z length:", len(Z))
print("Final sequence K length:", len(K))

print("\nFirst 12 values of K:")
print(K[:12])

# -------------------------------------------------
# Verify exact reversibility
# -------------------------------------------------

X_recovered = K[0::3]
Y_recovered = K[1::3]
Z_recovered = K[2::3]

if (
    np.array_equal(X, X_recovered)
    and np.array_equal(Y, Y_recovered)
    and np.array_equal(Z, Z_recovered)
):
    print("\nREVERSIBILITY CHECK: PASSED")
else:
    print("\nREVERSIBILITY CHECK: FAILED")