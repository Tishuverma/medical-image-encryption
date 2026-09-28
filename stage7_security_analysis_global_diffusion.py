import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = "/Users/tishuverma/Desktop/Image"

ORIGINAL_FILE = os.path.join(
    BASE_DIR,
    "chest x ray.jpeg"
)

ENCRYPTED_FILE = os.path.join(
    BASE_DIR,
    "encrypted_medical_image.png"
)

K_FILE = os.path.join(
    BASE_DIR,
    "final_sequence.npy"
)

GENERATOR_FILE = os.path.join(
    BASE_DIR,
    "stage5_gan_generator.pth"
)

RESULT_FILE = os.path.join(
    BASE_DIR,
    "stage7_security_results.txt"
)


# ============================================================
# 2. PARAMETERS
# ============================================================

BLOCK_SIZE = 256
BLOCK_PIXELS = BLOCK_SIZE * BLOCK_SIZE
LATENT_SIZE = 128


# ============================================================
# 3. DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# ============================================================
# 4. GENERATOR
#
# EXACTLY THE SAME ARCHITECTURE AS STAGE 5
# ============================================================

class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.key_encoder = nn.Sequential(
            nn.Linear(30000, 1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, LATENT_SIZE),
            nn.LeakyReLU(0.2),
        )

        self.block_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(0.2),

            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(
                LATENT_SIZE + 32,
                4 * 4 * 256
            ),
            nn.LeakyReLU(0.2),
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                256, 256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                32, 16,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                16, 1,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Sigmoid()
        )

    def forward(self, k, block_index):

        key_features = self.key_encoder(k)

        block_features = self.block_encoder(
            block_index
        )

        combined = torch.cat(
            [key_features, block_features],
            dim=1
        )

        x = self.fc(combined)

        x = x.view(
            -1,
            256,
            4,
            4
        )

        return self.decoder(x)


# ============================================================
# 5. LOAD K AND GAN
# ============================================================

K = np.load(K_FILE).astype(np.float32)

if len(K) != 30000:
    raise ValueError(
        "K must contain exactly 30000 values."
    )

K_tensor = torch.tensor(
    K,
    dtype=torch.float32,
    device=DEVICE
).unsqueeze(0)


generator = Generator().to(DEVICE)

generator.load_state_dict(
    torch.load(
        GENERATOR_FILE,
        map_location=DEVICE,
        weights_only=True
    )
)

generator.eval()

print("K and GAN loaded successfully.")


# ============================================================
# 6. GENERATE PERMUTATION
# ============================================================

def generate_permutation(block_index):

    normalized_index = (
        block_index / 511
    )

    block_tensor = torch.tensor(
        [[normalized_index]],
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():

        score_map = generator(
            K_tensor,
            block_tensor
        )

    scores = (
        score_map
        .squeeze()
        .cpu()
        .numpy()
    )

    permutation = np.argsort(
        scores.flatten(),
        kind="stable"
    )

    return permutation


# ============================================================
# 7. DIFFUSION MASK
# ============================================================

def generate_diffusion_mask(
    block_index,
    length
):

    indices = (
        np.arange(length)
        + block_index * length
    ) % len(K)

    return np.floor(
        K[indices] * 256
    ).astype(np.uint16)


# ============================================================
# 8. ENCRYPT IMAGE ARRAY
#
# This test encryption MUST match Stage 6 exactly:
# 1. GAN permutation is applied independently to each 256x256 block.
# 2. All scrambled blocks are reassembled into one complete image.
# 3. A single global two-pass K-derived diffusion is applied.
#
# NPCR/UACI therefore measure the current Stage 6 architecture.
# ============================================================

def encrypt_array(image_array):

    height, width = image_array.shape

    padded_height = (
        (height + BLOCK_SIZE - 1)
        // BLOCK_SIZE
    ) * BLOCK_SIZE

    padded_width = (
        (width + BLOCK_SIZE - 1)
        // BLOCK_SIZE
    ) * BLOCK_SIZE

    padded = np.pad(
        image_array,
        (
            (0, padded_height - height),
            (0, padded_width - width)
        ),
        mode="constant",
        constant_values=0
    )

    # --------------------------------------------------------
    # Stage 1: block-wise GAN permutation
    # --------------------------------------------------------

    scrambled_image = np.zeros_like(
        padded,
        dtype=np.uint8
    )

    blocks_y = padded_height // BLOCK_SIZE
    blocks_x = padded_width // BLOCK_SIZE

    block_index = 0

    for by in range(blocks_y):

        for bx in range(blocks_x):

            y1 = by * BLOCK_SIZE
            y2 = y1 + BLOCK_SIZE

            x1 = bx * BLOCK_SIZE
            x2 = x1 + BLOCK_SIZE

            block = padded[
                y1:y2,
                x1:x2
            ]

            permutation = generate_permutation(
                block_index
            )

            scrambled = block.flatten()[
                permutation
            ]

            scrambled_image[
                y1:y2,
                x1:x2
            ] = scrambled.reshape(
                BLOCK_SIZE,
                BLOCK_SIZE
            )

            block_index += 1

    # --------------------------------------------------------
    # Stage 2: GLOBAL K-derived two-pass diffusion
    # --------------------------------------------------------

    total_pixels = padded_height * padded_width

    indices = (
        np.arange(total_pixels)
    ) % len(K)

    mask = np.floor(
        K[indices] * 256
    ).astype(np.uint16)

    scrambled_flat = scrambled_image.flatten().astype(
        np.uint16
    )

    # Forward pass
    forward = np.zeros(
        total_pixels,
        dtype=np.uint16
    )

    forward[0] = (
        scrambled_flat[0]
        + mask[0]
    ) % 256

    for i in range(1, total_pixels):

        forward[i] = (
            scrambled_flat[i]
            + mask[i]
            + forward[i - 1]
        ) % 256

    # Backward pass with reversed mask
    reverse_mask = mask[::-1]

    encrypted_flat = np.zeros(
        total_pixels,
        dtype=np.uint16
    )

    last = total_pixels - 1

    encrypted_flat[last] = (
        forward[last]
        + reverse_mask[last]
    ) % 256

    for i in range(last - 1, -1, -1):

        encrypted_flat[i] = (
            forward[i]
            + reverse_mask[i]
            + encrypted_flat[i + 1]
        ) % 256

    return encrypted_flat.astype(
        np.uint8
    ).reshape(
        padded_height,
        padded_width
    )


# 9. LOAD ORIGINAL + ENCRYPTED IMAGE
# ============================================================

original = np.array(
    Image.open(
        ORIGINAL_FILE
    ).convert("L"),
    dtype=np.uint8
)

encrypted = np.array(
    Image.open(
        ENCRYPTED_FILE
    ).convert("L"),
    dtype=np.uint8
)

# ------------------------------------------------------------
# Pad original image exactly as Stage 6 did.
# This makes original and encrypted images the same size.
# ------------------------------------------------------------

height, width = original.shape

padded_height = (
    (height + BLOCK_SIZE - 1)
    // BLOCK_SIZE
) * BLOCK_SIZE

padded_width = (
    (width + BLOCK_SIZE - 1)
    // BLOCK_SIZE
) * BLOCK_SIZE

original_padded = np.pad(
    original,
    (
        (0, padded_height - height),
        (0, padded_width - width)
    ),
    mode="constant",
    constant_values=0
)

if original_padded.shape != encrypted.shape:
    raise ValueError(
        f"Dimension mismatch: "
        f"original padded = {original_padded.shape}, "
        f"encrypted = {encrypted.shape}"
    )

print(
    "Original padded shape:",
    original_padded.shape
)

print(
    "Encrypted shape:",
    encrypted.shape
)

print("\nImage shape:", original.shape)


# ============================================================
# 10. HISTOGRAM
# ============================================================

original_histogram = np.bincount(
    original_padded.flatten(),
    minlength=256
)

encrypted_histogram = np.bincount(
    encrypted.flatten(),
    minlength=256
)

print("\nHistogram calculated.")


# ============================================================
# 11. ENTROPY
# ============================================================

def entropy(image):

    histogram = np.bincount(
        image.flatten(),
        minlength=256
    )

    probabilities = (
        histogram / image.size
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    return -np.sum(
        probabilities *
        np.log2(probabilities)
    )


original_entropy = entropy(original_padded)
encrypted_entropy = entropy(encrypted)

print(
    "Original entropy:",
    original_entropy
)

print(
    "Encrypted entropy:",
    encrypted_entropy
)


# ============================================================
# 12. CORRELATION COEFFICIENT
# ============================================================

def correlation(x, y):

    x = x.astype(np.float64)
    y = y.astype(np.float64)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum(
        (x - x_mean) *
        (y - y_mean)
    )

    denominator = np.sqrt(
        np.sum((x - x_mean) ** 2)
        *
        np.sum((y - y_mean) ** 2)
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


def correlation_values(image):

    horizontal_x = image[:, :-1].flatten()
    horizontal_y = image[:, 1:].flatten()

    vertical_x = image[:-1, :].flatten()
    vertical_y = image[1:, :].flatten()

    diagonal_x = image[:-1, :-1].flatten()
    diagonal_y = image[1:, 1:].flatten()

    return (
        correlation(
            horizontal_x,
            horizontal_y
        ),
        correlation(
            vertical_x,
            vertical_y
        ),
        correlation(
            diagonal_x,
            diagonal_y
        )
    )


original_corr = correlation_values(
    original_padded
)

encrypted_corr = correlation_values(
    encrypted
)

print("\nCorrelation:")
print(
    "Original H/V/D:",
    original_corr
)

print(
    "Encrypted H/V/D:",
    encrypted_corr
)


# ============================================================
# 13. MSE
# ============================================================

def mse(image1, image2):

    difference = (
        image1.astype(np.float64)
        -
        image2.astype(np.float64)
    )

    return np.mean(
        difference ** 2
    )


mse_value = mse(
    original_padded,
    encrypted
)


# ============================================================
# 14. PSNR
# ============================================================

if mse_value == 0:

    psnr_value = float("inf")

else:

    psnr_value = (
        10 *
        np.log10(
            (255 ** 2) /
            mse_value
        )
    )


print("\nMSE:", mse_value)
print("PSNR:", psnr_value)


# ============================================================
# 15. NPCR + UACI
#
# Create a copy with ONE PIXEL changed.
# ============================================================

modified = original.copy()

modified[0, 0] = (
    int(modified[0, 0]) + 1
) % 256


print("\nEncrypting one-pixel modified image...")

encrypted_modified = encrypt_array(
    modified
)


# ============================================================
# 16. NPCR
# ============================================================

changed_pixels = np.sum(
    encrypted != encrypted_modified
)

total_pixels = encrypted.size

npcr = (
    changed_pixels /
    total_pixels
) * 100


# ============================================================
# 17. UACI
# ============================================================

uaci = (
    np.mean(
        np.abs(
            encrypted.astype(np.int16)
            -
            encrypted_modified.astype(
                np.int16
            )
        )
    )
    / 255
) * 100


print("Changed pixels:", changed_pixels)
print("Total pixels:", total_pixels)
print("NPCR:", npcr, "%")
print("UACI:", uaci, "%")


# ============================================================
# 18. SAVE HISTOGRAM DATA
# ============================================================

histogram_file = os.path.join(
    BASE_DIR,
    "stage7_histogram_data.npz"
)

np.savez(
    histogram_file,
    original=original_histogram,
    encrypted=encrypted_histogram
)


# ============================================================
# 19. SAVE RESULTS
# ============================================================

with open(
    RESULT_FILE,
    "w"
) as f:

    f.write(
        "STAGE 7 SECURITY ANALYSIS\n"
    )

    f.write(
        "==========================\n\n"
    )

    f.write(
        f"Image dimensions: "
        f"{original.shape[1]} x "
        f"{original.shape[0]}\n\n"
    )

    f.write(
        "ENTROPY\n"
    )

    f.write(
        f"Original: {original_entropy:.10f}\n"
    )

    f.write(
        f"Encrypted: {encrypted_entropy:.10f}\n\n"
    )

    f.write(
        "CORRELATION COEFFICIENT\n"
    )

    f.write(
        f"Original Horizontal: "
        f"{original_corr[0]:.10f}\n"
    )

    f.write(
        f"Original Vertical: "
        f"{original_corr[1]:.10f}\n"
    )

    f.write(
        f"Original Diagonal: "
        f"{original_corr[2]:.10f}\n"
    )

    f.write(
        f"Encrypted Horizontal: "
        f"{encrypted_corr[0]:.10f}\n"
    )

    f.write(
        f"Encrypted Vertical: "
        f"{encrypted_corr[1]:.10f}\n"
    )

    f.write(
        f"Encrypted Diagonal: "
        f"{encrypted_corr[2]:.10f}\n\n"
    )

    f.write(
        "MSE / PSNR\n"
    )

    f.write(
        f"MSE: {mse_value:.10f}\n"
    )

    f.write(
        f"PSNR: {psnr_value:.10f}\n\n"
    )

    f.write(
        "NPCR / UACI\n"
    )

    f.write(
        f"NPCR: {npcr:.10f}%\n"
    )

    f.write(
        f"UACI: {uaci:.10f}%\n"
    )


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n========================================")
print("STAGE 7 COMPLETE")
print("========================================")

print(
    f"Encrypted entropy : "
    f"{encrypted_entropy:.6f}"
)

print(
    f"Encrypted correlation H : "
    f"{encrypted_corr[0]:.6f}"
)

print(
    f"Encrypted correlation V : "
    f"{encrypted_corr[1]:.6f}"
)

print(
    f"Encrypted correlation D : "
    f"{encrypted_corr[2]:.6f}"
)

print(
    f"MSE : {mse_value:.6f}"
)

print(
    f"PSNR : {psnr_value:.6f} dB"
)

print(
    f"NPCR : {npcr:.6f}%"
)

print(
    f"UACI : {uaci:.6f}%"
)

print("\nResults saved to:")
print(RESULT_FILE)

print("\nHistogram data saved to:")
print(histogram_file)