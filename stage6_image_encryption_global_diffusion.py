import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ============================================================
# 1. DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# ============================================================
# 2. FILE PATHS
# ============================================================

BASE_DIR = "/Users/tishuverma/Desktop/Image"

K_FILE = os.path.join(
    BASE_DIR,
    "final_sequence.npy"
)

GAN_FILE = os.path.join(
    BASE_DIR,
    "stage5_gan_generator.pth"
)

ORIGINAL_FILE = os.path.join(
    BASE_DIR,
    "chest x ray.jpeg"
)

ENCRYPTED_FILE = os.path.join(
    BASE_DIR,
    "encrypted_medical_image.png"
)

DECRYPTED_FILE = os.path.join(
    BASE_DIR,
    "decrypted_medical_image.png"
)


# ============================================================
# 3. PARAMETERS
# ============================================================

BLOCK_SIZE = 256
BLOCK_PIXELS = BLOCK_SIZE * BLOCK_SIZE
LATENT_SIZE = 128


# ============================================================
# 4. LOAD K
# ============================================================

K = np.load(K_FILE).astype(np.float32)

print("K loaded successfully.")
print("K length:", len(K))

if len(K) != 30000:
    raise ValueError(
        f"Expected K length 30000, but got {len(K)}"
    )


# ============================================================
# 5. GENERATOR
#
# EXACT SAME ARCHITECTURE AS STAGE 5
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
            [
                key_features,
                block_features
            ],
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
# 6. LOAD GAN GENERATOR
# ============================================================

generator = Generator().to(DEVICE)

generator.load_state_dict(
    torch.load(
        GAN_FILE,
        map_location=DEVICE,
        weights_only=True
    )
)

generator.eval()

print("GAN generator loaded successfully.")


# ============================================================
# 7. K TENSOR
# ============================================================

K_tensor = torch.tensor(
    K,
    dtype=torch.float32,
    device=DEVICE
).unsqueeze(0)


# ============================================================
# 8. GENERATE GAN PERMUTATION
# ============================================================

@torch.no_grad()
def generate_permutation(block_index):

    # Same normalization used in Stage 5
    normalized_index = block_index / 511.0

    block_tensor = torch.tensor(
        [[normalized_index]],
        dtype=torch.float32,
        device=DEVICE
    )

    score_map = generator(
        K_tensor,
        block_tensor
    )

    scores = (
        score_map
        .squeeze()
        .reshape(-1)
        .cpu()
        .numpy()
    )

    permutation = np.argsort(
        scores,
        kind="stable"
    ).astype(np.int64)

    if len(permutation) != BLOCK_PIXELS:
        raise ValueError(
            "Invalid permutation length."
        )

    if not np.array_equal(
        np.sort(permutation),
        np.arange(BLOCK_PIXELS)
    ):
        raise ValueError(
            "Generated permutation is invalid."
        )

    return permutation


# ============================================================
# 9. GLOBAL K-DERIVED DIFFUSION MASK
#
# IMPORTANT:
# The mask is generated for the COMPLETE padded image,
# not separately for each block.
# ============================================================

def generate_global_diffusion_mask(length):

    indices = (
        np.arange(length)
    ) % len(K)

    return np.floor(
        K[indices] * 256
    ).astype(np.uint16)


# ============================================================
# 10. TWO-PASS GLOBAL DIFFUSION
#
# Forward pass:
#
# T[0] = S[0] + M[0]
# T[i] = S[i] + M[i] + T[i-1]
#
# Backward pass:
#
# C[last] = T[last] + M2[last]
# C[i] = T[i] + M2[i] + C[i+1]
#
# M2 = reverse(M)
# ============================================================

def global_diffusion_encrypt(
    scrambled_flat,
    mask
):

    scrambled_flat = scrambled_flat.astype(
        np.uint16
    )

    length = len(scrambled_flat)

    forward = np.zeros(
        length,
        dtype=np.uint16
    )

    # Forward chained diffusion
    forward[0] = (
        scrambled_flat[0]
        + mask[0]
    ) % 256

    for i in range(1, length):

        forward[i] = (
            scrambled_flat[i]
            + mask[i]
            + forward[i - 1]
        ) % 256

    # Backward chained diffusion
    reverse_mask = mask[::-1]

    encrypted = np.zeros(
        length,
        dtype=np.uint16
    )

    last = length - 1

    encrypted[last] = (
        forward[last]
        + reverse_mask[last]
    ) % 256

    for i in range(last - 1, -1, -1):

        encrypted[i] = (
            forward[i]
            + reverse_mask[i]
            + encrypted[i + 1]
        ) % 256

    return encrypted.astype(np.uint8)


# ============================================================
# 11. INVERSE TWO-PASS GLOBAL DIFFUSION
# ============================================================

def global_diffusion_decrypt(
    encrypted_flat,
    mask
):

    encrypted_flat = encrypted_flat.astype(
        np.int16
    )

    length = len(encrypted_flat)

    reverse_mask = mask[::-1]

    # Undo backward pass
    forward = np.zeros(
        length,
        dtype=np.int16
    )

    last = length - 1

    forward[last] = (
        encrypted_flat[last]
        - reverse_mask[last]
    ) % 256

    for i in range(last - 1, -1, -1):

        forward[i] = (
            encrypted_flat[i]
            - reverse_mask[i]
            - encrypted_flat[i + 1]
        ) % 256

    # Undo forward pass
    scrambled = np.zeros(
        length,
        dtype=np.int16
    )

    scrambled[0] = (
        forward[0]
        - mask[0]
    ) % 256

    for i in range(1, length):

        scrambled[i] = (
            forward[i]
            - mask[i]
            - forward[i - 1]
        ) % 256

    return scrambled.astype(np.uint8)


# ============================================================
# 12. ENCRYPT IMAGE
#
# Step 1: pad
# Step 2: GAN permutation independently for each block
# Step 3: reassemble the COMPLETE scrambled image
# Step 4: apply ONE GLOBAL diffusion to the entire image
# ============================================================

def encrypt_image(image_array):

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

    scrambled_image = np.zeros_like(
        padded,
        dtype=np.uint8
    )

    blocks_y = padded_height // BLOCK_SIZE
    blocks_x = padded_width // BLOCK_SIZE

    block_index = 0

    print("\nPermutation stage started...")

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

            flat_block = block.flatten()

            scrambled_block = flat_block[
                permutation
            ]

            scrambled_image[
                y1:y2,
                x1:x2
            ] = scrambled_block.reshape(
                BLOCK_SIZE,
                BLOCK_SIZE
            )

            print(
                f"Permuted block {block_index}"
            )

            block_index += 1

    # --------------------------------------------------------
    # GLOBAL DIFFUSION
    # --------------------------------------------------------

    total_pixels = (
        padded_height * padded_width
    )

    mask = generate_global_diffusion_mask(
        total_pixels
    )

    scrambled_flat = scrambled_image.flatten()

    encrypted_flat = global_diffusion_encrypt(
        scrambled_flat,
        mask
    )

    encrypted = encrypted_flat.reshape(
        padded_height,
        padded_width
    )

    return encrypted


# ============================================================
# 13. DECRYPT IMAGE
#
# Step 1: inverse GLOBAL diffusion
# Step 2: split complete scrambled image into blocks
# Step 3: inverse GAN permutation for each block
# Step 4: remove padding
# ============================================================

def decrypt_image(
    encrypted_array,
    original_height,
    original_width
):

    padded_height, padded_width = (
        encrypted_array.shape
    )

    total_pixels = (
        padded_height * padded_width
    )

    mask = generate_global_diffusion_mask(
        total_pixels
    )

    # --------------------------------------------------------
    # Undo global diffusion
    # --------------------------------------------------------

    encrypted_flat = encrypted_array.flatten()

    scrambled_flat = global_diffusion_decrypt(
        encrypted_flat,
        mask
    )

    scrambled_image = scrambled_flat.reshape(
        padded_height,
        padded_width
    )

    # --------------------------------------------------------
    # Undo block permutations
    # --------------------------------------------------------

    decrypted = np.zeros_like(
        scrambled_image,
        dtype=np.uint8
    )

    blocks_y = padded_height // BLOCK_SIZE
    blocks_x = padded_width // BLOCK_SIZE

    block_index = 0

    print("\nInverse permutation stage started...")

    for by in range(blocks_y):

        for bx in range(blocks_x):

            y1 = by * BLOCK_SIZE
            y2 = y1 + BLOCK_SIZE

            x1 = bx * BLOCK_SIZE
            x2 = x1 + BLOCK_SIZE

            scrambled_block = scrambled_image[
                y1:y2,
                x1:x2
            ]

            permutation = generate_permutation(
                block_index
            )

            scrambled_block_flat = (
                scrambled_block.flatten()
            )

            original_block_flat = np.zeros(
                BLOCK_PIXELS,
                dtype=np.uint8
            )

            # Encryption used:
            # scrambled = original[permutation]
            #
            # Therefore:
            # original[permutation] = scrambled

            original_block_flat[
                permutation
            ] = scrambled_block_flat

            decrypted[
                y1:y2,
                x1:x2
            ] = original_block_flat.reshape(
                BLOCK_SIZE,
                BLOCK_SIZE
            )

            print(
                f"Recovered block {block_index}"
            )

            block_index += 1

    # Remove padding
    return decrypted[
        :original_height,
        :original_width
    ]


# ============================================================
# 14. LOAD MEDICAL IMAGE
# ============================================================

original_image = Image.open(
    ORIGINAL_FILE
)

print("\nOriginal image:")
print("Mode:", original_image.mode)
print("Size:", original_image.size)

# Keep the same grayscale processing used previously
original_gray = original_image.convert("L")

original_array = np.array(
    original_gray,
    dtype=np.uint8
)

height, width = original_array.shape

print(
    "Grayscale shape:",
    original_array.shape
)


# ============================================================
# 15. ENCRYPT
# ============================================================

encrypted_array = encrypt_image(
    original_array
)

encrypted_image = Image.fromarray(
    encrypted_array,
    mode="L"
)

encrypted_image.save(
    ENCRYPTED_FILE
)

print("\nEncrypted image saved:")
print(ENCRYPTED_FILE)


# ============================================================
# 16. DECRYPT
# ============================================================

decrypted_array = decrypt_image(
    encrypted_array,
    height,
    width
)

decrypted_image = Image.fromarray(
    decrypted_array,
    mode="L"
)

decrypted_image.save(
    DECRYPTED_FILE
)

print("\nDecrypted image saved:")
print(DECRYPTED_FILE)


# ============================================================
# 17. EXACT DECRYPTION VERIFICATION
# ============================================================

print("\n")
print("=" * 40)
print("DECRYPTION VERIFICATION")
print("=" * 40)

if np.array_equal(
    original_array,
    decrypted_array
):

    print(
        "ORIGINAL == DECRYPTED : PASSED"
    )

    print(
        "Exact pixel recovery confirmed."
    )

else:

    print(
        "ORIGINAL == DECRYPTED : FAILED"
    )

    difference = np.abs(
        original_array.astype(np.int16)
        -
        decrypted_array.astype(np.int16)
    )

    print(
        "Maximum pixel difference:",
        difference.max()
    )

    print(
        "Different pixels:",
        np.sum(difference != 0)
    )