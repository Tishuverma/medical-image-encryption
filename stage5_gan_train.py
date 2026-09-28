import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
# 2. PARAMETERS
# ============================================================

K_FILE = "final_sequence.npy"

BLOCK_SIZE = 256
OUTPUT_SIZE = BLOCK_SIZE * BLOCK_SIZE

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.0002

# Number of deterministic training conditions generated
NUM_TRAINING_SAMPLES = 512

LATENT_SIZE = 128


# ============================================================
# 3. LOAD FINAL CHAOTIC SEQUENCE K
# ============================================================

K = np.load(K_FILE).astype(np.float32)

print("K length:", len(K))

if len(K) != 30000:
    raise ValueError(
        f"Expected K length 30000, but got {len(K)}"
    )


# ============================================================
# 4. GENERATOR
#
# Input:
#   K + block index
#
# Output:
#   256 x 256 score map
# ============================================================

class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        # Compress the 30,000-value K into a feature vector.
        self.key_encoder = nn.Sequential(
            nn.Linear(30000, 1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, LATENT_SIZE),
            nn.LeakyReLU(0.2),
        )

        # Block index is converted into a small feature vector.
        self.block_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(0.2),

            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(LATENT_SIZE + 32, 4 * 4 * 256),
            nn.LeakyReLU(0.2),
        )

        # 4x4 -> 8x8 -> 16x16 -> 32x32
        # -> 64x64 -> 128x128 -> 256x256
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

        block_features = self.block_encoder(block_index)

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

        x = self.decoder(x)

        return x


# ============================================================
# 5. DISCRIMINATOR
# ============================================================

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(
                1, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2),

            nn.Conv2d(
                32, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(
                64, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(
                128, 256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(
                256, 512,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.network(x)

        return self.output(x)


# ============================================================
# 6. CREATE TRAINING INPUTS
#
# We use the SAME K.
# Different block indices provide different conditions.
# ============================================================

K_tensor = torch.tensor(K, dtype=torch.float32)

training_K = K_tensor.unsqueeze(0).repeat(
    NUM_TRAINING_SAMPLES,
    1
)

block_indices = torch.arange(
    NUM_TRAINING_SAMPLES,
    dtype=torch.float32
).unsqueeze(1)

# Normalize block index to approximately [0,1]
block_indices = block_indices / max(
    NUM_TRAINING_SAMPLES - 1,
    1
)

# Move to device
training_K = training_K.to(DEVICE)
block_indices = block_indices.to(DEVICE)


# ============================================================
# 7. CREATE MODELS
# ============================================================

G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

print("\nGenerator:")
print(G)

print("\nDiscriminator:")
print(D)


# ============================================================
# 8. LOSS + OPTIMIZERS
# ============================================================

criterion = nn.BCELoss()

optimizer_G = optim.Adam(
    G.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)

optimizer_D = optim.Adam(
    D.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)


# ============================================================
# 9. TRAIN GAN
# ============================================================

print("\nStarting GAN training...\n")

num_batches = (
    NUM_TRAINING_SAMPLES + BATCH_SIZE - 1
) // BATCH_SIZE


for epoch in range(EPOCHS):

    permutation = torch.randperm(
        NUM_TRAINING_SAMPLES,
        device=DEVICE
    )

    total_g_loss = 0.0
    total_d_loss = 0.0

    for batch_number in range(num_batches):

        start = batch_number * BATCH_SIZE
        end = min(
            start + BATCH_SIZE,
            NUM_TRAINING_SAMPLES
        )

        indices = permutation[start:end]

        real_k = training_K[indices]

        real_block = block_indices[indices]

        batch_size = real_k.shape[0]

        # ----------------------------------------------------
        # REAL SCORE MAP
        #
        # Training target distribution: uniform [0,1].
        # This is only GAN training data.
        # It is NOT used as an encryption key.
        # ----------------------------------------------------

        real_maps = torch.rand(
            batch_size,
            1,
            BLOCK_SIZE,
            BLOCK_SIZE,
            device=DEVICE
        )

        real_labels = torch.ones(
            batch_size,
            1,
            device=DEVICE
        )

        fake_labels = torch.zeros(
            batch_size,
            1,
            device=DEVICE
        )

        # ====================================================
        # TRAIN DISCRIMINATOR
        # ====================================================

        optimizer_D.zero_grad()

        real_output = D(real_maps)

        real_loss = criterion(
            real_output,
            real_labels
        )

        fake_maps = G(
            real_k,
            real_block
        )

        fake_output = D(
            fake_maps.detach()
        )

        fake_loss = criterion(
            fake_output,
            fake_labels
        )

        d_loss = real_loss + fake_loss

        d_loss.backward()

        optimizer_D.step()

        # ====================================================
        # TRAIN GENERATOR
        # ====================================================

        optimizer_G.zero_grad()

        fake_maps = G(
            real_k,
            real_block
        )

        fake_output = D(fake_maps)

        g_loss = criterion(
            fake_output,
            real_labels
        )

        g_loss.backward()

        optimizer_G.step()

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()

    avg_g = total_g_loss / num_batches
    avg_d = total_d_loss / num_batches

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"D Loss: {avg_d:.4f} | "
        f"G Loss: {avg_g:.4f}"
    )


# ============================================================
# 10. SAVE TRAINED GENERATOR
# ============================================================

generator_file = "stage5_gan_generator.pth"

torch.save(
    G.state_dict(),
    generator_file
)

print("\nGenerator saved as:")
print(generator_file)


# ============================================================
# 11. GENERATE SCORE MAP FOR BLOCK 0
# ============================================================

G.eval()

with torch.no_grad():

    test_k = K_tensor.unsqueeze(0).to(DEVICE)

    test_block = torch.tensor(
        [[0.0]],
        dtype=torch.float32,
        device=DEVICE
    )

    score_map = G(
        test_k,
        test_block
    )

score_map = score_map.squeeze().cpu().numpy()


# ============================================================
# 12. CONVERT SCORE MAP INTO PERMUTATION
# ============================================================

flat_scores = score_map.flatten()

permutation = np.argsort(
    flat_scores,
    kind="stable"
)


# ============================================================
# 13. VERIFY PERMUTATION
# ============================================================

expected = np.arange(
    OUTPUT_SIZE
)

is_valid = np.array_equal(
    np.sort(permutation),
    expected
)

print("\nPermutation length:", len(permutation))

print(
    "Permutation validity:",
    "PASSED" if is_valid else "FAILED"
)

print("\nFirst 20 permutation indices:")
print(permutation[:20])


# ============================================================
# 14. DETERMINISM TEST
# ============================================================

G.eval()

with torch.no_grad():

    score_map_2 = G(
        test_k,
        test_block
    )

score_map_2 = (
    score_map_2
    .squeeze()
    .cpu()
    .numpy()
)

permutation_2 = np.argsort(
    score_map_2.flatten(),
    kind="stable"
)

deterministic = np.array_equal(
    permutation,
    permutation_2
)

print(
    "\nDETERMINISM CHECK:",
    "PASSED" if deterministic else "FAILED"
)


# ============================================================
# 15. SAVE BLOCK-0 PERMUTATION
# ============================================================

np.save(
    "stage5_block0_permutation.npy",
    permutation
)

np.save(
    "stage5_block0_score_map.npy",
    score_map
)

print("\nSaved:")
print("stage5_block0_permutation.npy")
print("stage5_block0_score_map.npy")