import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# FILE PATHS
# ============================================================

IMAGE_PATH = "/Users/tishuverma/Desktop/Image/chest x ray.jpeg"

ENCRYPTED_PATH = "/Users/tishuverma/Desktop/Image/encrypted_medical_image.png"

DECRYPTED_PATH = "/Users/tishuverma/Desktop/Image/decrypted_medical_image.png"

RESULTS_PATH = "/Users/tishuverma/Desktop/Image/stage7_security_results.txt"

OUTPUT_DIR = "/Users/tishuverma/Desktop/Image/stage8_results"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD IMAGES
# ============================================================

original = Image.open(IMAGE_PATH).convert("L")
encrypted = Image.open(ENCRYPTED_PATH).convert("L")
decrypted = Image.open(DECRYPTED_PATH).convert("L")


original_array = np.array(original)
encrypted_array = np.array(encrypted)
decrypted_array = np.array(decrypted)


print("=" * 60)
print("STAGE 8 - VISUAL VALIDATION")
print("=" * 60)


# ============================================================
# IMAGE INFORMATION
# ============================================================

print()
print("Original image:")
print("Shape:", original_array.shape)

print()
print("Encrypted image:")
print("Shape:", encrypted_array.shape)

print()
print("Decrypted image:")
print("Shape:", decrypted_array.shape)


# ============================================================
# EXACT DECRYPTION CHECK
# ============================================================

exact_recovery = np.array_equal(
    original_array,
    decrypted_array
)


print()
print("=" * 60)
print("DECRYPTION VERIFICATION")
print("=" * 60)

if exact_recovery:

    print("ORIGINAL == DECRYPTED : PASSED")
    print("Exact pixel recovery confirmed.")

else:

    print("ORIGINAL == DECRYPTED : FAILED")
    print("Decrypted image is not identical to original image.")


# ============================================================
# PAD ORIGINAL IMAGE
# ============================================================

pad_bottom = encrypted_array.shape[0] - original_array.shape[0]
pad_right = encrypted_array.shape[1] - original_array.shape[1]


if pad_bottom < 0 or pad_right < 0:

    raise ValueError(
        "Encrypted image is smaller than the original image."
    )


padded_original = np.pad(
    original_array,
    (
        (0, pad_bottom),
        (0, pad_right)
    ),
    mode="constant",
    constant_values=0
)


print()
print("Padded original shape:")
print(padded_original.shape)


# ============================================================
# ORIGINAL VS ENCRYPTED VS DECRYPTED
# ============================================================

plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)

plt.imshow(
    original_array,
    cmap="gray"
)

plt.title("Original Medical Image")
plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    encrypted_array,
    cmap="gray"
)

plt.title("Encrypted Image")
plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    decrypted_array,
    cmap="gray"
)

plt.title("Decrypted Image")
plt.axis("off")


plt.tight_layout()


comparison_path = os.path.join(
    OUTPUT_DIR,
    "stage8_image_comparison.png"
)


plt.savefig(
    comparison_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print()
print("Image comparison saved:")
print(comparison_path)


# ============================================================
# HISTOGRAM COMPARISON
# ============================================================

plt.figure(figsize=(12, 5))


plt.hist(
    original_array.flatten(),
    bins=256,
    range=(0, 256),
    alpha=0.6,
    label="Original"
)


plt.hist(
    encrypted_array.flatten(),
    bins=256,
    range=(0, 256),
    alpha=0.6,
    label="Encrypted"
)


plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.title(
    "Original vs Encrypted Histogram"
)

plt.legend()


plt.tight_layout()


histogram_path = os.path.join(
    OUTPUT_DIR,
    "stage8_histogram_comparison.png"
)


plt.savefig(
    histogram_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print()
print("Histogram comparison saved:")
print(histogram_path)


# ============================================================
# ABSOLUTE DIFFERENCE:
# ORIGINAL VS ENCRYPTED
# ============================================================

difference_encrypted = np.abs(
    padded_original.astype(np.int16)
    -
    encrypted_array.astype(np.int16)
)


plt.figure(figsize=(6, 6))


plt.imshow(
    difference_encrypted,
    cmap="gray"
)


plt.title(
    "Absolute Difference: Original vs Encrypted"
)

plt.axis("off")


plt.tight_layout()


difference_encrypted_path = os.path.join(
    OUTPUT_DIR,
    "stage8_difference_original_encrypted.png"
)


plt.savefig(
    difference_encrypted_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print()
print("Original-encrypted difference image saved:")
print(difference_encrypted_path)


# ============================================================
# DECRYPTION DIFFERENCE
# ============================================================
#
# This should contain ONLY zeros if exact recovery succeeds.
# ============================================================

difference_decrypted = np.abs(
    original_array.astype(np.int16)
    -
    decrypted_array.astype(np.int16)
)


max_decryption_difference = np.max(
    difference_decrypted
)

total_decryption_difference = np.sum(
    difference_decrypted
)

changed_decryption_pixels = np.count_nonzero(
    difference_decrypted
)


print()
print("=" * 60)
print("DECRYPTION DIFFERENCE ANALYSIS")
print("=" * 60)

print(
    "Maximum pixel difference:",
    max_decryption_difference
)

print(
    "Total absolute difference:",
    total_decryption_difference
)

print(
    "Changed pixels:",
    changed_decryption_pixels
)


plt.figure(figsize=(6, 6))


plt.imshow(
    difference_decrypted,
    cmap="gray"
)


plt.title(
    "Absolute Difference: Original vs Decrypted"
)

plt.axis("off")


plt.tight_layout()


difference_decrypted_path = os.path.join(
    OUTPUT_DIR,
    "stage8_difference_original_decrypted.png"
)


plt.savefig(
    difference_decrypted_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print()
print("Original-decrypted difference image saved:")
print(difference_decrypted_path)


# ============================================================
# READ STAGE 7 RESULTS
# ============================================================

stage7_text = ""


if os.path.exists(RESULTS_PATH):

    with open(
        RESULTS_PATH,
        "r"
    ) as file:

        stage7_text = file.read()

else:

    print()
    print("WARNING:")
    print("Stage 7 results file was not found.")


# ============================================================
# CREATE VALIDATION REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "stage8_validation_report.txt"
)


with open(
    report_path,
    "w"
) as report:

    report.write("=" * 60 + "\n")
    report.write(
        "STAGE 8 - VISUAL AND QUANTITATIVE VALIDATION\n"
    )
    report.write("=" * 60 + "\n\n")


    # --------------------------------------------------------
    # IMAGE INFORMATION
    # --------------------------------------------------------

    report.write("IMAGE INFORMATION\n")
    report.write("-" * 60 + "\n")

    report.write(
        f"Original image shape: {original_array.shape}\n"
    )

    report.write(
        f"Encrypted image shape: {encrypted_array.shape}\n"
    )

    report.write(
        f"Decrypted image shape: {decrypted_array.shape}\n"
    )

    report.write(
        f"Padded original shape: {padded_original.shape}\n\n"
    )


    # --------------------------------------------------------
    # DECRYPTION VERIFICATION
    # --------------------------------------------------------

    report.write("DECRYPTION VERIFICATION\n")
    report.write("-" * 60 + "\n")

    if exact_recovery:

        report.write(
            "ORIGINAL == DECRYPTED : PASSED\n"
        )

        report.write(
            "Exact pixel recovery confirmed.\n"
        )

    else:

        report.write(
            "ORIGINAL == DECRYPTED : FAILED\n"
        )

        report.write(
            "Decrypted image is not identical to original image.\n"
        )


    report.write("\n")


    # --------------------------------------------------------
    # DECRYPTION DIFFERENCE
    # --------------------------------------------------------

    report.write("DECRYPTION DIFFERENCE ANALYSIS\n")
    report.write("-" * 60 + "\n")

    report.write(
        f"Maximum pixel difference: "
        f"{max_decryption_difference}\n"
    )

    report.write(
        f"Total absolute difference: "
        f"{total_decryption_difference}\n"
    )

    report.write(
        f"Changed pixels: "
        f"{changed_decryption_pixels}\n\n"
    )


    # --------------------------------------------------------
    # STAGE 7 RESULTS
    # --------------------------------------------------------

    report.write("STAGE 7 SECURITY RESULTS\n")
    report.write("-" * 60 + "\n")

    if stage7_text:

        report.write(stage7_text)

    else:

        report.write(
            "Stage 7 results unavailable.\n"
        )


    report.write("\n")


    # --------------------------------------------------------
    # GENERATED FIGURES
    # --------------------------------------------------------

    report.write("GENERATED FIGURES\n")
    report.write("-" * 60 + "\n")

    report.write(
        f"{comparison_path}\n"
    )

    report.write(
        f"{histogram_path}\n"
    )

    report.write(
        f"{difference_encrypted_path}\n"
    )

    report.write(
        f"{difference_decrypted_path}\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 60)
print("STAGE 8 COMPLETE")
print("=" * 60)


print()
print("Output directory:")
print(OUTPUT_DIR)


print()
print("Generated files:")

print(
    "1.",
    comparison_path
)

print(
    "2.",
    histogram_path
)

print(
    "3.",
    difference_encrypted_path
)

print(
    "4.",
    difference_decrypted_path
)

print(
    "5.",
    report_path
)


print()
print("Stage 8 visual validation finished.")