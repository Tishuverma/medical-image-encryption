import hashlib
from pathlib import Path


# ============================================================
# 1. FINGERPRINT IMAGE PATH
# ============================================================

folder_path = Path(
    "/Users/tishuverma/Desktop/Project/forensic_biometric_dataset/Real/ID 2/jpg"
)


# ============================================================
# 2. FIND IMAGE FILES
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

images = sorted(
    [
        p
        for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]
)

if not images:
    print("No fingerprint images found.")
    exit()


print("Images found:")
for i, image in enumerate(images, start=1):
    print(f"{i}. {image.name}")


# ============================================================
# 3. SELECT ONE FINGERPRINT IMAGE
# ============================================================

# Currently selecting the first image
image_path = images[0]

print("\nSelected image:")
print(image_path)


# ============================================================
# 4. GENERATE SHA-256 HASH
# ============================================================

sha256 = hashlib.sha256()

with open(image_path, "rb") as f:
    while chunk := f.read(8192):
        sha256.update(chunk)


# Hexadecimal representation
hash_hex = sha256.hexdigest()

# Raw 32-byte representation
hash_bytes = sha256.digest()


print("\nSHA-256 hash:")
print(hash_hex)

print("\nHash length:")
print(len(hash_hex), "hexadecimal characters")

print("\nHash size:")
print(len(hash_bytes), "bytes = 256 bits")


# ============================================================
# 5. CONVERT SHA-256 TO 256-BIT BINARY SEQUENCE
# ============================================================

binary_256 = bin(int.from_bytes(hash_bytes, byteorder="big"))[2:]

# Add leading zeros if necessary
binary_256 = binary_256.zfill(256)


print("\n256-bit binary sequence:")
print(binary_256)

print("\nNumber of bits:")
print(len(binary_256))


# ============================================================
# 6. TAKE FIRST 240 BITS
# ============================================================

first_240_bits = binary_256[:240]

remaining_16_bits = binary_256[240:]

print("\nFirst 240 bits:")
print(first_240_bits)

print("\nRemaining 16 bits:")
print(remaining_16_bits)


# ============================================================
# 7. DIVIDE 240 BITS INTO SIX 40-BIT PARTS
# ============================================================

B1_bits = first_240_bits[0:40]
B2_bits = first_240_bits[40:80]
B3_bits = first_240_bits[80:120]
B4_bits = first_240_bits[120:160]
B5_bits = first_240_bits[160:200]
B6_bits = first_240_bits[200:240]


B_bits = [
    B1_bits,
    B2_bits,
    B3_bits,
    B4_bits,
    B5_bits,
    B6_bits
]


print("\n40-bit blocks:")

for i, bits in enumerate(B_bits, start=1):
    print(f"B{i} = {bits}")
    print(f"     Length = {len(bits)} bits")


# ============================================================
# 8. CONVERT EACH 40-BIT BLOCK TO INTEGER
# ============================================================

B_values = [
    int(bits, 2)
    for bits in B_bits
]


print("\nInteger values:")

for i, value in enumerate(B_values, start=1):
    print(f"B{i} = {value}")


# ============================================================
# 9. NORMALIZE EACH VALUE TO RANGE [0, 1]
# ============================================================

MAX_40_BIT = (2 ** 40) - 1

normalized_values = [
    value / MAX_40_BIT
    for value in B_values
]


# ============================================================
# 10. STORE AS n1 TO n6
# ============================================================

n1, n2, n3, n4, n5, n6 = normalized_values


print("\nNormalized values:")

print(f"n1 = {n1:.15f}")
print(f"n2 = {n2:.15f}")
print(f"n3 = {n3:.15f}")
print(f"n4 = {n4:.15f}")
print(f"n5 = {n5:.15f}")
print(f"n6 = {n6:.15f}")


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL KEY GENERATION SUMMARY")
print("=" * 60)

print("\nSHA-256:")
print(hash_hex)

print("\nB1 to B6:")

for i, (bits, value, normalized) in enumerate(
    zip(B_bits, B_values, normalized_values),
    start=1
):
    print(f"\nB{i} (40 bits) = {bits}")
    print(f"B{i} (integer) = {value}")
    print(f"n{i} = {normalized:.15f}")

print("\n" + "=" * 60)