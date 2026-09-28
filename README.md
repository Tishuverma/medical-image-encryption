# Medical Image Encryption Using an Image-Derived Key, 3D CSLCM and GAN-Based Pixel Permutation

This repository contains the implementation of a medical image encryption framework using an image-derived key, a 3D Cubic-Sine Logistic Coupled Map (3D CSLCM), GAN-based pixel permutation, and global two-pass diffusion.

---

## System Overview

The encryption pipeline consists of the following stages:

```text
Input Image
     ↓
   SHA-256
     ↓
Image-Derived Key
     ↓
   3D CSLCM
     ↓
    X, Y, Z
     ↓
Reversible Interleaving
     ↓
 K (30,000)
     ↓
GAN-Based Pixel Permutation
     ↓
Scrambled Image
     ↓
Global Two-Pass Diffusion
     ↓
Encrypted Medical Image

The corresponding decryption process is:

Encrypted Medical Image
        ↓
Inverse Global Diffusion
        ↓
Inverse GAN-Based Permutation
        ↓
Remove Padding
        ↓
Original Medical Image
Project Structure
medical-image-encryption/
│
├── figures/
│   ├── cslcm_bifurcation_X.png
│   ├── cslcm_bifurcation_Y.png
│   ├── cslcm_bifurcation_Z.png
│   ├── stage8_histogram_comparison.png
│   └── stage8_image_comparison.png
│
├── Results/
│   ├── stage7_security_results.txt
│   └── stage8_validation_report.txt
│
├── stage1_key_generation.py
├── stage2_cslcm.py
├── stage3_cslcm_analysis.py
├── stage4_sequence_combination.py
├── stage5_gan_train.py
├── stage6_image_encryption_global_diffusion.py
├── stage7_security_analysis_global_diffusion.py
├── stage8_visual_validation.py
│
├── .gitignore
├── README.md
└── requirements.txt
Methodology
1. Image-Derived Key Generation

An input image is processed using SHA-256 to derive the parameters used by the encryption system.

The SHA-256 digest is converted into six normalized parameters:

n1, n2, n3, n4, n5, n6

These parameters are used to initialize and parameterize the 3D CSLCM system.

2. 3D CSLCM Sequence Generation

The image-derived parameters are mapped to the initial conditions and parameters of the 3D CSLCM system.

Three sequences are generated:

X
Y
Z

Each sequence contains 10,000 values.

3. Chaotic-System Analysis

The generated chaotic system is analyzed using:

Bifurcation analysis
Largest Lyapunov exponent analysis
4. Reversible Sequence Combination

The three generated sequences are reversibly interleaved:

K = [X1, Y1, Z1, X2, Y2, Z2, ...]

The resulting sequence contains 30,000 values.

A reversibility check is performed to verify that the original sequences can be recovered.

5. GAN-Based Pixel Permutation

The complete sequence K and the normalized block index are provided to the GAN.

For each 256 × 256 image block, the GAN generates a score map containing 65,536 scores.

The score map is converted into a deterministic pixel permutation using sorting.

6. Medical Image Encryption

The input medical image is converted to grayscale.

For the experimental image:

Original size: 447 × 447
Padded size:   512 × 512
Block size:    256 × 256

The padded image is divided into four 256 × 256 blocks.

Each block is independently permuted using the GAN-generated permutation.

The scrambled blocks are reassembled into a single 512 × 512 image.

A global two-pass K-derived diffusion operation is then applied to the complete scrambled image.

Security Analysis

The encrypted image is evaluated using:

Entropy
Horizontal correlation
Vertical correlation
Diagonal correlation
Mean Squared Error (MSE)
Peak Signal-to-Noise Ratio (PSNR)
Number of Pixels Change Rate (NPCR)
Unified Average Changing Intensity (UACI)
Experimental Results
Metric	Result
Original entropy	5.780389
Encrypted entropy	7.999288
Horizontal correlation	-0.003726
Vertical correlation	-0.001111
Diagonal correlation	0.003142
MSE	14441.355354
PSNR	6.534724 dB
NPCR	99.710464%
UACI	32.253705%

The security metrics are calculated on the 512 × 512 padded encryption domain.

Decryption and Visual Validation

The decrypted image is compared pixel-by-pixel with the original image.

ORIGINAL == DECRYPTED : PASSED

Maximum pixel difference: 0
Total absolute difference: 0
Changed pixels: 0
Environment

The implementation was developed and tested using:

Python 3.12
PyTorch
NumPy
Matplotlib
Pillow
OpenCV
Apple MPS
Running the Pipeline

Run the stages sequentially:

python stage1_key_generation.py
python stage2_cslcm.py
python stage3_cslcm_analysis.py
python stage4_sequence_combination.py
python stage5_gan_train.py
python stage6_image_encryption_global_diffusion.py
python stage7_security_analysis_global_diffusion.py
python stage8_visual_validation.py
Important Notes
The input medical image is not included in this repository.
The image dataset used during development is not included.
Generated model weights and NumPy sequence files are excluded from version control.
The reported security metrics correspond to the 512 × 512 padded encryption domain.
The reported metrics are experimental measurements and should not be interpreted individually as proof that the encryption system is completely secure or unbreakable.
