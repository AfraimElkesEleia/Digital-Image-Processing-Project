# DIP Studio Algorithms Registry

This document outlines the available image processing algorithms in DIP Studio, categorized by their domain of operation. Each algorithm details its core function, whether it supports kernel modifications or padding, and its tunable parameters.

---

## 📌 Point Processing

Algorithms that operate on individual pixels independently of their neighbors.

### Negative (`NEG`)
* **Description**: Inverts pixel intensities: `out = 255 - in`.

### Log Transform (`LOG`)
* **Description**: Compresses high-intensity range: `out = c · log(1 + in)`.
* **Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `c` | c | `float` | 1.0 | 0.1 to 10.0 | 0.1 |

### Gamma / Power Law (`GAM`)
* **Description**: Power-law transformation: `out = c · inᵞ`.
* **Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `c` | c | `float` | 1.0 | 0.1 to 5.0 | 0.1 |
| `gamma` | γ | `float` | 1.0 | 0.1 to 5.0 | 0.05 |

### Thresholding (`THR`)
* **Description**: Binary threshold: pixels above T become 255, rest 0.
* **Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `threshold` | T | `int` | 128 | 0 to 255 | 1 |

### Gray-Level Slicing (`GLS`)
* **Description**: Highlight a band `[low, high]` of gray levels.
* **Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `low` | Low | `int` | 100 | 0 to 255 | 1 |
| `high` | High | `int` | 200 | 0 to 255 | 1 |
| `preserve_bg` | Preserve | `bool` | True | - | - |

### Bit-Plane Slicing (`BIT`)
* **Description**: Extract a single bit plane (0=LSB … 7=MSB).
* **Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `bit` | Bit | `int` | 7 | 0 to 7 | 1 |

---

## 📊 Histogram

Techniques for modifying the image histogram to enhance contrast.

### Contrast Stretching (`CS`)
* **Description**: Piecewise-linear stretch from `(r1,s1) → (r2,s2)`.
* **Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `r1` | r1 | `int` | 50 | 0 to 255 | 1 |
| `s1` | s1 | `int` | 20 | 0 to 255 | 1 |
| `r2` | r2 | `int` | 200 | 0 to 255 | 1 |
| `s2` | s2 | `int` | 230 | 0 to 255 | 1 |

### Histogram Equalization (`HEQ`)
* **Description**: Global histogram equalization via CDF mapping.

### CLAHE (`CLA`)
* **Description**: Contrast-limited adaptive histogram equalization.
* **Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `clip_limit` | Clip | `float` | 2.0 | 0.5 to 10.0 | 0.5 |
| `tile_grid_size`| Grid | `int` | 8 | 2 to 16 | 2 |

---

## 🔲 Spatial Filter

Operations based on local pixel neighborhoods.

### Simple Average (`AVG`)
* **Description**: Box blur — uniform kernel, suppresses noise.
* **Flags**: `Has Kernel` | `Has Padding`
* **Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `kernel_size` | Kernel | `kernel` | 3 | - | `["3", "5", "9"]` |

### Weighted Average (`WA`)
* **Description**: Gaussian-weighted blur — smooth roll-off from center.
* **Flags**: `Has Kernel` | `Has Padding`
* **Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `kernel_size` | Kernel | `kernel` | 3 | - | `["3", "5", "9"]` |

### Median Filter (`MED`)
* **Description**: Replaces each pixel with the median of its neighborhood.
* **Flags**: `Has Kernel` | `Has Padding`
* **Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `kernel_size` | Kernel | `kernel` | 3 | - | `["3", "5", "9"]` |

### 1st Derivative (Gradient) (`GRD`)
* **Description**: Sobel magnitude — highlights edges.
* **Flags**: `Has Kernel`

### 2nd Derivative (Laplacian) (`LAP`)
* **Description**: Fixed 3×3 Laplacian kernel. Detects edges and fine detail.
* **Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `variant` | Variant | `choice` | Standard (−4) | - | `["Standard (−4)", "Diagonal (−8)", "Enhancement (+5)"]` |

### Sobel Filter (`SOB`)
* **Description**: Directional Sobel edge detection.
* **Flags**: `Has Kernel`
* **Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `dx` | dx | `int` | 1 | 0 to 1 | 1 |
| `dy` | dy | `int` | 1 | 0 to 1 | 1 |
| `ksize` | Kernel | `kernel` | 3 | - | `["3", "5", "9"]` |

---

## 🌊 Frequency Domain

Filtering operations performed after a Fourier transform.

### Low-Pass Filters (LPF)
These filters attenuate high frequencies, blurring the image.

* **Ideal LPF (`ILP`)**: Hard cutoff low-pass filter in frequency domain.
* **Gaussian LPF (`GLP`)**: Gaussian-shaped low-pass frequency filter.
* **Butterworth LPF (`BLP`)**: Smooth low-pass with adjustable order.

**Parameters:**
| Algorithm | Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *All LPFs* | `d0` | D0 | `float` | 30.0 | 1.0 to 200.0 | 1.0 |
| *BLP Only* | `order` | Order | `int` | 2 | 1 to 10 | 1 |

### High-Pass Filters (HPF)
These filters attenuate low frequencies, sharpening edges.

* **Ideal HPF (`IHP`)**: Hard cutoff high-pass filter.
* **Gaussian HPF (`GHP`)**: Gaussian-shaped high-pass frequency filter.
* **Butterworth HPF (`BHP`)**: Smooth high-pass with adjustable order.

**Parameters:**
| Algorithm | Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *All HPFs* | `d0` | D0 | `float` | 30.0 | 1.0 to 200.0 | 1.0 |
| *BHP Only* | `order` | Order | `int` | 2 | 1 to 10 | 1 |

### Band-Reject Filters
These filters block specific frequency bands (useful for periodic noise).

* **Band-Reject Ideal (`BRI`)**: Ideal notch/band-reject filter.
* **Band-Reject Gaussian (`BRG`)**: Gaussian band-reject filter.
* **Band-Reject Butterworth (`BRB`)**: Butterworth band-reject filter.

**Parameters:**
| Algorithm | Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *All BRs* | `d0` | D0 | `float` | 30.0 | 1.0 to 200.0 | 1.0 |
| *All BRs* | `w` | W | `float` | 10.0 | 1.0 to 100.0 | 1.0 |
| *BRB Only*| `order` | Order | `int` | 2 | 1 to 10 | 1 |

---

## 🛠️ Restoration

Algorithms specifically aimed at repairing degraded or noisy images.

*(Note: All Restoration filters below rely on the `kernel_size` parameter, defaulting to 3, with choices `["3", "5", "9"]`, unless otherwise specified.)*

### Arithmetic Mean (`AM`)
* **Description**: Simple averaging filter for noise reduction.
* **Flags**: `Has Kernel`

### Geometric Mean (`GM`)
* **Description**: Product-root mean — less blurring than arithmetic.

### Harmonic Mean (`HM`)
* **Description**: Good for salt noise; poor on pepper noise.
* **Flags**: `Has Kernel`

### Contraharmonic Mean (`CHM`)
* **Description**: Q>0 removes pepper; Q<0 removes salt noise.
* **Flags**: `Has Kernel`
* **Additional Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `q` | Q | `float` | 1.5 | -5.0 to 5.0 | 0.5 |

### Max Filter (`MAX`)
* **Description**: Replaces each pixel with neighborhood maximum.
* **Flags**: `Has Kernel`

### Min Filter (`MIN`)
* **Description**: Replaces each pixel with neighborhood minimum.
* **Flags**: `Has Kernel`

### Midpoint Filter (`MID`)
* **Description**: Average of max and min in neighborhood.
* **Flags**: `Has Kernel`

### Alpha-Trimmed Mean (`ATM`)
* **Description**: Mean after trimming d/2 extremes from each end.
* **Flags**: `Has Kernel`
* **Additional Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `d` | d | `int` | 2 | 0 to 8 | 2 |

### Adaptive Median (`ADM`)
* **Description**: Grows window until median is an impulse-free value.
* **Flags**: `Has Kernel`
* **Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `max_size` | Max K | `kernel` | 7 | - | `["3", "5", "7", "9"]` |

---

## 🧩 Segmentation

Tools for partitioning an image into distinct regions or objects.

### Point Detection (`PD`)
* **Description**: Laplacian-based isolated point detector.
* **Flags**: `Has Kernel`
* **Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `threshold` | T | `int` | 128 | 0 to 255 | 1 |

### Line Detection (`LD`)
* **Description**: Directional line detector using preset kernels.
* **Flags**: `Has Kernel`
* **Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `direction` | Dir | `choice` | Horizontal | - | `["Horizontal", "Vertical", "+45°", "-45°"]` |
| `threshold` | T | `int` | 128 | 0 to 255 | 1 |

### Edge Detectors
* **Prewitt Edge (`PRW`)**: Prewitt operator for edge detection. *(Flags: Has Kernel)*
* **Roberts Edge (`ROB`)**: Roberts cross-gradient operator. *(Flags: Has Kernel)*
* **Sobel Edge (`SBL`)**: Sobel magnitude followed by binary threshold. *(Flags: Has Kernel)*
    * **Parameters**: `threshold` (`int`, Display: T, Default: 128, Range: 0 to 255)
* **LoG Edge (`LOG`)**: Laplacian-of-Gaussian edge detector. *(Flags: Has Kernel)*
    * **Parameters**:
        * `ksize` (`kernel`, Display: Kernel, Default: 5, Choices: `["3", "5", "9"]`)
        * `sigma` (`float`, Display: σ, Default: 1.0, Range: 0.1 to 5.0, Step: 0.1)

### Thresholding
* **Global Threshold (`GT`)**: Iterative optimal global threshold estimation.
* **Adaptive Threshold (`AT`)**: Per-region threshold based on local mean. *(Flags: Has Kernel)*
    * **Parameters**:
        * `block_size` (`kernel`, Display: Block, Default: 11, Choices: `["3", "5", "9", "11", "15", "21"]`)
        * `c` (`int`, Display: C, Default: 2, Range: 0 to 20, Step: 1)

---

## 💥 Noise

Simulated noise generators for testing restoration algorithms.

### Salt & Pepper (`S&P`)
* **Description**: Randomly sets pixels to 255 (salt) or 0 (pepper).
* **Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `amount` | Amount | `float` | 0.05 | 0.001 to 0.5 | 0.005 |
| `salt_ratio` | Salt Ratio | `float` | 0.5 | 0.0 to 1.0 | 0.05 |

### Gaussian Noise (`GSN`)
* **Description**: Additive Gaussian (normal) random noise.
* **Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `mean` | Mean | `float` | 0.0 | -50.0 to 50.0 | 1.0 |
| `sigma` | Sigma | `float` | 25.0 | 1.0 to 150.0 | 1.0 |

### Speckle Noise (`SPK`)
* **Description**: Multiplicative noise: `out = in + in · N(0, σ²)`.
* **Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `sigma` | Sigma | `float` | 0.2 | 0.01 to 1.0 | 0.01 |

---

## 🧮 Arithmetic

Operations combining multiple images or states.

### Image Subtraction (`SUB`)
* **Description**: Subtract original or previous step — useful for unsharp masking.
* **Parameters**:

| Name | Display | Type | Default | Range | Step / Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `target` | Subtract | `choice` | Original | - | `["Original", "Previous Step"]` |
| `scale` | Scale | `float` | 1.0 | 0.0 to 3.0 | 0.05 |
| `mode` | Mode | `choice` | Absolute | - | `["Absolute", "Clipped", "Scaled"]` |
