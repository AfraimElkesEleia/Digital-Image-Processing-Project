"""
models.py — Algorithm registry for DIP Studio
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Parameter:
    name:         str
    display_name: str
    type:         str          # "int" | "float" | "choice" | "bool" | "kernel"
    default:      object
    min_val:      float = 0
    max_val:      float = 255
    step:         float = 1
    choices:      List[str] = field(default_factory=list)


@dataclass
class AlgorithmModel:
    id:           str
    name:         str
    category:     str
    abbreviation: str
    description:  str
    parameters:   List[Parameter] = field(default_factory=list)
    has_kernel:   bool = False
    has_padding:  bool = False


# ── Kernel choices ────────────────────────────────────────────────────
KS = ["3", "5", "9"]          # standard odd kernel sizes
KS_LARGE = ["3", "5", "9", "11", "15", "21"]


CATEGORIES = [
    "Point Processing",
    "Histogram",
    "Spatial Filter",
    "Frequency Domain",
    "Restoration",
    "Segmentation",
    "Noise",
    "Arithmetic",
]


ALGORITHMS: List[AlgorithmModel] = [

    # ── Point Processing ──────────────────────────────────────────────
    AlgorithmModel(
        id="negative", name="Negative", category="Point Processing",
        abbreviation="NEG",
        description="Inverts pixel intensities: out = 255 - in.",
    ),
    AlgorithmModel(
        id="log", name="Log Transform", category="Point Processing",
        abbreviation="LOG",
        description="Compresses high-intensity range: out = c · log(1 + in).",
        parameters=[
            Parameter("c", "c", "float", 1.0, 0.1, 10.0, 0.1),
        ],
    ),
    AlgorithmModel(
        id="gamma", name="Gamma / Power Law", category="Point Processing",
        abbreviation="GAM",
        description="Power-law transformation: out = c · inᵞ.",
        parameters=[
            Parameter("c",     "c",     "float", 1.0, 0.1, 5.0,  0.1),
            Parameter("gamma", "γ",     "float", 1.0, 0.1, 5.0,  0.05),
        ],
    ),
    AlgorithmModel(
        id="thresholding", name="Thresholding", category="Point Processing",
        abbreviation="THR",
        description="Binary threshold: pixels above T become 255, rest 0.",
        parameters=[
            Parameter("threshold", "T", "int", 128, 0, 255, 1),
        ],
    ),
    AlgorithmModel(
        id="gray_slicing", name="Gray-Level Slicing", category="Point Processing",
        abbreviation="GLS",
        description="Highlight a band [low, high] of gray levels.",
        parameters=[
            Parameter("low",         "Low",      "int",  100, 0, 255, 1),
            Parameter("high",        "High",     "int",  200, 0, 255, 1),
            Parameter("preserve_bg", "Preserve", "bool", True),
        ],
    ),
    AlgorithmModel(
        id="bit_plane", name="Bit-Plane Slicing", category="Point Processing",
        abbreviation="BIT",
        description="Extract a single bit plane (0=LSB … 7=MSB).",
        parameters=[
            Parameter("bit", "Bit", "int", 7, 0, 7, 1),
        ],
    ),

    # ── Histogram ─────────────────────────────────────────────────────
    AlgorithmModel(
        id="contrast_stretching", name="Contrast Stretching", category="Histogram",
        abbreviation="CS",
        description="Piecewise-linear stretch from (r1,s1)→(r2,s2).",
        parameters=[
            Parameter("r1", "r1", "int",  50,  0, 255, 1),
            Parameter("s1", "s1", "int",  20,  0, 255, 1),
            Parameter("r2", "r2", "int", 200,  0, 255, 1),
            Parameter("s2", "s2", "int", 230,  0, 255, 1),
        ],
    ),
    AlgorithmModel(
        id="hist_equalization", name="Histogram Equalization", category="Histogram",
        abbreviation="HEQ",
        description="Global histogram equalization via CDF mapping.",
    ),
    AlgorithmModel(
        id="clahe", name="CLAHE", category="Histogram",
        abbreviation="CLA",
        description="Contrast-limited adaptive histogram equalization.",
        parameters=[
            Parameter("clip_limit",    "Clip",  "float", 2.0, 0.5, 10.0, 0.5),
            Parameter("tile_grid_size","Grid",  "int",   8,   2,   16,   2),
        ],
    ),

    # ── Spatial Filter ────────────────────────────────────────────────
    AlgorithmModel(
        id="simple_average", name="Simple Average", category="Spatial Filter",
        abbreviation="AVG",
        description="Box blur — uniform kernel, suppresses noise.",
        has_kernel=True,
        has_padding = True,
        parameters=[
            Parameter("kernel_size", "Kernel", "kernel", 3, choices=KS),
        ],
    ),
    AlgorithmModel(
        id="weighted_average", name="Weighted Average", category="Spatial Filter",
        abbreviation="WA",
        description="Gaussian-weighted blur — smooth roll-off from center.",
        has_kernel=True,
        has_padding = True,
        parameters=[
            Parameter("kernel_size", "Kernel", "kernel", 3, choices=KS),
        ],
    ),
    AlgorithmModel(
        id="median", name="Median Filter", category="Spatial Filter",
        abbreviation="MED",
        description="Replaces each pixel with the median of its neighbourhood.",
        has_kernel=True,
        has_padding = True,
        parameters=[
            Parameter("kernel_size", "Kernel", "kernel", 3, choices=KS),
        ],
    ),
    AlgorithmModel(
        id="gradient_1st_deriv", name="1st Derivative (Gradient)", category="Spatial Filter",
        abbreviation="GRD",
        description="Sobel magnitude — highlights edges.",
        has_kernel=True,
    ),
        AlgorithmModel(
        id="laplacian_2nd_deriv", name="2nd Derivative (Laplacian)",
        category="Spatial Filter", abbreviation="LAP",
        description="Fixed 3×3 Laplacian kernel. Detects edges and fine detail.",
        has_kernel=False,       
        parameters=[
            Parameter(
                name="variant",
                display_name="Variant",
                type="choice",
                default="Standard (−4)",
                choices=["Standard (−4)", "Diagonal (−8)", "Enhancement (+5)"],
            ),
        ],
    ),
    AlgorithmModel(
        id="sobel", name="Sobel Filter", category="Spatial Filter",
        abbreviation="SOB",
        description="Directional Sobel edge detection.",
        has_kernel=True,
        parameters=[
            Parameter("dx",    "dx",     "int",    1, 0, 1, 1),
            Parameter("dy",    "dy",     "int",    1, 0, 1, 1),
            Parameter("ksize", "Kernel", "kernel", 3, choices=KS),
        ],
    ),

    # ── Frequency Domain ──────────────────────────────────────────────
    AlgorithmModel(
        id="ideal_lpf", name="Ideal LPF", category="Frequency Domain",
        abbreviation="ILP",
        description="Hard cutoff low-pass filter in frequency domain.",
        parameters=[Parameter("d0", "D0", "float", 30.0, 1.0, 200.0, 1.0)],
    ),
    AlgorithmModel(
        id="butterworth_lpf", name="Butterworth LPF", category="Frequency Domain",
        abbreviation="BLP",
        description="Smooth low-pass with adjustable order.",
        parameters=[
            Parameter("d0",    "D0",    "float", 30.0, 1.0, 200.0, 1.0),
            Parameter("order", "Order", "int",   2,    1,   10,    1),
        ],
    ),
    AlgorithmModel(
        id="gaussian_lpf", name="Gaussian LPF", category="Frequency Domain",
        abbreviation="GLP",
        description="Gaussian-shaped low-pass frequency filter.",
        parameters=[Parameter("d0", "D0", "float", 30.0, 1.0, 200.0, 1.0)],
    ),
    AlgorithmModel(
        id="ideal_hpf", name="Ideal HPF", category="Frequency Domain",
        abbreviation="IHP",
        description="Hard cutoff high-pass filter.",
        parameters=[Parameter("d0", "D0", "float", 30.0, 1.0, 200.0, 1.0)],
    ),
    AlgorithmModel(
        id="butterworth_hpf", name="Butterworth HPF", category="Frequency Domain",
        abbreviation="BHP",
        description="Smooth high-pass with adjustable order.",
        parameters=[
            Parameter("d0",    "D0",    "float", 30.0, 1.0, 200.0, 1.0),
            Parameter("order", "Order", "int",   2,    1,   10,    1),
        ],
    ),
    AlgorithmModel(
        id="gaussian_hpf", name="Gaussian HPF", category="Frequency Domain",
        abbreviation="GHP",
        description="Gaussian-shaped high-pass frequency filter.",
        parameters=[Parameter("d0", "D0", "float", 30.0, 1.0, 200.0, 1.0)],
    ),
    AlgorithmModel(
        id="band_reject_ideal", name="Band-Reject Ideal", category="Frequency Domain",
        abbreviation="BRI",
        description="Ideal notch/band-reject filter.",
        parameters=[
            Parameter("d0", "D0", "float", 30.0, 1.0, 200.0, 1.0),
            Parameter("w",  "W",  "float", 10.0, 1.0, 100.0, 1.0),
        ],
    ),
    AlgorithmModel(
        id="band_reject_butterworth", name="Band-Reject Butterworth",
        category="Frequency Domain", abbreviation="BRB",
        description="Butterworth band-reject filter.",
        parameters=[
            Parameter("d0",    "D0",    "float", 30.0, 1.0, 200.0, 1.0),
            Parameter("w",     "W",     "float", 10.0, 1.0, 100.0, 1.0),
            Parameter("order", "Order", "int",   2,    1,   10,    1),
        ],
    ),
    AlgorithmModel(
        id="band_reject_gaussian", name="Band-Reject Gaussian",
        category="Frequency Domain", abbreviation="BRG",
        description="Gaussian band-reject filter.",
        parameters=[
            Parameter("d0", "D0", "float", 30.0, 1.0, 200.0, 1.0),
            Parameter("w",  "W",  "float", 10.0, 1.0, 100.0, 1.0),
        ],
    ),

    # ── Restoration ───────────────────────────────────────────────────
    AlgorithmModel(
        id="arithmetic_mean", name="Arithmetic Mean", category="Restoration",
        abbreviation="AM",
        description="Simple averaging filter for noise reduction.",
        has_kernel=True,
        parameters=[Parameter("kernel_size", "Kernel", "kernel", 3, choices=KS)],
    ),
    AlgorithmModel(
        id="geometric_mean", name="Geometric Mean", category="Restoration",
        abbreviation="GM",
        description="Product-root mean — less blurring than arithmetic.",
        parameters=[Parameter("kernel_size", "Kernel", "kernel", 3, choices=KS)],
    ),
    AlgorithmModel(
        id="harmonic_mean", name="Harmonic Mean", category="Restoration",
        abbreviation="HM",
        description="Good for salt noise; poor on pepper noise.",
        has_kernel=True,
        parameters=[Parameter("kernel_size", "Kernel", "kernel", 3, choices=KS)],
    ),
    AlgorithmModel(
        id="contraharmonic_mean", name="Contraharmonic Mean", category="Restoration",
        abbreviation="CHM",
        description="Q>0 removes pepper; Q<0 removes salt noise.",
        has_kernel=True,
        parameters=[
            Parameter("kernel_size", "Kernel", "kernel", 3, choices=KS),
            Parameter("q",           "Q",      "float",  1.5, -5.0, 5.0, 0.5),
        ],
    ),
    AlgorithmModel(
        id="max_filter", name="Max Filter", category="Restoration",
        abbreviation="MAX",
        description="Replaces each pixel with neighbourhood maximum.",
        has_kernel=True,
        parameters=[Parameter("kernel_size", "Kernel", "kernel", 3, choices=KS)],
    ),
    AlgorithmModel(
        id="min_filter", name="Min Filter", category="Restoration",
        abbreviation="MIN",
        description="Replaces each pixel with neighbourhood minimum.",
        has_kernel=True,
        parameters=[Parameter("kernel_size", "Kernel", "kernel", 3, choices=KS)],
    ),
    AlgorithmModel(
        id="midpoint_filter", name="Midpoint Filter", category="Restoration",
        abbreviation="MID",
        description="Average of max and min in neighbourhood.",
        has_kernel=True,
        parameters=[Parameter("kernel_size", "Kernel", "kernel", 3, choices=KS)],
    ),
    AlgorithmModel(
        id="alpha_trimmed_mean", name="Alpha-Trimmed Mean", category="Restoration",
        abbreviation="ATM",
        description="Mean after trimming d/2 extremes from each end.",
        has_kernel=True,
        parameters=[
            Parameter("kernel_size", "Kernel", "kernel", 3, choices=KS),
            Parameter("d",           "d",      "int",    2, 0, 8, 2),
        ],
    ),
    AlgorithmModel(
        id="adaptive_median", name="Adaptive Median", category="Restoration",
        abbreviation="ADM",
        description="Grows window until median is an impulse-free value.",
        has_kernel=True,
        parameters=[
            Parameter("max_size", "Max K", "kernel", 7,
                      choices=["3", "5", "7", "9"]),
        ],
    ),

    # ── Segmentation ──────────────────────────────────────────────────
    AlgorithmModel(
        id="point_detection", name="Point Detection", category="Segmentation",
        abbreviation="PD",
        description="Laplacian-based isolated point detector.",
        has_kernel=True,
        parameters=[Parameter("threshold", "T", "int", 128, 0, 255, 1)],
    ),
    AlgorithmModel(
        id="line_detection", name="Line Detection", category="Segmentation",
        abbreviation="LD",
        description="Directional line detector using preset kernels.",
        has_kernel=True,
        parameters=[
            Parameter("direction",  "Dir", "choice", "Horizontal",
                      choices=["Horizontal", "Vertical", "+45°", "-45°"]),
            Parameter("threshold",  "T",   "int",    128, 0, 255, 1),
        ],
    ),
    AlgorithmModel(
        id="edge_prewitt", name="Prewitt Edge", category="Segmentation",
        abbreviation="PRW",
        description="Prewitt operator for edge detection.",
        has_kernel=True,
    ),
    AlgorithmModel(
        id="edge_roberts", name="Roberts Edge", category="Segmentation",
        abbreviation="ROB",
        description="Roberts cross-gradient operator.",
        has_kernel=True,
    ),
    AlgorithmModel(
        id="edge_sobel_seg", name="Sobel Edge", category="Segmentation",
        abbreviation="SBL",
        description="Sobel magnitude followed by binary threshold.",
        has_kernel=True,
        parameters=[Parameter("threshold", "T", "int", 128, 0, 255, 1)],
    ),
    AlgorithmModel(
        id="log_seg", name="LoG Edge", category="Segmentation",
        abbreviation="LOG",
        description="Laplacian-of-Gaussian edge detector.",
        has_kernel=True,
        parameters=[
            Parameter("ksize", "Kernel", "kernel", 5, choices=KS),
            Parameter("sigma", "σ",      "float",  1.0, 0.1, 5.0, 0.1),
        ],
    ),
    AlgorithmModel(
        id="global_thresh", name="Global Threshold", category="Segmentation",
        abbreviation="GT",
        description="Iterative optimal global threshold estimation.",
    ),
    AlgorithmModel(
        id="adaptive_thresh", name="Adaptive Threshold", category="Segmentation",
        abbreviation="AT",
        description="Per-region threshold based on local mean.",
        has_kernel=True,
        parameters=[
            Parameter("block_size", "Block", "kernel", 11,
                      choices=["3", "5", "9", "11", "15", "21"]),
            Parameter("c",          "C",     "int",    2, 0, 20, 1),
        ],
    ),

    # ── Noise ─────────────────────────────────────────────────────────
    AlgorithmModel(
        id="noise_salt_pepper", name="Salt & Pepper", category="Noise",
        abbreviation="S&P",
        description="Randomly sets pixels to 255 (salt) or 0 (pepper).",
        parameters=[
            Parameter("amount",      "Amount",      "float", 0.05, 0.001, 0.5,  0.005),
            Parameter("salt_ratio",  "Salt Ratio",  "float", 0.5,  0.0,   1.0,  0.05),
        ],
    ),
    AlgorithmModel(
        id="noise_gaussian", name="Gaussian Noise", category="Noise",
        abbreviation="GSN",
        description="Additive Gaussian (normal) random noise.",
        parameters=[
            Parameter("mean",  "Mean",   "float", 0.0,  -50.0, 50.0,  1.0),
            Parameter("sigma", "Sigma",  "float", 25.0,  1.0,  150.0, 1.0),
        ],
    ),
    AlgorithmModel(
        id="noise_speckle", name="Speckle Noise", category="Noise",
        abbreviation="SPK",
        description="Multiplicative noise: out = in + in · N(0, σ²).",
        parameters=[
            Parameter("sigma", "Sigma", "float", 0.2, 0.01, 1.0, 0.01),
        ],
    ),

    # ── Arithmetic ────────────────────────────────────────────────────
    AlgorithmModel(
        id="subtraction", name="Image Subtraction", category="Arithmetic",
        abbreviation="SUB",
        description="Subtract original or previous step — useful for unsharp masking.",
        parameters=[
            Parameter("target", "Subtract",  "choice", "Original",
                      choices=["Original", "Previous Step"]),
            Parameter("scale",  "Scale",     "float",  1.0, 0.0, 3.0, 0.05),
            Parameter("mode",   "Mode",      "choice", "Absolute",
                      choices=["Absolute", "Clipped", "Scaled"]),
        ],
    ),
]