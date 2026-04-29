"""
filters.py — All image processing algorithms for DIP Studio
Supports padding modes: zero | replicate | wrap | truncate
"""
import cv2
import numpy as np
from typing import Dict, Any


# ──────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────

def _odd(k: int) -> int:
    """Force k to a positive odd integer (safety net — UI enforces this too)."""
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1


def _cv2_border(mode: str) -> int:
    """Map padding mode string → cv2 border constant."""
    return {
        "zero":      cv2.BORDER_CONSTANT,
        "replicate": cv2.BORDER_REPLICATE,
        "wrap":      cv2.BORDER_WRAP,
        "truncate":  cv2.BORDER_CONSTANT,   # handled manually below
    }.get(mode, cv2.BORDER_CONSTANT)


def _np_pad_mode(mode: str) -> str:
    """Map padding mode string → numpy pad mode."""
    return {
        "zero":      "constant",
        "replicate": "edge",
        "wrap":      "wrap",
        "truncate":  "constant",
    }.get(mode, "constant")


def _restore_border(result: np.ndarray, original: np.ndarray, pad: int) -> np.ndarray:
    """Copy original pixel values into the `pad`-wide border (for omit/truncate borders)."""
    if pad <= 0:
        return result
    out = result.copy().astype(np.float32)
    src = original.astype(np.float32)
    out[:pad, :]  = src[:pad, :]
    out[-pad:, :] = src[-pad:, :]
    out[:, :pad]  = src[:, :pad]
    out[:, -pad:] = src[:, -pad:]
    return out


def _filter2d_padded(image: np.ndarray, kernel: np.ndarray,
                     mode: str = "zero") -> np.ndarray:
    """
    Apply a 2-D linear kernel with the selected padding strategy.

    zero      – pad with 0 s (BORDER_CONSTANT)
    replicate – mirror edge pixels outward (BORDER_REPLICATE)
    wrap      – tile the image (BORDER_WRAP)
    truncate  – zero-pad then re-weight each output pixel so partial
                border overlaps are properly normalised (no dark rim)
    """
    img_f  = image.astype(np.float32)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    if mode == "replicate":
        out = cv2.filter2D(img_f, cv2.CV_32F, kernel,
                           borderType=cv2.BORDER_REPLICATE)

    elif mode == "wrap":
        out = cv2.filter2D(img_f, cv2.CV_32F, kernel,
                           borderType=cv2.BORDER_WRAP)

    elif mode == "truncate":
        raw  = cv2.filter2D(img_f, cv2.CV_32F, kernel,
                            borderType=cv2.BORDER_CONSTANT)
        ones = np.ones_like(img_f)
        k_abs = np.abs(kernel).astype(np.float32)
        wmap  = cv2.filter2D(ones, cv2.CV_32F, k_abs,
                             borderType=cv2.BORDER_CONSTANT)
        k_sum = float(np.sum(k_abs)) or 1.0
        wmap  = np.where(wmap > 1e-6, wmap, k_sum)
        out   = raw * (k_sum / wmap)

    else:  # "zero" (default)
        out = cv2.filter2D(img_f, cv2.CV_32F, kernel,
                           borderType=cv2.BORDER_CONSTANT)

    return np.clip(out, 0, 255).astype(np.uint8)


def _morph_padded(image: np.ndarray, struct: np.ndarray,
                  op: str, mode: str) -> np.ndarray:
    """Morphological dilate/erode with padding mode support."""
    btype = _cv2_border(mode)
    if op == "dilate":
        return cv2.dilate(image, struct, borderType=btype)
    return cv2.erode(image, struct, borderType=btype)


# ──────────────────────────────────────────────────────────────────────
# Main dispatcher
# ──────────────────────────────────────────────────────────────────────

def apply_filter(image: np.ndarray, alg_id: str, params: Dict[str, Any]) -> np.ndarray:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_float = image.astype(np.float32)
    pad = params.get("padding_mode", "zero")

    # ── Point Processing ──────────────────────────────────────────────
    if alg_id == "negative":
        return 255 - image

    elif alg_id == "log":
        c = params.get("c", 1.0)
        return np.clip(c * np.log1p(img_float), 0, 255).astype(np.uint8)

    elif alg_id == "gamma":
        c     = params.get("c", 1.0)
        gamma = params.get("gamma", 1.0)
        return np.clip(c * np.power(img_float, gamma), 0, 255).astype(np.uint8)

    elif alg_id == "thresholding":
        _, res = cv2.threshold(image, params.get("threshold", 128), 255, cv2.THRESH_BINARY)
        return res

    elif alg_id == "gray_slicing":
        low  = params.get("low",  100)
        high = params.get("high", 200)
        preserve = params.get("preserve_bg", True)
        res  = np.zeros_like(image)
        mask = (image >= low) & (image <= high)
        res[mask] = 255
        if preserve:
            res[~mask] = image[~mask]
        return res

    elif alg_id == "bit_plane":
        bit = params.get("bit", 7)
        return ((image >> bit) & 1).astype(np.uint8) * 255

    # ── Histogram ─────────────────────────────────────────────────────
    elif alg_id == "contrast_stretching":
        r1, s1 = params.get("r1",  50), params.get("s1",  20)
        r2, s2 = params.get("r2", 200), params.get("s2", 230)
        res   = np.zeros_like(img_float)
        m1    = s1 / r1 if r1 > 0 else 0.0
        mask1 = img_float <= r1
        res[mask1] = img_float[mask1] * m1
        m2    = (s2 - s1) / (r2 - r1) if r2 > r1 else 0.0
        mask2 = (img_float > r1) & (img_float <= r2)
        res[mask2] = s1 + (img_float[mask2] - r1) * m2
        m3    = (255 - s2) / (255 - r2) if r2 < 255 else 0.0
        mask3 = img_float > r2
        res[mask3] = s2 + (img_float[mask3] - r2) * m3
        return np.clip(res, 0, 255).astype(np.uint8)

    elif alg_id == "hist_equalization":
        return cv2.equalizeHist(image)

    elif alg_id == "clahe":
        clip = params.get("clip_limit", 2.0)
        grid = params.get("tile_grid_size", 8)
        return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(image)

    # ── Spatial Filtering ─────────────────────────────────────────────
    elif alg_id == "simple_average":
        k = _odd(params.get("kernel_size", 3))
        kernel = np.ones((k, k), np.float32) / (k * k)
        return _filter2d_padded(image, kernel, pad)

    elif alg_id == "weighted_average":
        k  = _odd(params.get("kernel_size", 3))
        g  = cv2.getGaussianKernel(k, 0)
        kernel = (g @ g.T).astype(np.float32)
        kernel /= kernel.sum()
        return _filter2d_padded(image, kernel, pad)

    elif alg_id == "median":
        k     = _odd(params.get("kernel_size", 3))
        ph    = k // 2
        btype = _cv2_border(pad)
        padded = cv2.copyMakeBorder(image, ph, ph, ph, ph, btype)
        result = cv2.medianBlur(padded, k)
        result = result[ph:ph + image.shape[0], ph:ph + image.shape[1]]
        return result

    elif alg_id == "gradient_1st_deriv":
        bt = _cv2_border(pad)
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, borderType=bt)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, borderType=bt)
        mag = cv2.magnitude(gx, gy)
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    elif alg_id == "laplacian_2nd_deriv":
        variant = params.get("variant", "Standard (−4)")

        kernels = {
            "Standard (−4)":    np.array([[ 0,  1,  0],
                                        [ 1, -4,  1],
                                        [ 0,  1,  0]], np.float32),

            "Diagonal (−8)":    np.array([[ 1,  1,  1],
                                        [ 1, -8,  1],
                                        [ 1,  1,  1]], np.float32),

            "Enhancement (+5)": np.array([[ 0, -1,  0],
                                        [-1,  5, -1],
                                        [ 0, -1,  0]], np.float32),
        }

        k = kernels.get(variant, kernels["Standard (−4)"])
        result = cv2.filter2D(image, cv2.CV_32F, k)

        # Enhancement variant produces a directly usable sharpened image
        if "Enhancement" in variant:
            return np.clip(result, 0, 255).astype(np.uint8)

        # Standard and Diagonal produce an edge map — take absolute and normalise
        return cv2.normalize(np.abs(result), None, 0, 255,
                            cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    elif alg_id == "sobel":
        dx = params.get("dx", 1)
        dy = params.get("dy", 1)
        k  = _odd(params.get("ksize", 3))
        if dx == 0 and dy == 0:
            return image
        bt    = _cv2_border(pad)
        sobel = cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize=k, borderType=bt)
        return cv2.normalize(np.abs(sobel), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # ── Frequency Domain ──────────────────────────────────────────────
    elif alg_id in ("ideal_lpf", "butterworth_lpf", "gaussian_lpf",
                    "ideal_hpf", "butterworth_hpf", "gaussian_hpf",
                    "band_reject_ideal", "band_reject_butterworth", "band_reject_gaussian"):

        d0   = params.get("d0", 30.0)
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        dft       = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        iy, ix = np.mgrid[0:rows, 0:cols]
        dist   = np.sqrt((ix - ccol) ** 2 + (iy - crow) ** 2).astype(np.float32)

        mask = np.zeros((rows, cols, 2), np.float32)

        if alg_id == "ideal_lpf":
            mask[dist <= d0] = 1
        elif alg_id == "butterworth_lpf":
            n = params.get("order", 2)
            h = 1 / (1 + (dist / (d0 + 1e-8)) ** (2 * n))
            mask[:, :, 0] = h;  mask[:, :, 1] = h
        elif alg_id == "gaussian_lpf":
            h = np.exp(-(dist ** 2) / (2 * d0 ** 2))
            mask[:, :, 0] = h;  mask[:, :, 1] = h
        elif alg_id == "ideal_hpf":
            mask[dist > d0] = 1
        elif alg_id == "butterworth_hpf":
            n = params.get("order", 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                h = 1 / (1 + (d0 / (dist + 1e-8)) ** (2 * n))
                h[dist == 0] = 0
            mask[:, :, 0] = h;  mask[:, :, 1] = h
        elif alg_id == "gaussian_hpf":
            h = 1 - np.exp(-(dist ** 2) / (2 * d0 ** 2))
            mask[:, :, 0] = h;  mask[:, :, 1] = h
        elif "band_reject" in alg_id:
            w = params.get("w", 10.0)
            if alg_id == "band_reject_ideal":
                mask[:] = 1
                bmask = (dist >= d0 - w / 2) & (dist <= d0 + w / 2)
                mask[bmask] = 0
            elif alg_id == "band_reject_butterworth":
                n = params.get("order", 2)
                with np.errstate(divide="ignore", invalid="ignore"):
                    denom = dist ** 2 - d0 ** 2
                    h = 1 / (1 + ((dist * w) / (denom + 1e-8)) ** (2 * n))
                    h[denom == 0] = 0
                mask[:, :, 0] = h;  mask[:, :, 1] = h
            elif alg_id == "band_reject_gaussian":
                h = 1 - np.exp(-((dist ** 2 - d0 ** 2) ** 2)
                               / (dist ** 2 * w ** 2 + 1e-8))
                mask[:, :, 0] = h;  mask[:, :, 1] = h

        fshift   = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
        return img_back.astype(np.uint8)

    # ── Restoration ───────────────────────────────────────────────────
    elif alg_id == "arithmetic_mean":
        k = _odd(params.get("kernel_size", 3))
        kernel = np.ones((k, k), np.float32) / (k * k)
        return _filter2d_padded(image, kernel, pad)

    elif alg_id == "geometric_mean":
        k      = _odd(params.get("kernel_size", 3))
        kernel = np.ones((k, k), np.float32) / (k * k)
        img_f  = img_float + 1.0
        log_f  = np.log(img_f)
        log_res = _filter2d_padded(
            np.clip(log_f / log_f.max() * 255, 0, 255).astype(np.uint8),
            kernel, pad).astype(np.float32) / 255.0 * log_f.max()
        return np.clip(np.exp(log_res) - 1, 0, 255).astype(np.uint8)

    elif alg_id == "harmonic_mean":
        k      = _odd(params.get("kernel_size", 3))
        kernel = np.ones((k, k), np.float32)
        img_f  = img_float.copy()
        img_f[img_f == 0] = 1e-5
        inv    = 1.0 / img_f
        bt     = _cv2_border(pad)
        inv_sum = cv2.filter2D(inv, cv2.CV_32F, kernel, borderType=bt)
        res    = (k * k) / (inv_sum + 1e-5)
        return np.clip(res, 0, 255).astype(np.uint8)

    elif alg_id == "contraharmonic_mean":
        k      = _odd(params.get("kernel_size", 3))
        q      = params.get("q", 1.5)
        kernel = np.ones((k, k), np.float32)
        bt     = _cv2_border(pad)
        img_f  = img_float + 1e-5
        num = cv2.filter2D(np.power(img_f, q + 1).astype(np.float32),
                           cv2.CV_32F, kernel, borderType=bt)
        den = cv2.filter2D(np.power(img_f, q).astype(np.float32),
                           cv2.CV_32F, kernel, borderType=bt)
        return np.clip(num / (den + 1e-5), 0, 255).astype(np.uint8)

    elif alg_id == "max_filter":
        k = _odd(params.get("kernel_size", 3))
        struct = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        return _morph_padded(image, struct, "dilate", pad)

    elif alg_id == "min_filter":
        k = _odd(params.get("kernel_size", 3))
        struct = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        return _morph_padded(image, struct, "erode", pad)

    elif alg_id == "midpoint_filter":
        k      = _odd(params.get("kernel_size", 3))
        struct = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        maxf   = _morph_padded(image, struct, "dilate", pad).astype(np.float32)
        minf   = _morph_padded(image, struct, "erode",  pad).astype(np.float32)
        return ((maxf + minf) / 2).astype(np.uint8)

    elif alg_id == "alpha_trimmed_mean":
        k        = _odd(params.get("kernel_size", 3))
        d        = int(params.get("d", 2))
        pad_size = k // 2
        np_mode  = _np_pad_mode(pad)
        padded   = np.pad(image.astype(np.float32), pad_size, mode=np_mode)
        res      = np.zeros_like(img_float)
        h, w     = image.shape
        for i in range(h):
            for j in range(w):
                window = padded[i:i + k, j:j + k].flatten()
                window.sort()
                trim = d // 2
                if 0 < trim < len(window) // 2:
                    window = window[trim:-trim]
                res[i, j] = window.mean()
        return np.clip(res, 0, 255).astype(np.uint8)

    elif alg_id == "adaptive_median":
        max_size = _odd(params.get("max_size", 7))
        np_mode  = _np_pad_mode(pad)
        pad_max  = max_size // 2
        padded   = np.pad(image, pad_max, mode=np_mode)
        res      = image.copy()
        h, w     = image.shape
        for i in range(h):
            for j in range(w):
                for k in range(3, max_size + 2, 2):
                    pk     = k // 2
                    window = padded[i + pad_max - pk: i + pad_max + pk + 1,
                                    j + pad_max - pk: j + pad_max + pk + 1]
                    z_min  = int(window.min())
                    z_max  = int(window.max())
                    z_med  = float(np.median(window))
                    z_xy   = int(image[i, j])
                    if z_min < z_med < z_max:
                        res[i, j] = z_xy if z_min < z_xy < z_max else int(z_med)
                        break
                    elif k >= max_size:
                        res[i, j] = int(z_med)
        return res.astype(np.uint8)

    # ── Segmentation ──────────────────────────────────────────────────
    elif alg_id == "point_detection":
        thresh = params.get("threshold", 128)
        kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], np.float32)
        result = _filter2d_padded(image, kernel, pad)
        _, res = cv2.threshold(result, thresh, 255, cv2.THRESH_BINARY)
        return res

    elif alg_id == "line_detection":
        direction = params.get("direction", "Horizontal")
        thresh    = params.get("threshold", 128)
        kernels   = {
            "Horizontal": np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]], np.float32),
            "Vertical":   np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]], np.float32),
            "+45°":       np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]], np.float32),
            "-45°":       np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]], np.float32),
        }
        result = _filter2d_padded(image, kernels.get(direction, kernels["Horizontal"]), pad)
        _, res = cv2.threshold(result, thresh, 255, cv2.THRESH_BINARY)
        return res

    elif alg_id == "edge_prewitt":
        kx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], np.float32)
        ky = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], np.float32)
        gx = _filter2d_padded(image, kx, pad).astype(np.int32)
        gy = _filter2d_padded(image, ky, pad).astype(np.int32)
        return np.clip(gx + gy, 0, 255).astype(np.uint8)

    elif alg_id == "edge_roberts":
        kx = np.array([[1,0],[0,-1]], np.float32)
        ky = np.array([[0,1],[-1,0]], np.float32)
        gx = _filter2d_padded(image, kx, pad)
        gy = _filter2d_padded(image, ky, pad)
        return cv2.addWeighted(gx, 0.5, gy, 0.5, 0)

    elif alg_id == "edge_sobel_seg":
        thresh = params.get("threshold", 128)
        bt     = _cv2_border(pad)
        gx     = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, borderType=bt)
        gy     = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, borderType=bt)
        mag    = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255,
                               cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, res = cv2.threshold(mag, thresh, 255, cv2.THRESH_BINARY)
        return res

    elif alg_id == "log_seg":
        k     = _odd(params.get("ksize", 5))
        sigma = params.get("sigma", 1.0)
        bt    = _cv2_border(pad)
        blur  = cv2.GaussianBlur(image, (k, k), sigma, borderType=bt)
        lap   = cv2.Laplacian(blur, cv2.CV_64F, borderType=bt)
        return cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    elif alg_id == "global_thresh":
        T = float(image.mean())
        for _ in range(100):
            m1    = float(image[image > T].mean())  if (image > T).any()  else 0.0
            m2    = float(image[image <= T].mean()) if (image <= T).any() else 0.0
            new_T = (m1 + m2) / 2
            if abs(T - new_T) < 0.5:
                break
            T = new_T
        _, res = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)
        return res

    elif alg_id == "adaptive_thresh":
        block = _odd(params.get("block_size", 11))
        c     = params.get("c", 2)
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, c)

    # ── Noise ─────────────────────────────────────────────────────────
    elif alg_id == "noise_salt_pepper":
        amount     = float(params.get("amount",     0.05))
        salt_ratio = float(params.get("salt_ratio", 0.5))
        out        = image.copy()
        h, w       = image.shape
        n_total    = int(amount * h * w)
        n_salt     = int(n_total * salt_ratio)
        n_pepper   = n_total - n_salt

        # Salt
        r = np.random.randint(0, h, n_salt)
        c_ = np.random.randint(0, w, n_salt)
        out[r, c_] = 255

        # Pepper
        r = np.random.randint(0, h, n_pepper)
        c_ = np.random.randint(0, w, n_pepper)
        out[r, c_] = 0

        return out

    elif alg_id == "noise_gaussian":
        mean  = float(params.get("mean",  0.0))
        sigma = float(params.get("sigma", 25.0))
        noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
        return np.clip(img_float + noise, 0, 255).astype(np.uint8)

    elif alg_id == "noise_speckle":
        sigma = float(params.get("sigma", 0.2))
        noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
        return np.clip(img_float + img_float * noise, 0, 255).astype(np.uint8)

    # ── Arithmetic ────────────────────────────────────────────────────
    elif alg_id == "subtraction":
        target_img = params.get("_target_image")
        scale      = float(params.get("scale", 1.0))
        mode       = params.get("mode", "Absolute")
        if target_img is None:
            return image
        diff = (img_float - target_img.astype(np.float32)) * scale
        if mode == "Absolute":
            result = np.abs(diff)
        elif mode == "Clipped":
            result = np.clip(diff, 0, 255)
        else:
            result = diff - diff.min()
            mx = result.max()
            result = result / mx * 255.0 if mx > 0 else result
        return np.clip(result, 0, 255).astype(np.uint8)

    return image