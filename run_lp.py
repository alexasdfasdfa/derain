"""Rain streak removal using layer priors (Li et al., CVPR 2016).

Usage:
    python run_lp.py
    python run_lp.py --input path\\to\\image.jpg --output path\\to\\dir

Default input is LPDerain-main\\case1.jpg. The image is resized by 1/2,
matching main.m. Outputs in the output directory:
    derained.png   background in color
    rain.png       rain layer, shifted by 0.5 for display
    compare.png    input and result side by side

GMM_Decomp.p cannot be run. Patch order, the background weights, and
aprxMAPGMM follow the readable MATLAB. The rain direction comes from the
rain GMM. The component constant along that direction is removed, then
aprxMAPGMM returns the natural-image detail.
"""

import argparse
import os
import time

import numpy as np
from PIL import Image
from scipy import io
from scipy.ndimage import gaussian_filter, median_filter, minimum_filter, uniform_filter
from sklearn.mixture import GaussianMixture


PATCH = 8
RAIN_COMPONENTS = 20
RAIN_WINDOW = 64
STREAK_PASSES = 2
DETAIL_NOISE = 0.01


def rgb_to_ycbcr(image):
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    y = 16.0 + 65.481 * red + 128.553 * green + 24.966 * blue
    cb = 128.0 - 37.797 * red - 74.203 * green + 112.0 * blue
    cr = 128.0 + 112.0 * red - 93.786 * green - 18.214 * blue
    return y / 255.0, cb, cr


def ycbcr_to_rgb(y, cb, cr):
    y = y * 255.0
    r = 1.164 * (y - 16.0) + 1.596 * (cr - 128.0)
    g = 1.164 * (y - 16.0) - 0.392 * (cb - 128.0) - 0.813 * (cr - 128.0)
    b = 1.164 * (y - 16.0) + 2.017 * (cb - 128.0)
    return np.clip(np.stack([r, g, b], axis=2) / 255.0, 0.0, 1.0)


MAT_GMM = (
    r"c:\Users\17237\Desktop\人工智能数学基础\LPDerain-main"
    r"\epllcode\GSModel_8x8_200_2M_noDC_zeromean.mat"
)
NPZ_GMM = r"c:\Users\17237\Desktop\人工智能数学基础\lp_derain\gsmodel_8x8_200.npz"


def convert_background_gmm(mat_path=MAT_GMM, npz_path=NPZ_GMM):
    """Write the Zoran-Weiss background GMM as a NumPy archive.

    MATLAB stores covs as 64 x 64 x 200 and means as 64 x 200, both in
    column-major patch order. The archive keeps that pixel order.
    """
    model = io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["GS"]
    covs = np.transpose(np.asarray(model.covs, dtype=np.float64), (2, 0, 1))
    means = np.asarray(model.means, dtype=np.float64)
    weights = np.asarray(model.mixweights, dtype=np.float64)
    weights = weights / weights.sum()
    np.savez_compressed(
        npz_path,
        covs=covs,
        means=means,
        weights=weights,
        dim=np.int64(model.dim),
        nmodels=np.int64(model.nmodels),
    )
    return npz_path


def load_background_gmm(path=NPZ_GMM):
    if not os.path.exists(path):
        convert_background_gmm(npz_path=path)
    data = np.load(path)
    means = np.asarray(data["means"], dtype=np.float64)
    return {
        "covs": np.asarray(data["covs"], dtype=np.float64),
        "means": means,
        "weights": np.asarray(data["weights"], dtype=np.float64),
        "count": int(data["nmodels"]),
    }


def im2patches(image):
    """64 x N patches. Pixels inside a patch follow MATLAB column order."""
    view = np.lib.stride_tricks.sliding_window_view(image, (PATCH, PATCH))
    # view[..., row, col]. Move col before row so a C-order flatten is column-major.
    ordered = np.transpose(view, (0, 1, 3, 2))
    rows, cols = ordered.shape[:2]
    return np.ascontiguousarray(ordered.reshape(rows * cols, PATCH * PATCH).T)


def merge_patches(patches, shape):
    """Sum patches onto the image. Inverse of im2patches, as merge_patches.m."""
    height, width = shape
    rows = height - PATCH + 1
    cols = width - PATCH + 1
    # Last axis of the 8x8 is the patch row, the one before it is the patch column.
    block = patches.T.reshape(rows, cols, PATCH, PATCH)
    out = np.zeros(shape, dtype=np.float64)
    for col in range(PATCH):
        for row in range(PATCH):
            out[row : row + rows, col : col + cols] += block[:, :, col, row]
    return out


def lowest_variance_window(image, size):
    mean = uniform_filter(image, size=size, mode="nearest")
    second = uniform_filter(image * image, size=size, mode="nearest")
    variance = second - mean * mean
    half = size // 2
    variance[:half, :] = np.inf
    variance[-half:, :] = np.inf
    variance[:, :half] = np.inf
    variance[:, -half:] = np.inf
    center = np.unravel_index(np.argmin(variance), variance.shape)
    row, col = center
    return image[row - half : row - half + size, col - half : col - half + size]


def _rain_samples(region):
    samples = im2patches(region)
    samples = samples - samples.mean(axis=0, keepdims=True)
    keep = samples.var(axis=0, ddof=1) > 1e-5
    return samples[:, keep]


def _rain_region(image, size, components):
    """Lowest-variance window that still contains rain streaks.

    A blank sky can be the flattest region in a storm photo and then has too
    few patches to fit the mixture. In that case, take the flattest window
    that still has enough varying patches.
    """
    region = lowest_variance_window(image, size)
    samples = _rain_samples(region)
    if samples.shape[1] >= components:
        return region, samples
    best = None
    for row in range(0, image.shape[0] - size + 1, 16):
        for col in range(0, image.shape[1] - size + 1, 16):
            candidate = image[row : row + size, col : col + size]
            kept = _rain_samples(candidate)
            if kept.shape[1] < components:
                continue
            score = (float(candidate.var()), -kept.shape[1])
            if best is None or score < best[0]:
                best = (score, candidate, kept)
    if best is None:
        raise ValueError("no window contains enough rain patches")
    return best[1], best[2]


def learn_rain_gmm(image, components=RAIN_COMPONENTS, window=RAIN_WINDOW):
    """onlineGMM.m, with the paper's lowest-variance window in place of ginput."""
    region, samples = _rain_region(image, window, components)
    mixture = GaussianMixture(
        n_components=components,
        covariance_type="full",
        reg_covar=1e-5,
        max_iter=100,
        random_state=0,
    )
    mixture.fit(samples.T)
    covs = mixture.covariances_ + 1e-5 * np.eye(PATCH * PATCH)
    weights = mixture.weights_ / mixture.weights_.sum()
    return {
        "covs": covs,
        "means": mixture.means_.T,
        "weights": weights,
        "count": components,
    }


def aprx_map_gmm(patches, model, noise_var):
    """aprxMAPGMM.m. patches are 64 x N. Returns restored patches, DC included."""
    mean_y = patches.mean(axis=0, keepdims=True)
    centered = patches - mean_y
    count = model["count"]
    scores = np.empty((count, centered.shape[1]), dtype=np.float64)
    eye = np.eye(PATCH * PATCH)
    constant = PATCH * PATCH * np.log(2.0 * np.pi)
    for index in range(count):
        biased = model["covs"][index] + noise_var * eye
        chol = np.linalg.cholesky(biased)
        logdet = constant + 2.0 * np.sum(np.log(np.diag(chol)))
        solved = np.linalg.solve(chol, centered)
        quad = np.sum(solved * solved, axis=0)
        scores[index] = np.log(model["weights"][index]) - 0.5 * (logdet + quad)
    choice = np.argmax(scores, axis=0)
    restored = np.zeros_like(centered)
    for index in range(count):
        selected = np.flatnonzero(choice == index)
        if selected.size == 0:
            continue
        cov = model["covs"][index]
        biased = cov + noise_var * eye
        prior_mean = model["means"][:, index : index + 1]
        rhs = cov @ centered[:, selected] + noise_var * prior_mean
        restored[:, selected] = np.linalg.solve(biased, rhs)
    return restored + mean_y


def _average_map(image, model, noise_var):
    restored = aprx_map_gmm(im2patches(image), model, noise_var)
    total = merge_patches(restored, image.shape)
    cover = merge_patches(np.ones_like(restored), image.shape)
    return total / np.maximum(cover, 1.0)


def _streak_slope(model):
    """Column shift per row of the strongest rain eigenvector."""
    strongest = -1.0
    vector = None
    for index in range(model["count"]):
        values, vectors = np.linalg.eigh(model["covs"][index])
        if values[-1] > strongest:
            strongest = values[-1]
            vector = vectors[:, -1]
    grid = vector.reshape(PATCH, PATCH, order="F")
    columns = np.argmin(grid, axis=1).astype(np.float64)
    slope = np.polyfit(np.arange(PATCH, dtype=np.float64), columns, 1)[0]
    return float(slope)


def _streak_basis(slope):
    """Patterns that are constant along the streak and vary across it.

    A streak shifted by one pixel is a different vector, so the leading
    eigenvector of one rain component misses most of them. Binning the
    coordinate perpendicular to the streak spans every phase inside the patch.
    """
    rows, cols = np.meshgrid(np.arange(PATCH), np.arange(PATCH), indexing="ij")
    bins = np.round((cols - slope * rows) * 2.0) / 2.0
    vectors = []
    for value in np.unique(np.round(bins, 5)):
        mask = (np.round(bins, 5) == np.round(value, 5)).astype(np.float64)
        if mask.sum() < 2:
            continue
        mask -= mask.mean()
        norm = np.linalg.norm(mask)
        if norm > 1e-8:
            vectors.append((mask / norm).reshape(-1, order="F"))
    orthogonal, _ = np.linalg.qr(np.stack(vectors, axis=1))
    return orthogonal


def _basis_size(slope, size):
    """Same streak basis as _streak_basis, on a size x size patch."""
    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    bins = np.round((cols - slope * rows) * 2.0) / 2.0
    vectors = []
    for value in np.unique(np.round(bins, 5)):
        mask = (np.round(bins, 5) == np.round(value, 5)).astype(np.float64)
        if mask.sum() < 2:
            continue
        mask -= mask.mean()
        norm = np.linalg.norm(mask)
        if norm > 1e-8:
            vectors.append((mask / norm).reshape(-1, order="F"))
    orthogonal, _ = np.linalg.qr(np.stack(vectors, axis=1))
    return orthogonal


def _take_patches(image, size, stride):
    view = np.lib.stride_tricks.sliding_window_view(image, (size, size))[::stride, ::stride]
    ordered = np.transpose(view, (0, 1, 3, 2))
    rows, cols = ordered.shape[:2]
    patches = np.ascontiguousarray(ordered.reshape(rows * cols, size * size).T)
    return patches, rows, cols


def _merge_strided(patches, shape, size, stride, rows, cols):
    block = patches.T.reshape(rows, cols, size, size)
    out = np.zeros(shape, dtype=np.float64)
    cover = np.zeros(shape, dtype=np.float64)
    for col in range(size):
        for row in range(size):
            out[row : row + stride * rows : stride, col : col + stride * cols : stride] += block[:, :, col, row]
            cover[row : row + stride * rows : stride, col : col + stride * cols : stride] += 1.0
    return out, cover


def _crossings(coeff):
    """How many times the across-streak profile changes sign.

    A dense curtain crosses many times. One edge crosses once.
    """
    scale = np.std(coeff, axis=0, keepdims=True) + 1e-8
    signed = np.where(np.abs(coeff) > 0.3 * scale, np.sign(coeff), 0.0)
    return np.sum(np.abs(np.diff(signed, axis=0)) > 0, axis=0)


def _profile_crossings(centered, slope, size):
    """Sign changes of the mean across the streak, in perpendicular order.

    QR coefficients are not in that order, so the profile is built from the bins.
    """
    grid = centered.reshape(size, size, centered.shape[1], order="F")
    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    bins = np.round((cols - slope * rows) * 2.0) / 2.0
    profile = []
    for value in np.unique(np.round(bins, 5)):
        mask = np.round(bins, 5) == np.round(value, 5)
        profile.append(grid[mask].mean(axis=0))
    return _crossings(np.stack(profile, axis=0))


def _streak_masks(image, slope, size, stride):
    """Streak component, and how much of each patch it explains."""
    basis = _basis_size(slope, size)
    patches, rows, cols = _take_patches(image, size, stride)
    mean = patches.mean(axis=0, keepdims=True)
    centered = patches - mean
    projection = basis @ (basis.T @ centered)
    energy = np.sum(centered * centered, axis=0)
    captured = np.sum(projection * projection, axis=0)
    ratio = captured / np.maximum(energy, 1e-8)
    crosses = _profile_crossings(centered, slope, size)
    return projection, mean, centered, ratio, crosses, captured, energy, rows, cols


def _rain_slope(image):
    """Direction of the repeated streaks inside the flattest rainy window.

    Strong edges elsewhere have more energy than the rain, so the whole frame
    would vote for a roof or a branch.
    """
    region, _ = _rain_region(image, RAIN_WINDOW, RAIN_COMPONENTS)
    best_slope, best_score = 0.0, -1.0
    for degrees in range(-45, 46, 5):
        slope = float(np.tan(np.deg2rad(degrees)))
        _, _, _, _, crosses, captured, energy, _, _ = _streak_masks(region, slope, 16, 2)
        score = float(np.sum(captured * (crosses >= 2)) / max(float(np.sum(energy)), 1e-8))
        print(f"  angle {degrees:3d}  score {score:.3f}", flush=True)
        if score > best_score:
            best_slope, best_score = slope, score
    return best_slope


def _remove_scale(image, slope, size, stride, min_ratio, min_cross):
    projection, mean, centered, ratio, crosses, _, _, rows, cols = _streak_masks(image, slope, size, stride)
    if min_ratio > 0.0 or min_cross > 0:
        projection[:, (ratio < min_ratio) | (crosses < min_cross)] = 0.0
    merged, cover = _merge_strided(mean + centered - projection, image.shape, size, stride, rows, cols)
    # Pixels no patch covered keep the input.
    removed = image.copy()
    seen = cover > 0.0
    removed[seen] = (merged / np.maximum(cover, 1.0))[seen]
    return removed


def decompose_scales(observed, background_gmm):
    """R at several patch sizes. Larger patches catch longer and thicker streaks.

    The direction is the one that fills the frame with thin lines. A broad edge
    is not thin, so it stays. After the background detail is put back, the
    large scales run once more and take the streaks that detail step returned.
    """
    print("direction search", flush=True)
    slope = _rain_slope(observed)
    print(f"scale slope {slope:.3f}", flush=True)
    background = observed.copy()
    for size, stride, passes, min_ratio, min_cross in (
        (8, 1, 2, 0.15, 2),
        (16, 2, 1, 0.25, 2),
        (32, 2, 1, 0.3, 2),
    ):
        for _ in range(passes):
            background = _remove_scale(background, slope, size, stride, min_ratio, min_cross)
        print(
            f"scale {size}  hgrad {np.mean(np.abs(np.diff(background, axis=1))):.4f}",
            flush=True,
        )
    kept = _average_map(observed - background, background_gmm, DETAIL_NOISE)
    background = np.clip(background + kept, 0.0, 1.0)
    print(f"detail back  hgrad {np.mean(np.abs(np.diff(background, axis=1))):.4f}", flush=True)
    for size, min_ratio in ((16, 0.3), (32, 0.35)):
        background = _remove_scale(background, slope, size, 2, min_ratio, 2)
    background = np.clip(background, 0.0, 1.0)
    print(f"cleanup  hgrad {np.mean(np.abs(np.diff(background, axis=1))):.4f}", flush=True)
    rain = _thin_rain(observed - background, slope)
    background = np.clip(observed - rain, 0.0, 1.0)
    print(f"thin rain  hgrad {np.mean(np.abs(np.diff(background, axis=1))):.4f}", flush=True)
    return background, rain, slope


def _thin_rain(rain, slope, width=9):
    """Return the part of R narrower than the median window.

    A 9-pixel median still ignores a streak a few pixels wide, including the
    faint shoulder, and it follows a shed edge or a leaf. That wide part goes
    back to the background.
    """
    margin = int(np.ceil(abs(slope) * (rain.shape[0] - 1))) + width
    padded = np.pad(rain, ((0, 0), (margin, margin)), mode="edge")
    upright = _shear(padded, slope, inverse=False)
    thin = _shear(upright - median_filter(upright, size=(1, width), mode="nearest"), slope, inverse=True)
    return thin[:, margin:-margin]


def _shear(image, slope, inverse):
    """Integer row shift. A streak of this slope stands upright, or the shift is undone.

    The shift is a whole number of pixels, so a one-pixel streak is not blurred
    and the round trip is exact away from the border.
    """
    out = np.zeros_like(image)
    direction = -1.0 if inverse else 1.0
    for row in range(image.shape[0]):
        shift = int(np.rint(direction * slope * row))
        if shift >= 0:
            out[row, shift:] = image[row, : image.shape[1] - shift]
        else:
            out[row, : image.shape[1] + shift] = image[row, -shift :]
    return out


def _regularize_rain(rain, slope, along=2.0, sparse=0.0):
    """min ||R-R0||^2 + along ||dR/dθ||^2 + sparse ||R||_1.

    A streak is constant along its length, so the derivative term leaves it.
    A piece of background that bends along the streak is sent back.
    A straight edge parallel to the rain is constant too, so this term keeps it.
    """
    upright = _shear(rain, slope, inverse=False)
    pad = 32
    padded = np.pad(upright, pad, mode="reflect")
    freq_row = np.fft.fftfreq(padded.shape[0])[:, None]
    weight = 1.0 + along * (2.0 * np.pi * freq_row) ** 2
    filtered = np.fft.ifft2(np.fft.fft2(padded) / weight).real[pad:-pad, pad:-pad]
    filtered = np.sign(filtered) * np.maximum(np.abs(filtered) - sparse, 0.0)
    return _shear(filtered, slope, inverse=True)


def decompose(observed, background_gmm, rain_gmm):
    slope = _streak_slope(rain_gmm)
    basis = _streak_basis(slope)
    print(f"streak slope {slope:.3f}  basis {basis.shape[1]}", flush=True)
    background = observed.copy()
    for step in range(1, STREAK_PASSES + 1):
        started = time.perf_counter()
        patches = im2patches(background)
        centered = patches - patches.mean(axis=0, keepdims=True)
        projection = basis @ (basis.T @ centered)
        merged = merge_patches(patches - projection, observed.shape)
        cover = merge_patches(np.ones_like(patches), observed.shape)
        background = merged / np.maximum(cover, 1.0)
        print(
            f"pass {step}/{STREAK_PASSES}  hgrad {np.mean(np.abs(np.diff(background, axis=1))):.4f}  "
            f"{time.perf_counter() - started:5.1f}s",
            flush=True,
        )
    detail = observed - background
    started = time.perf_counter()
    kept = _average_map(detail, background_gmm, DETAIL_NOISE)
    print(
        f"background map  kept {np.mean(np.abs(kept)):.4f}  "
        f"{time.perf_counter() - started:5.1f}s",
        flush=True,
    )
    background = np.clip(background + kept, 0.0, 1.0)
    rain = observed - background
    return background, rain


def _guided(guide, source, radius, eps):
    size = 2 * radius + 1
    mean_g = uniform_filter(guide, size=size, mode="nearest")
    mean_s = uniform_filter(source, size=size, mode="nearest")
    corr = uniform_filter(guide * source, size=size, mode="nearest")
    var = uniform_filter(guide * guide, size=size, mode="nearest") - mean_g * mean_g
    scale = (corr - mean_g * mean_s) / (var + eps)
    offset = mean_s - scale * mean_g
    return (
        uniform_filter(scale, size=size, mode="nearest") * guide
        + uniform_filter(offset, size=size, mode="nearest")
    )


def _strip_thin(delta, slope):
    thin = _thin_rain(delta, 0.0)
    if abs(slope) > 1e-3:
        other = _thin_rain(delta, slope)
        if np.mean(np.abs(other)) > np.mean(np.abs(thin)):
            thin = other
    return delta - thin


def _sharpen_scatter(luminance, transmission, slope, radius=12.0, amount=2.2, lift=0.02):
    """Boost the wide detail where t is low. Thin streaks in that boost are removed.

    A pixel may rise only a little, so the bright side of an edge does not become a white rim.
    """
    depth = np.clip((0.92 - transmission) / 0.45, 0.0, 1.0)
    detail = _strip_thin(luminance - gaussian_filter(luminance, radius), slope)
    corrected = luminance + amount * depth * detail
    return np.clip(np.minimum(corrected, luminance + lift), 0.0, 1.0)


def defog_joint(derained, guide, slope=0.0, radius=15, omega=0.95, t_min=0.28):
    """O = h(t) * B t + R + F. The veil is read from the rainy guide, then applied to O-R.

    t is smoothed on the derained image. A pixel is never lifted above O-R by the
    veil. Wide detail is then boosted where t is low, and a pixel may rise only
    slightly. The thin part of either change is put back.
    """
    window = 2 * radius + 1
    dark = minimum_filter(guide.min(axis=2), size=window, mode="nearest")
    count = max(int(dark.size * 0.001), 1)
    picks = np.argpartition(dark.ravel(), -count)[-count:]
    airlight = float(np.clip(np.mean(guide.reshape(-1, 3)[picks]), 0.55, 0.98))
    scaled = np.min(guide / airlight, axis=2)
    raw = np.clip(1.0 - omega * minimum_filter(scaled, size=window, mode="nearest"), 0.0, 1.0)
    transmission = np.clip(_guided(derained.mean(axis=2), raw, radius * 2, 1e-2), t_min, 1.0)
    fog = airlight * (1.0 - transmission)
    stretched = np.minimum((derained - fog[..., None]) / transmission[..., None], derained)
    delta = stretched - derained
    thin = np.stack([_thin_rain(delta[:, :, channel], 0.0) for channel in range(3)], axis=2)
    if abs(slope) > 1e-3:
        other = np.stack([_thin_rain(delta[:, :, channel], slope) for channel in range(3)], axis=2)
        if np.mean(np.abs(other)) > np.mean(np.abs(thin)):
            thin = other
    clean = np.clip(np.minimum(stretched - thin, derained), 0.0, 1.0)
    luminance, cb, cr = rgb_to_ycbcr(clean)
    luminance = _sharpen_scatter(luminance, transmission, slope)
    clean = ycbcr_to_rgb(luminance, cb, cr)
    return clean, fog, transmission, airlight


def defog(color, radius=15, omega=0.7, t_min=0.35):
    """Solve B = B' t + F with the dark-channel method.

    F = A(1-t) is the veil. Splashes and mist both raise the local floor,
    so they are carried by the same F. B' = (B - F) / t.
    When t is near 1 this is the additive split B = B' + F.
    """
    window = 2 * radius + 1
    dark = minimum_filter(color.min(axis=2), size=window, mode="nearest")
    count = max(int(dark.size * 0.001), 1)
    picks = np.argpartition(dark.ravel(), -count)[-count:]
    airlight = float(np.clip(np.mean(color.reshape(-1, 3)[picks]), 0.55, 0.98))
    scaled = np.min(color / airlight, axis=2)
    raw = 1.0 - omega * minimum_filter(scaled, size=window, mode="nearest")
    guide = color.mean(axis=2)
    transmission = np.clip(_guided(guide, np.clip(raw, 0.0, 1.0), radius * 2, 1e-3), t_min, 1.0)
    fog = airlight * (1.0 - transmission)
    clean = np.clip((color - fog[..., None]) / transmission[..., None], 0.0, 1.0)
    return clean, fog, transmission, airlight


def save_outputs(directory, color, restored, rain):
    os.makedirs(directory, exist_ok=True)
    image = np.clip(np.rint(restored * 255.0), 0, 255).astype(np.uint8)
    original = np.clip(np.rint(color * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(image).save(os.path.join(directory, "derained.png"))
    Image.fromarray(np.clip(np.rint((rain + 0.5) * 255.0), 0, 255).astype(np.uint8)).save(
        os.path.join(directory, "rain.png")
    )
    Image.fromarray(np.concatenate([original, image], axis=1)).save(
        os.path.join(directory, "compare.png")
    )


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Li et al. CVPR 2016 layer-prior deraining")
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(here), "LPDerain-main", "case1.jpg"),
        help="rainy RGB image; default is the paper example case1.jpg",
    )
    parser.add_argument(
        "--output",
        default=here,
        help="directory for derained.png, rain.png, and compare.png",
    )
    parser.add_argument(
        "--method",
        choices=("paper", "current"),
        default="paper",
        help="paper: the layer-prior reproduction; current: multi-scale rain plus the veil",
    )
    args = parser.parse_args()
    opened = Image.open(args.input)
    color = np.asarray(
        opened.convert("RGB").resize(
            (opened.size[0] // 2, opened.size[1] // 2),
            Image.Resampling.BICUBIC,
        ),
        dtype=np.float64,
    ) / 255.0
    luminance, cb, cr = rgb_to_ycbcr(color)
    print(f"Y {luminance.shape}", flush=True)
    background_gmm = load_background_gmm()
    print(
        f"background GMM {background_gmm['count']}  covs {background_gmm['covs'].shape}",
        flush=True,
    )
    if args.method == "paper":
        rain_gmm = learn_rain_gmm(luminance)
        print(f"rain GMM components {rain_gmm['count']}", flush=True)
        background, rain = decompose(luminance, background_gmm, rain_gmm)
        restored = ycbcr_to_rgb(np.clip(background, 0, 1), cb, cr)
    else:
        background, rain, slope = decompose_scales(luminance, background_gmm)
        derained = ycbcr_to_rgb(np.clip(background, 0, 1), cb, cr)
        restored, _, _, _ = defog_joint(derained, color, slope=slope)
    save_outputs(args.output, color, restored, rain)
    print(
        f"rain abs {np.mean(np.abs(rain)):.4f}  "
        f"saved {os.path.abspath(args.output)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
