import numpy as np
from scipy import signal
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import stats
import math

def savgol_smoothing(keypoints, window_length=10, polyorder=3, deriv=0):
    """
    Savgol (Savitzky-Golay) filter
    :param keypoints: the keypoints to smooth
    :param window_length: size of the smoothing window
    :param polyorder: order of the polynomial
    :param deriv: order of the derivative
    :return: the smoothed data
    """
    return signal.savgol_filter(keypoints, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=0)


def median_smoothing(keypoints, kernel_size=(5,1)):
    # smoothed = np.zeros_like(keypoints)
    return signal.medfilt(keypoints, kernel_size=kernel_size)


def gaussian_smoothing(keypoints):
    return ndimage.gaussian_filter1d(keypoints, 1, axis=0)


def filter_keypoints(keypoints, trim=None, smooth=False):
    keypoints = keypoints.reshape((keypoints.shape[0], -1, 2))

    diff_pixel_cutoff = [-25, 25]
    med_pixel_cutoff = [200, 25]
    win_size = [3, 3]
    savgol_window = [7,9]
    savgol_order = [2,3]
    n_outliers = [0, 0]
    n_ok = [0, 0]
    for i in range(2):  # x,y coordinates

        ## 1. Remove outliers
        for k in range(keypoints[:, :, i].shape[1]):  # for each keypoint
            s = keypoints[:, k, i]

            for j in range(0, s.shape[0] - win_size[i]):  # for each window
                win = s[j:j + win_size[i] + 1]
                delta = np.diff(win, prepend=win[0])

                if i == 0:  # for x, remove pixels to the left
                    ok_px = delta > diff_pixel_cutoff[i]
                else:
                    ok_px = abs(delta) < diff_pixel_cutoff[i]

                win_med = np.median(win)
                med_delta = win - win_med
                ok_100 = abs(med_delta) < med_pixel_cutoff[i]

                ok = np.all([ok_px, ok_100], axis=0)
                keypoints[j:j + win_size[i] + 1, k, i] = np.where(ok, win, win_med)
                n_outliers[i] += np.sum(np.invert(ok))
                n_ok[i] += np.sum(ok)
        ## 2. Savgol smoothing
        if smooth:
            keypoints[:, 0:14, i] = savgol_smoothing(keypoints[:, 0:14, i], window_length=savgol_window[i], polyorder=savgol_order[i])
            keypoints[:, 14:, i] = savgol_smoothing(keypoints[:, 14:, i], window_length=savgol_window[i]*2, polyorder=savgol_order[i])  # spine keypoints are more smoothed

    keypoints = keypoints.reshape((keypoints.shape[0], -1))
    if trim is not None:
        keypoints = keypoints[trim[0]:trim[1], :]

    # if n_outliers > 0:
    #     print("outliers", n_outliers)
    return keypoints, n_outliers, n_ok


def plot_circle(center, radius, show=True):
    if center is None:
        return None

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim((center[0]-radius, center[0] + radius))
    ax.set_ylim((center[1]-radius, center[1] + radius))
    # plt.figure(figsize=(4, 4))
    circle = plt.Circle(center, radius, color='r')
    ax.add_artist(circle)

    if show:
        plt.show()

    return fig


def compute_circle_radius(a, b, c):
    """
        Source: https://stackoverflow.com/questions/28910718/give-3-points-and-a-plot-circle
        Returns the center and radius of the circle passing the given 3 points.
        In case the 3 points form a line, returns (None, infinity).
        """
    temp = b[0] * b[0] + b[1] * b[1]
    bc = (a[0] * a[0] + a[1] * a[1] - temp) / 2
    cd = (temp - c[0] * c[0] - c[1] * c[1]) / 2
    det = (a[0] - b[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - b[1])

    if abs(det) < 1.0e-6:
        return None, np.inf

    # Center of circle
    cx = (bc * (b[1] - c[1]) - cd * (a[1] - b[1])) / det
    cy = ((a[0] - b[0]) * cd - (b[0] - c[0]) * bc) / det

    radius = np.sqrt((cx - a[0]) ** 2 + (cy - a[1]) ** 2)
    return (cx, cy), radius


def compute_BPM(keypoints, norm=1.0):
    """
    Back Posture Measurment following the methodology from Poursaberry 2010.
    Place 3 points on the spine: A at the hip area, C at the shoulder area, and B in the middle of A and C.
    Fit a circle passing through these 3 points. The curvature of the back is 1/Radius of the circle,
    the radius is expressed in pixels
    """
    if len(keypoints.shape) == 2:
        return _compute_BPM_one_frame(keypoints, norm)

    BPMs = []
    for i in range(keypoints.shape[0]):
        BPMs.append(_compute_BPM_one_frame(keypoints[i, :, :], norm))

    return np.asarray(BPMs)


def _compute_BPM_one_frame(frame, norm=1.0):
    a, b, c = frame[0, :], frame[1, :], frame[2, :]
    center, radius = compute_circle_radius(a, b, c)
    if center is None:
        bpm = 0.0
    else:
        bpm = 1 / (radius + np.spacing(1))  # spacing to avoid div by 0
        bpm *= norm  # 1920 = width of the image
    return bpm

def compute_back_angle(keypoints):
    if len(keypoints.shape) == 2:
        return compute_angle_abc(keypoints[0, :], keypoints[1, :], keypoints[2, :])

    angles = []
    for i in range(keypoints.shape[0]):
        angles.append(compute_angle_abc(keypoints[i, 0, :], keypoints[i, 1, :], keypoints[i, 2, :]))

    return np.asarray(angles)

def compute_angle_abc(a, b, c, degrees=False):
    """
    Source: https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
    Computes the angle between 3 points. Returns the angle in gradients or degrees. Default: gradients.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    if degrees:
        angle = np.degrees(angle)

    return angle


def zero_runs(x):
    """
    Helper function for finding sequences of 0s in a signal
    https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array/24892274#24892274
    """
    iszero = np.concatenate(([0], np.equal(x, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def find_plateaus(F, min_length=10, tolerance=0.75, smoothing=3):
    """
    Finds plateaus of signal using second derivative of F.
    source: https://stackoverflow.com/a/68083384

    Parameters
    ----------
    F : Signal.
    min_length: Minimum length of plateau.
    tolerance: Number between 0 and 1 indicating how tolerant
        the requirement of constant slope of the plateau is.
    smoothing: Size of uniform filter 1D applied to F and its derivatives.

    Returns
    -------
    plateaus: array of plateau left and right edges pairs
    dF: (smoothed) derivative of F
    d2F: (smoothed) Second Derivative of F
    """
    from scipy.ndimage import uniform_filter1d
    # calculate smooth gradients
    smoothF = uniform_filter1d(F, size=smoothing)
    dF = uniform_filter1d(np.gradient(smoothF), size=smoothing)
    d2F = uniform_filter1d(np.gradient(dF), size=smoothing)

    # Find ranges where second derivative is zero
    # Values under eps are assumed to be zero.
    eps = np.quantile(abs(d2F), tolerance)
    smalld2F = (abs(d2F) <= eps)

    # Find repititions in the mask "smalld2F" (i.e. ranges where d2F is constantly zero)
    p = zero_runs(np.diff(smalld2F))

    # np.diff(p) gives the length of each range found.
    # only accept plateaus of min_length
    plateaus = p[(np.diff(p) > min_length).flatten()]

    return plateaus, dF, d2F


def find_plateaus_2(F, widths=3, noise_perc=30):
    """
    Another approach to find plateaus using wavelets
    """
    start = signal.find_peaks_cwt(F, widths, noise_perc=noise_perc)
    end = signal.find_peaks_cwt(-F, widths, noise_perc=noise_perc)

    return start, end


def find_plateaus_numpy(sig, tolerance, min_length, smoothing=0):
    """
    My own approach to find plateaus.
    Not as mathy as the other ones, but it seemed to work better for detecting the stance phases.
    """
    plateaus = []
    plateau_start = None

    if smoothing > 1:
        sig = signal.medfilt(sig, kernel_size=smoothing)

    for i in range(1, sig.shape[0]):
        if math.isclose(sig[i], sig[i - 1], abs_tol=tolerance):
            if plateau_start is None:
                plateau_start = i - 1
        else:
            if plateau_start is not None:
                plateau_end = i - 1
                plateau_size = plateau_end - plateau_start + 1
                if plateau_size >= min_length:
                    plateaus.append((plateau_start, plateau_end))
                plateau_start = None

    # Check if a plateau continues till the end of the signal
    if plateau_start is not None:
        plateau_end = sig.shape[0] - 1
        plateau_size = plateau_end - plateau_start + 1
        if plateau_size >= min_length:
            plateaus.append((plateau_start, plateau_end))

    plateaus = np.array(plateaus)
    return plateaus


def rolling_numpy(x, w, func, args):
    """
    Rolling window
    """
    res = []
    step = w - 1
    center = w - ((w + 1) // 2)  # index of the center of the window

    for i in range(0, x.shape[0] - step):
        res.append(func(x[i:i + w], *args))
    res = np.array(res)
    res = np.pad(res, (center, (w - 1) - center), 'edge')
    return res

def rolling_median(x, w=3):
    return rolling_numpy(x, w, np.median, [])

def rolling_sigma(x, w=3):
    return rolling_numpy(x, w, stats.median_abs_deviation, [0, np.median, 'normal'])


def rolling_hampel(keypoints, w, n=3):
    """
    Hampel fitler with rolling window
    """
    step = w - 1
    filtered_keypoints = np.zeros_like(keypoints)

    for i in range(keypoints.shape[1]):
        x = np.array(keypoints[:, i])  # copy the array
        for j in range(0, x.shape[0] - step):
            x_win = x[j:j + w]
            med = np.median(x_win)
            sig = stats.median_abs_deviation(x_win, 0, np.median, 'normal')

            outlier_indices = np.asarray(np.abs(x_win - med) >= (n * sig))
            x_win[outlier_indices] = med

        filtered_keypoints[:, i] = x
    return filtered_keypoints


def hampel_filter(keypoints, w, n=3, replace=False):
    """
    Numpy implementation of the Hampel filter (Aka MAD filter).
    Adapted from https://github.com/MichaelisTrofficus/hampel_filter (pandas implementation)
    The Hampel filter is generally used to detect anomalies in data with a timeseries structure.
    It basically consists of a sliding window of a parameterizable size.
    For each window, each observation will be compared with the Median Absolute Deviation (MAD).
    The observation will be considered an outlier in the case in which it exceeds the MAD by n times
    :param x: the time series
    :param w: window size
    :param n: threshold, default is 3 (Pearson's rule).
    :param replace: whether to replace the outliers in x. Default: False
    :return: x, indices of the outliers
    """
    filtered_keypoints = np.zeros_like(keypoints)
    outliers = np.zeros_like(keypoints)
    for i in range(keypoints.shape[1]):
        x = np.array(keypoints[:,i])  # copy the array
        r_med = rolling_median(x, w)
        r_sig = rolling_sigma(x, w)

        outlier_indices = np.asarray(np.abs(x - r_med) >= (n * r_sig))

        if replace:
            # x[outlier_indices] = r_med[outlier_indices]
            x[outlier_indices] = np.NaN
            # x = np.interp(outlier_indices, np.arange(0, x.shape[0]), x)
        filtered_keypoints[:,i] = x
        outliers[:,i] = outlier_indices

    return filtered_keypoints, outliers