"""Signal processing."""
import warnings

from astropy import convolution as ap_conv
import numpy as np
from obspy.signal.cross_correlation import correlate, xcorr_max
import scipy as sp
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
import scipy.ndimage
from scipy.signal import hilbert


#
# Coefficients for (central) finite difference computation
# https://www.wikiwand.com/en/Finite_difference_coefficient
# First derivatives
FDC_X_1 = {
    2: np.array([[1, 0, -1]]) / 2,
    4: np.array([[-1, 8, 0, -8, 1]]) / 12,
    6: np.array([[1, -9, 45, 0, -45, 9, -1]]) / 60,
}
FDC_Y_1 = {order: coef.T for order, coef in FDC_X_1.items()}
# Second derivatives
FDC_X_2 = {
    2: np.array([[1, -2, 1]]),
    4: np.array([[-1, 16, -30, 16, -1]]) / 12,
    6: np.array([[2, -27, 270, -490, 270, -27, 2]]) / 180,
}
FDC_Y_2 = {order: coef.T for order, coef in FDC_X_2.items()}


def _normal(x, *p):
    """
    Definition of Gauss function to fit.
    """
    A, mu, sigma = p
    pdf = A * np.exp(-(x-mu)**2 / (2*sigma**2))

    return pdf


def _skew_normal(x, *p):
    """
    Definition of skewed normal distribution.

    https://www.wikiwand.com/en/Skew_normal_distribution
    """
    A, mu, sigma, skew = p
    norm = np.exp(-(x-mu)**2 / (2*sigma**2))
    erf = 1 + sp.special.erf(skew*(x-mu)/sigma/np.sqrt(2))
    pdf = A * norm * erf

    return pdf


def _laplace(x, *p):
    """
    Laplace distribution.
    """
    A, mu, sigma = p
    pdf = A * np.exp(-np.abs(x-mu) / sigma)

    return pdf


def _asymmetric_laplace(x, *p):
    """
    Asymmetric Laplace distribution.

    https://www.wikiwand.com/en/Asymmetric_Laplace_distribution
    """
    A, mu, sigma, skew = p
    s = np.sign(x - mu)
    pdf = A * np.exp(- np.abs(x-mu) / sigma * skew**s)

    return pdf


def fit_dist(y, dist='normal', **kwargs):
    """
    Fit histogram to a distribution.
    """
    hist, bin_edges = np.histogram(y, **kwargs)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    dist = dist.lower()
    if dist in ['normal', 'gauss', 'gaussian']:
        p0 = [np.max(hist), np.mean(y), np.std(y)]
        func = _normal
    elif dist in ['skew_normal']:
        p0 = [np.max(hist), np.mean(y), np.std(y), sp.stats.skew(y)]
        func = _skew_normal
    elif dist in ['laplace']:
        p0 = [np.max(hist), np.mean(y), np.std(y)]
        func = _laplace
    elif dist in ['asymmetric_laplace', 'ald']:
        p0 = [np.max(hist), np.mean(y), np.std(y), sp.stats.skew(y)]
        func = _asymmetric_laplace

    popt, pcov = curve_fit(func, bin_centers, hist, p0=p0)
    popt[2] = abs(popt[2])
    hist_fit = func(bin_centers, *popt)

    return popt, bin_centers, hist, hist_fit


def normalized(a, norm=None):
    """
    Normalize 1D array to its absolute maximum.

    https://docs.obspy.org/_modules/obspy/core/trace.html#Trace.normalize
    """
    # normalize, use norm-kwarg otherwise normalize to 1
    if norm is not None:
        norm = norm
        if norm < 0:
            msg = "Normalizing with negative values is forbidden. " + \
                  "Using absolute value."
            warnings.warn(msg)
    else:
        norm = np.abs(a).max()

    # Don't do anything for zero norm but raise a warning.
    if not norm:
        msg = ("Attempting to normalize by dividing through zero. This "
               "is not allowed and the data will thus not be changed.")
        warnings.warn(msg)
        return a

    # Convert data if it's not a floating point type.
    if not np.issubdtype(a.dtype, float):
        a = np.require(a, dtype=np.float64)

    return a / abs(norm)


def rms(a):
    """
    Root mean square.
    """
    if (a.size == 0) or np.all(a == 0):
        warnings.warn('Empty array.')
        return None
    else:
        return np.sqrt((a**2).sum() / a.size)


def corr_coef(reference, current, shift=None, demean=True, abs_max=True,
              domain='freq'):
    """
    Return shift and value of maximum of cross-correlation.
    """
    if shift == 0:
        domain = 'time'

    fct = correlate(reference, current, shift=shift,
                    demean=demean, normalize=True, domain=domain)

    return xcorr_max(fct, abs_max=abs_max)


def ddot(x, y, interp=True):
    """
    Return 2nd order derivative.
    """
    if interp:
        iinc = np.argsort(x)
        _x = x[iinc]
        _y = y[iinc]
        fy = CubicSpline(_x, _y)
        yddot = fy.derivative(2)(x)
    else:
        ydot = np.gradient(y, x)
        yddot = np.gradient(ydot, x)

    return yddot


def inst_ph(a):
    """
    Return instantaneous phase.
    """
    return np.unwrap(np.angle(hilbert(a)))


def corr2(a, b):
    """
    2-D correlation coefficient.

    https://www.mathworks.com/help/images/ref/corr2.html#f1-227958
    """
    a_mean = a.mean()
    b_mean = b.mean()
    a_diff = a - a_mean
    b_diff = b - b_mean
    numerator = np.nansum(a_diff * b_diff)
    denominator = np.sqrt(np.nansum(a_diff**2) * np.nansum(b_diff**2))

    return numerator / denominator


def ma_corr2(a, b):
    """
    Similar to :meth:`corr2()` but for masked array.
    """
    mask = a.mask + b.mask
    a_mean = a.mean()
    b_mean = b.mean()
    a_diff = a - a_mean
    a_diff.mask += mask
    b_diff = b - b_mean
    b_diff.mask += mask
    numerator = np.nansum(a_diff * b_diff)
    denominator = np.sqrt(np.nansum(a_diff**2) * np.nansum(b_diff**2))

    return numerator / denominator


def nextpow2(n):
    """
    Exponent of next higher power of 2.

    https://www.mathworks.com/help/matlab/ref/nextpow2.html
    """
    return np.ceil(np.log2(n))


def intersect2d(a, b, assume_unique=False, return_indices=False):
    """
    Find the intersection of two 2-D arrays.

    https://stackoverflow.com/a/8317403/8877268

    :param assume_unique:
    :param return_indices:
    """
    av = a.view([('', a.dtype)]*a.shape[1]).ravel()
    bv = b.view([('', b.dtype)]*b.shape[1]).ravel()
    res = np.intersect1d(
        av, bv,
        assume_unique=assume_unique,
        return_indices=return_indices,
    )
    if return_indices:
        int2d = res[0].view(a.dtype).reshape(-1, a.shape[1])
        return int2d, res[1], res[2]
    else:
        int2d = res.view(a.dtype).reshape(-1, a.shape[1])
        return int2d


def spectrum(a, delta=1, **kwargs):
    """
    Return frequencies, amplitude and phase of FFT.
    """
    nfreq = kwargs.get('nfreq')
    if nfreq is None:
        nfreq = sp.fftpack.next_fast_len(a.size)
    spectrum = np.fft.rfft(a, nfreq)
    freqs = np.fft.rfftfreq(n=nfreq, d=delta)

    am = np.abs(spectrum)
    ph = np.angle(spectrum)

    return freqs, am, ph


def nsig(x, n=2):
    """
    Round to n signifcant figures.

    https://stackoverflow.com/a/3411435/8877268
    """
    return round(x, -int(np.floor(np.log10(x))) + (n - 1))


def linrange(start, stop, step=1):
    """
    Mix of arange & linspace.
    """
    if (stop-start) % step != 0:
        raise ValueError('Can not include end')
    num = int(abs(stop-start)/step + 1)

    return np.linspace(start, stop, num)


def conv_spc(in1, in2, axes=None):
    """
    Return spectra of convolution (spectral multiplication).

    https://github.com/scipy/scipy/blob/v1.3.0/scipy/signal/signaltools.py#L282-L440
    """
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)
    noaxes = axes is None

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return np.array([])

    _, axes = sp.fftpack.helper._init_nd_shape_and_axes_sorted(
        in1,
        shape=None,
        axes=axes,
    )

    if not noaxes and not axes.size:
        raise ValueError("when provided, axes cannot be empty")

    if noaxes:
        other_axes = np.array([], dtype=np.intc)
    else:
        other_axes = np.setdiff1d(np.arange(in1.ndim), axes)

    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)

    if not np.all((s1[other_axes] == s2[other_axes])
                  | (s1[other_axes] == 1) | (s2[other_axes] == 1)):
        raise ValueError("incompatible shapes for in1 and in2:"
                         " {0} and {1}".format(in1.shape, in2.shape))

    shape = np.maximum(s1, s2)
    shape[axes] = s1[axes] + s2[axes] - 1

    # Speed up FFT by padding to optimal size for sp.fftpack
    fshape = [sp.fftpack.helper.next_fast_len(d) for d in shape[axes]]
    sp1 = np.fft.fftn(in1, fshape, axes=axes)
    sp2 = np.fft.fftn(in2, fshape, axes=axes)

    return sp1 * sp2


def norm_weight(w, axis=0):
    """
    Normalize weights.
    """
    w_sum = np.nansum(w, axis=axis)
    w_sum_all = np.broadcast_to(w_sum, w.shape)
    has_weight = (w_sum_all != 0)
    w[has_weight] = w[has_weight] / w_sum_all[has_weight]

    return w


def weighted_mean(x, w, axis=0):
    """
    https://www.wikiwand.com/en/Weighted_arithmetic_mean#/Mathematical_definition
    """
    w_sum = np.nansum(w, axis=axis)
    has_weight = (w_sum != 0)
    mean = np.zeros(w_sum.shape)
    mean[has_weight] = np.nansum(x*w, axis=axis)[has_weight] / w_sum[has_weight]

    return mean, w_sum, has_weight


def weighted_sd(x, w, axis=0):
    """
    https://www.wikiwand.com/en/Weighted_arithmetic_mean#/Weighted_sample_variance
    """
    mean, w_sum, has_weight = weighted_mean(x, w, axis)
    mean_bc = np.broadcast_to(mean, x.shape)
    numerator = np.nansum(w * (x-mean_bc)**2, axis=axis)[has_weight]
    denomitor = w_sum[has_weight]
    sd = np.zeros(mean.shape)
    sd[has_weight] = np.sqrt(numerator / denomitor)

    return sd, mean


def weighted_sem(x, w, axis=0):
    """
    No n/(n-1) factor.

    https://www.wikiwand.com/en/Weighted_arithmetic_mean#/Bootstrapping_validation
    """
    mean, w_sum, has_weight = weighted_mean(x, w, axis)
    mean_bc = np.broadcast_to(mean, x.shape)
    numerator = np.nansum(w**2 * (x-mean_bc)**2, axis=axis)[has_weight]
    denomitor = w_sum[has_weight]**2
    sem = np.zeros(mean.shape)
    sem[has_weight] = np.sqrt(numerator / denomitor)

    return sem, mean


def gaussian_filter(a, x_stddev, y_stddev=None, **kwargs):
    """
    http://docs.astropy.org/en/stable/convolution/index.html
    """
    kwargs_def = {
        'fill_value': np.nan,
        'preserve_nan': True,
        # 'boundary': 'extend',
        # 'nan_treatment': 'interpolate',
    }
    a2 = ap_conv.convolve(
        a,
        kernel=ap_conv.Gaussian2DKernel(x_stddev, y_stddev),
        **{**kwargs_def, **kwargs}
    )

    return a2


def gradient_2d(z, dx, dy, order=2):
    """
    Not use Numpy because np.gradient dx or dy must be 1D.

    https://github.com/numpy/numpy/issues/9401
    https://github.com/Unidata/MetPy/issues/174
    """
    coef_x = FDC_X_1[order]
    coef_y = FDC_Y_1[order]

    zx = sp.ndimage.convolve(z, coef_x) / dx
    zy = sp.ndimage.convolve(z, coef_y) / dy

    return zx, zy


def _lplc_green(zx, zy, dx, dy):
    """
    Green's first identity with psi=1
    https://www.wikiwand.com/en/Green%27s_identities#/Green's_first_identity
    """
    zxp = zx[1:-1, 2:]
    zxn = zx[1:-1, :-2]
    # Positive y direction is opposite to latitude
    zyn = zy[2:, 1:-1]
    zyp = zy[:-2, 1:-1]

    loopsum = (zxp-zxn)*dy - (zyp-zyn)*dx
    area = dy * dx
    lplc = loopsum / area

    return lplc


def _lplc_fd(z, dx, dy, order):
    coef_x = FDC_X_2[order]
    coef_y = FDC_Y_2[order]

    zxx = sp.ndimage.convolve(z, coef_x) / dx**2
    zyy = sp.ndimage.convolve(z, coef_y) / dy**2

    lplc = zxx + zyy

    return lplc


def laplacian_2d(dx, dy, z=None, zx=None, zy=None, method='Green', order=2):
    method = method.lower()
    if method in ['green']:
        lplc = _lplc_green(zx, zy, dx, dy)
    elif method in ['fd', 'finite_difference']:
        lplc = _lplc_fd(z, dx, dy, order)
    else:
        raise ValueError(f'Unknow method {method}')

    return lplc
