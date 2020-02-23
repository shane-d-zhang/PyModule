"""Seismology related utilities."""
import logging.config
from sys import exit


import numpy as np
import obspy
from obspy.core import inventory
from obspy.core.inventory.inventory import Inventory
from obspy.core.trace import Trace
from obspy.signal.cross_correlation import correlate
import pyproj
import scipy as sp

from pymodule.signal import (
    rms,
    spectrum,
)


logger = logging.getLogger(__name__)
GEOD = pyproj.Geod(ellps='WGS84')


def geod_inv(lon1, lat1, lon2, lat2, radians=False, km=True):
    """
    Can take array.

    https://pyproj4.github.io/pyproj/html/api/geod.html#pyproj.Geod.inv

    :return az: azimuth 1->2
    :return baz: azimuth 2->1 (backazimuth)
    """
    az, baz, dist = GEOD.inv(lon1, lat1, lon2, lat2, radians=radians)
    if km:
        dist /= 1000

    return dist, az, baz


def get_spectrum(tr):
    """
    Return frequencies, amplitude and phase of FFT.

    :param tr:
    :type tr: :class:`~obspy.core.trace.Trace`
    """
    freqs, am, ph = spectrum(tr.data, tr.stats.delta)

    return freqs, am, ph


def _cut_ends(tr, b, e):
    """
    Return ends to cut as :class:`obspy.core.UTCDateTime`.
    """
    ref = tr.stats.starttime
    try:
        starttime = ref + b - tr.stats.sac.b
        endtime = ref + e - tr.stats.sac.b
    except TypeError:
        starttime = None
        endtime = None

    return starttime, endtime


def sliced(tr, b=None, e=None, data_only=False):
    """
    Return sliced trace given begin/end relative to starttime.
    """
    if data_only:
        ib = int(max(0, b-tr.stats.sac.b) / tr.stats.delta)
        ie = int((min(tr.stats.sac.e, e)-tr.stats.sac.b) / tr.stats.delta)
        return tr.data[ib:ie]
    else:
        starttime, endtime = _cut_ends(tr, b, e)
        return tr.slice(starttime, endtime)


def trim(tr, b=None, e=None):
    """
    Trim trace given begin/end relative to starttime.
    """
    starttime, endtime = _cut_ends(tr, b, e)

    return tr.trim(starttime, endtime)


def _data_sym(a):
    """
    Return the sum of positive and negative lags.
    """
    npts = len(a)
    if npts % 2 == 0:
        x1 = int(npts/2) - 1
        x2 = x1 + 1
        return a[x1::-1] + a[x2:]
    else:
        half = int(npts/2)
        return a[half::-1] + a[half:]


def _hd_sym(ref, npts):
    hd = ref.copy()
    hd.npts = npts
    hd.starttime += hd.sac.e
    hd.sac.b = 0

    return hd


def _is_sym(tr):
    delta = tr.stats.delta
    npts = tr.stats.npts
    b = tr.stats.sac.b
    e = getattr(tr.stats.sac, 'e', int(b+npts*delta))

    return abs(b + e) <= delta


def _even_npts(tr):
    return tr.stats.npts % 2 == 0


def sym_xc(xc, noeven=False):
    """
    Symmetrize a cross-correlation.
    """
    pair = f'{xc.stats.sac.kevnm.split()[0]}-{xc.stats.station}'
    if not _is_sym(xc):
        logger.debug(f'Asymmetric {pair}')
        end = min(abs(xc.stats.sac.b), xc.stats.sac.e)
        xc = sliced(xc, -end, end)
    if _even_npts(xc) and noeven:
        logger.error(f'Even npts to symmexcize {pair}')
    else:
        sym = Trace()
        sym.data = _data_sym(xc.data)
        sym.stats = _hd_sym(xc.stats, npts=sym.data.size)
        return sym


def x_cor(tr1, tr2, **kwargs):
    """
    Do correlation/convolution between two
    :class:`~obspy.core.trace.Trace` objects.

    ObsPy:

    .. math::

        \\int f(\\tau) g(t+\\tau) d\\tau


    The definition in SAC is different:

    .. math::

        \\int f(t+\\tau) g(\\tau) d\\tau

    Results agree except at ends (~100 s for one lag of ~1400 s).
    Not sure about the reason.

    :param demean: demean data beforehand
    :param normalize: for correlation coefficient
    :param domain: "time" or "freq"
    :param Welch: apply Welch's method for correlation
    :param subwin: length of subwindow [s]
    :param lap: lap fraction of subwindows
    :param op: correlation, convolution
    """
    demean = kwargs.get('demean', False)
    normalize = kwargs.get('normalize', False)
    domain = kwargs.get('domain', 'freq')
    Welch = kwargs.get('Welch', False)
    subwin = kwargs.get('subwin', None)
    lap = kwargs.get('lap', None)
    op = kwargs.get('op', 'corr').lower()

    delta = tr1.stats.delta
    npts = min(tr1.stats.npts, tr2.stats.npts)

    # Interchange to make convention same as SAC
    if 'conv' in op:
        a2 = tr1.data[::-1]
    else:
        a2 = tr1.data
    a1 = tr2.data

    # Use Welch's method?
    if Welch:

        # Check length of subwindow
        twin = int(npts * delta)
        try:
            if subwin > twin:
                logger.warning(f'Welch subwin {subwin:.0f} > twin {twin:.0f}')
                raise ValueError
        except ValueError as e:
            logger.exception(e)
            subwin = twin

        winlen = int(subwin / delta)
        step = int(winlen * (1-lap))
        indices = np.arange(0, npts-winlen+1, step)
        correlation = 0
        for ind in indices:
            correlation += correlate(a1[ind:ind+winlen],
                                     a2[ind:ind+winlen],
                                     shift=winlen, demean=demean,
                                     normalize=normalize, domain=domain)
        return correlation / len(indices)
    else:
        shift = kwargs.get('shift', npts - 1)
        return correlate(a1, a2, shift=shift, demean=demean,
                         normalize=normalize, domain=domain)


def hdr_cor(reftime='2000-01-01', **kwargs):
    """
    Create a new header for correlatoin.

    :param kuser1: rec network
    :param user1: 1 for symmetric & 2 for double-sided
    """

    hd = sachd(reftime=reftime, **kwargs)
    dist, az, baz = geod_inv(
        lat1=hd.sac.evla,
        lon1=hd.sac.evlo,
        lat2=hd.sac.stla,
        lon2=hd.sac.stlo,
    )
    hd.sac.update({
        'dist': dist,
        'az': az,
        'baz': baz,
    })

    return hd


def _hdr_stack(st, stats=None, **kwargs):
    """
    Create a new header for stack.

    :param ref_hdr: reference header for delta
    """
    if stats is None:
        stats = st[0].stats.copy()

    key_nsided = kwargs.get('nsided', 'user1')

    if kwargs.get('symmetric', False):
        stats.sac.b = 0
        stats.sac[key_nsided] = 1
    # else:
        # stats.sac.b = 0

    return stats


def _data_stack(st, **kwargs):
    """
    Stack all traces in a stream.
    """
    key_snr = kwargs.get('snr', 'user2')
    symmetric = kwargs.get('symmetric', False)
    weight = kwargs.get('weight', 'None').lower()
    ow = kwargs.get('ow', True)

    ntr = st.count()

    data = 0
    for k, tr in enumerate(st):
        if 'none' in weight:
            wi = 1
        elif 'unit' in weight:
            wi = 1 / np.abs(tr.data).max()
        elif ('rms' in weight) or ('snr' in weight):
            _, rms, snr_ = snr(tr, return_all=True, **kwargs)
            if snr_ < 0:
                ntr -= 1
            else:
                if 'rms' in weight:
                    wi = 1 / rms
                elif 'snr' in weight:
                    wi = snr_
                tr.stats.sac[key_snr] = snr_
        else:
            raise NotImplementedError(f'Unknown weight method {weight}')

        data += wi * tr.data

        # Also change tr for random stack
        if ow:
            tr.data *= wi

    if ntr == 0:
        return
    else:
        data /= ntr
        if symmetric:
            return _data_sym(data)
        else:
            return data


def stack(st, stats=None, kw_stack={}, kw_snr={}):
    """
    Stack traces in stream.
    """
    if st.count() == 0:
        logger.error('No traces!')
        return st
    elif st.count() == 1:
        return st

    # Find lap
    if kw_stack.get('sort', False):
        st.sort(keys=['endtime'])
        st.trim(
            starttime=st[0].stats.starttime,
            endtime=st[0].stats.endtime,
            pad=True,
            fill_value=0,
        )

    sk = Trace()
    data = _data_stack(st, **kw_stack, **kw_snr)
    if data is None:
        return
    else:
        sk.data = data
        sk.stats = _hdr_stack(st, stats, **kw_stack, **kw_snr)

        key_snr = kw_stack.get('snr', 'user2')
        weight = kw_stack.get('weight', 'None').lower()
        if 'none' not in weight:
            sk.stats.sac[key_snr] = snr(sk, **kw_snr)

        return sk


def rand_stack(st, stats=None, kw_stack={}, kw_snr={}):
    nsub = kw_stack.get('nsub', 4)
    key_nsrc = kw_stack.get('key_nsrc', 'user0')

    ntr = st.count()
    if ntr < 2:
        return st

    sk = Trace()
    sk.stats = _hdr_stack(st, stats, **kw_stack, **kw_snr)

    if ntr <= nsub:
        for tr in st:
            yield tr
    else:
        ind = np.arange(ntr)
        np.random.shuffle(ind)
        for sub in np.array_split(ind, nsub):
            data = 0
            for i in sub:
                data += st[i].data
            n = sub.size
            data /= n
            sk.stats.sac[key_nsrc] = n
            sk.data = data
            yield sk

    return


def snr(tr, **kwargs):
    """
    Return SNR defined as
    (peak of signal amplitude) / (RMS of trailing noise)

    :param dsn: differential time between signal and noise
    :param nlen: length of noise window
    :param Tmax:
    :param [be]fact: [tau - bfact*Tmax, tau + efact*Tmax]
    :param fill: fill value if failing to measure SNR
    """
    Tmax = kwargs.get('Tmax', 150)
    bfact = kwargs.get('bfact', 1)
    efact = kwargs.get('efact', 1)
    dsn = kwargs.get('dsn', 200)
    nlen = kwargs.get('nlen', 200)
    vmax = kwargs.get('vmax', 5)
    vmin = kwargs.get('vmin', 2)
    fill = kwargs.get('fill', 3)
    debug = kwargs.get('debug', False)

    hd = tr.stats.sac
    try:
        pair = f'{hd.kevnm.split()[0]}_{hd.kstnm}'
    except (KeyError, AttributeError):
        pair = ''
    ref_sta = hd.get('kuser0', None)
    if ref_sta is not None:
        pair += f', {ref_sta}'

    dist = hd.dist
    ts1 = max(0, dist/vmax - bfact*Tmax)
    ts2 = dist/vmin + efact*Tmax
    tn1 = ts2 + dsn
    tn2 = tn1 + nlen
    te = hd.b + (hd.npts-1) * hd.delta

    if ts2 > te:
        ts2 = te - nlen
        tn1 = ts2
        tn2 = te
    elif tn2 > te:
        shift = min(tn2-te, dsn)
        tn1 -= shift
        tn2 -= shift

    sg = np.max(np.abs(sliced(tr, ts1, ts2, data_only=True)))
    ns = rms(sliced(tr, tn1, tn2, data_only=True))

    if ns is None:
        logger.warning(f'{pair}: noise is 0')
        snr_ = fill
    else:
        snr_ = sg / ns

    if debug:
        logger.debug(
            f'{pair}: signal [{ts1:.0f}, {ts2:.0f}], '
            + f'noise [{tn1:.0f}, {tn2:.0f}], '
            + f'SNR: {snr_:.0f}'
        )

    if kwargs.get('return_all', False):
        return sg, ns, snr_
    else:
        return snr_


def txt2inv(fin, **kwargs):
    """
    Return :class:`~obspy.core.inventory.Inventory` from text file.
    """
    col_sta = kwargs.get('col_sta', 0)
    col_lon = kwargs.get('col_lon', 1)
    col_lat = kwargs.get('col_lat', 2)
    col_net = kwargs.get('col_net', 3)
    source = kwargs.get('source', 'CIEI')
    chacodes = kwargs.get('chacodes', ['BHZ'])
    locode = kwargs.get('locode', '01')
    elev = kwargs.get('elev', 0)
    depth = kwargs.get('depth', 0)

    inv = Inventory(networks=[], source=source)
    ncha = len(chacodes)

    # These are not used but required by pyasdf
    creation_date = obspy.core.utcdatetime.UTCDateTime(0)
    site = inventory.util.Site(name='')

    with open(fin, 'r') as f:
        for line in f:
            meta = line.split()
            stacode = meta[col_sta]
            lon = meta[col_lon]
            lat = meta[col_lat]
            netcode = meta[col_net]
            lon = float(lon)
            if lon > 180:
                lon -= 360
            lat = float(lat)
            xyz = {'latitude': lat, 'longitude': lon, 'elevation': elev}

            channels = []
            for chacode in chacodes:
                channels.append(inventory.channel.Channel(
                    code=chacode,
                    location_code=locode,
                    depth=depth,
                    **xyz
                    )
                )

            station = inventory.station.Station(
                code=stacode,
                channels=channels,
                total_number_of_channels=ncha,
                site=site,
                creation_date=creation_date,
                **xyz
                )
            network = inventory.network.Network(
                code=netcode,
                stations=[station],
                )

            inv += Inventory(networks=[network], source=source)

    return inv


def sachd(reftime='2019-01-01', **kwargs):
    """
    :param kwargs: for SAC
    """
    hd = obspy.core.trace.Stats({
        # Set SAC header
        'sac': obspy.core.util.attribdict.AttribDict(**kwargs),
    })

    hd.update({
        # Stats has a higher priority for attributes common to sac
        'station': hd.sac.kstnm,
        'network': getattr(hd.sac, 'knetwk', -12345),
        'npts': hd.sac.npts,
        'delta': hd.sac.delta,
        'starttime': obspy.core.UTCDateTime(reftime) + hd.sac.b,
    })

    hd.sac.update({
        'e': hd.sac.b + hd.npts*hd.delta,
    })

    return hd


def slowness2velocity(sn, fill_value=0, dtype='float64'):
    """
    Slowness to velocity.
    """
    vel = np.full(sn.shape, fill_value, dtype)
    ind = (sn != 0)
    vel[ind] = 1 / sn[ind]

    return vel


def velocity2slowness(vel, fill_value=0, dtype='float64'):
    """
    Velocity to slowness.
    """
    sn = slowness2velocity(vel, fill_value, dtype)

    return sn


def wrap_lon(lon, wrap_lon=True):
    """
    https://github.com/mathause/regionmask/blob/791b3d510b687d9dbb045db8ca353f1cbb627919/regionmask/core/mask.py#L36

    :param wrap_lon: 180 or 360
    """
    lon = np.asarray(lon)
    new_lon = lon.copy()
    wl = int(wrap_lon)
    if (wl == 180) or (lon.max() > 180 and wl != 360):
        new_lon[new_lon > 180] -= 360
    elif (wl == 360) or (lon.min() < 0 and wl != 180):
        new_lon[new_lon < 0] += 360

    return new_lon


def dist_one2many(lon0, lat0, lons, lats):
    """
    Distance between one point and many points.
    """
    lon0 = np.broadcast_to(lon0, lons.shape)
    lat0 = np.broadcast_to(lat0, lons.shape)
    dist = geod_inv(lon0, lat0, lons, lats)[0]

    return dist


def transect(lon1, lat1, lon2, lat2,
             step=None, npts=None, **kwargs):
    """
    Return coordinates along a great circle.

    :param ends: If include initial and terminus points
    :param npts: Number of points in the transect.
    :param step: Transect interval in km.
    """
    ends = kwargs.get('ends', True)

    lon1 = wrap_lon(lon1, 360)
    lon2 = wrap_lon(lon2, 360)
    length = geod_inv(lon1, lat1, lon2, lat2)[0]
    if npts is None:
        npts = np.ceil(length / step)
    if ends:
        npts -= 2
    lonlats = np.array(GEOD.npts(lon1, lat1, lon2, lat2, npts=npts))
    lon = wrap_lon(lonlats[:, 0], 360)
    lat = lonlats[:, 1]
    dist = dist_one2many(lon1, lat1, lon, lat)

    if ends:
        lon = np.concatenate([[lon1], lon, [lon2]])
        lat = np.concatenate([[lat1], lat, [lat2]])
        dist = np.concatenate([[0], dist, [length]])

    return lon, lat, dist


def interp2d(x, y, z, xi, yi, **kwargs):
    """
    :param isgrid: Use RegularGridInterpolator if True otherwise griddata
    """
    isgrid = kwargs.get('isgrid', True)
    method = kwargs.get('method', 'nearest')

    if isgrid:
        zitp = sp.interpolate.RegularGridInterpolator(
            points=(x, y),
            values=z,
            method=method,
        )(np.column_stack([xi, yi]))
    else:
        zitp = sp.interpolate.griddata(
            points=(x, y),
            values=z,
            xi=(xi, yi),
            method=method,
        )

    return zitp


def interp_transect(lon1, lat1, lon2, lat2, x, y, z,
                    kwargs_interp={}, **kwargs):
    """
    Interpolate values along a transect.
    """
    xi, yi, dist = transect(lon1, lat1, lon2, lat2, **kwargs)
    zitp = interp2d(x, y, z, xi, yi, **kwargs_interp)

    return xi, yi, dist, zitp
