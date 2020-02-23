"""File IO utilities."""
import os
import pickle

import yaml


def ryml(fin):
    """
    Load YAML.
    """
    with open(fin, 'r') as f:
        return yaml.safe_load(f)


def wpk(fnm, data, protocol=pickle.HIGHEST_PROTOCOL):
    """
    Save into pickle.
    """
    if not fnm.endswith('.pickle'):
        fnm += '.pickle'
    with open(fnm, 'wb') as f:
        return pickle.dump(data, f, protocol)


def rpk(fnm):
    """
    Read pickle.
    """
    if not fnm.endswith('.pickle'):
        fnm += '.pickle'
    with open(fnm, 'rb') as f:
        return pickle.load(f)


def rcol(fname, col, rtype='list', delimiter=None, comments='#'):
    """
    Read columns from file.

    :param fname: input file
    :param col: columns to get
    :type col: int or sequence
    :param rtype: return list or set
    """
    try:
        col = list(col)
    except TypeError:
        col = [col]

    ncol = len(col)
    val = [[] for i in range(ncol)]
    with open(fname, 'r') as f:
        for line in f:
            _v = line.rstrip().split(delimiter)
            if comments in _v[0]:
                continue
            for k in range(ncol):
                val[k].append(_v[col[k]])

    if rtype == 'list':
        return val
    elif rtype == 'set':
        return [set(v) for v in val]
    else:
        raise ValueError(f'Unknown type {rtype}')


def flen(fnm):
    """
    Return # of lines in file.

    https://stackoverflow.com/a/845081/8877268
    """
    with open(fnm) as f:
        for i, _ in enumerate(f):
            pass

    return i + 1


def file_is_empty(path):
    """
    https://stackoverflow.com/a/52259169/8877268
    """
    return os.stat(path).st_size == 0
