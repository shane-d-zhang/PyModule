"""System related tools."""
import os
import pathlib
import logging.config
from subprocess import Popen, PIPE

import yaml


def log_cmd(cmd, logger):
    """
    Pass stdout to logger.
    https://www.endpoint.com/blog/2015/01/28/getting-realtime-output-using-python
    """
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    while True:
        out = p.stdout.readline()
        if p.poll() is not None:
            break
        if out:
            logger.info(out.strip().decode('utf-8'))

    return p.poll()


def setup_logging(fpar='logging.yml',
                  env_key='LOG_CFG',
                  default_level=logging.INFO):
    """
    Setup logging configuration.
    https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/

    :param fpar:
    :param env_key: environment variable
        e.g. LOG_CFG=my_logging.yml python my_server.py
    """
    value = os.getenv(env_key, None)
    path = value if value is not None else fpar

    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        print('Warning: no logging configuration file specified.')
        logging.basicConfig(level=default_level)

    return


def mkdir(path):
    """
    Meke intermediate directories and OK if exists.
    https://stackoverflow.com/a/14364249/8877268
    """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    return


def rm_flst(flst):
    """
    Remove files in a lst.
    """
    if flst is None:
        print("Warning: no files in list!")
    else:
        for file in flst:
            os.remove(file)

    return
