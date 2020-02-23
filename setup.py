from setuptools import setup, find_packages


setup(
    name='PyModule',
    version='0.0.1',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'astropy',
        # 'basemap',
        # 'bottleneck',
        'h5py',
        'matplotlib',
        # 'netcdf4',
        # 'numba',
        'numpy',
        'obspy',
        # 'pycpt',
        'pyproj',
        'pyyaml',
        'scipy',
        'setuptools',
    ],
    author='Shane Zhang',
    author_email='shzh3924@colorado.edu',
    description='Personal Python Modules for Convenient Import',
    license='MIT',
    url='https://github.com/shane-d-zhang/PyModule',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    )
