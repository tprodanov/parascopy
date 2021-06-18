import setuptools

from src.parascopy import __version__, __author__, __license__

with open('README.md') as inp:
    long_description = inp.read()

with open('requirements.txt') as inp:
    requirements = list(map(str.strip, inp))


setuptools.setup(
    name='parascopy',
    version=__version__,
    author=__author__,
    license=__license__,
    description='Robust and accurate estimation of paralog-specific copy number for duplicated genes using WGS',
    long_description=long_description,
    url='https://github.com/tprodanov/parascopy',

    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    python_requires='>=3.6',
    install_requires=requirements,
    include_package_data=True,

    entry_points = dict(console_scripts=['parascopy=parascopy.entry_point:main']),
    )
