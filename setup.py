# This file is part of Discrete and continuous learning machines.
# Copyright (C) 2020- Mattia G. Bergomi, Patrizio Frosini, Pietro Vertechi
#
# Discrete and continuous learning machines is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please use the tools available at
# https://gitlab.com/mattia.bergomi.
#
# [1]

import sys
from setuptools import find_packages, setup
import pathlib

CURRENT_PYTHON_VERSION = sys.version_info[:2]
MIN_REQUIRED_PYTHON_VERSION = (3, 10) # COMPATIBLE PYTHON VERSION
if CURRENT_PYTHON_VERSION < MIN_REQUIRED_PYTHON_VERSION:
    sys.stderr.write("""
==========================
Unsupported Python version
==========================
This version of hypergraph_machines requires Python {}.{}, but you're trying to
install it on Python {}.{}.
""".format(*(MIN_REQUIRED_PYTHON_VERSION + CURRENT_PYTHON_VERSION)))
    sys.exit(1)

requirements = (pathlib.Path(__file__).parent / "requirements.txt").read_text().splitlines()
EXCLUDE_FROM_PACKAGES = []

setup(
    name='hypergraph_machines',
    version='0.0.1',
    python_requires='>={}.{}'.format(*MIN_REQUIRED_PYTHON_VERSION),
    url='',
    author='',
    author_email='',
    description=(''),
    license='',
    packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
    include_package_data=True,
    install_requires=requirements,
    entry_points={},
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering'
        'Topic :: Scientific/Engineering :: Machine Learning',
        'Topic :: Scientific/Engineering :: Machine cognition',
    ],
)
