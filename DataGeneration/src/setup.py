# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

from setuptools import setup, find_packages

setup(
    name='Dkr5G',
    version='0.0.1',
    description='Docker 5G network environment simulation',
    author='Mattia Milani',
    author_email='mattia.milani@nokia.com',
    packages=find_packages(exclude=['Dkr5G.tests*', 'ez_setup']),
    install_requires=[
        'cython',
        'PyYAML',
        'argparse',
        'networkx'
    ],
    extras_require={'plotting': ['matplotlib', 'seaborn']},
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'pytest-mock'],
    include_package_data=True,
    package_data={'Dkr5G': ['conf/*.cfg', 'templates/*.template', 'samples/*']},
    entry_points={
        'console_scripts': ['dkr5g=Dkr5G.Dkr5G:main']
    }
)

