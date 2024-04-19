# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

from setuptools import setup, find_packages

setup(
    name='SNN',
    version='0.2.0',
    description='Siamese Neural Network self learning FrameWork',
    author='Mattia Milani',
    author_email='mattia.milani@nokia.com',
    packages=find_packages(exclude=['SNN.tests*', 'ez_setup']),
    package_data={"SNN2": ['samples/conf.yaml']},
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'pytest-mock'],
    entry_points={
        'console_scripts': ["snn=SNN2.main:main"]
    }
)

