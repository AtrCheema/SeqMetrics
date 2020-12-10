# -*- coding: utf-8 -*-
# Copyright (C) 2020  Ather Abbas
from setuptools import setup

with open("README.md", "r") as fd:
    long_desc = fd.read()

with open('version.py') as fv:
    exec(fv.read())

setup(
    name='TSErrors',

    version=1.3,

    description='TSErrors: Various errors for time-series data',
    long_description=long_desc,
    long_description_content_type="text/markdown",

    url='https://github.com/AtrCheema/TSErrors',

    author='Ather Abbas',
    author_email='ather_abbas786@yahoo.com',

    license='GPLv3',

    classifiers=[
        'Development Status :: 4 - Beta',

        'Natural Language :: English',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',

        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',


        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython'
    ],

    packages=['TSErrors'],

    install_requires=[
        'numpy'
    ],
)
