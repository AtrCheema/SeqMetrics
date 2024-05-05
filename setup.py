# -*- coding: utf-8 -*-
# Copyright (C) 2020  Ather Abbas
from setuptools import setup

import os
fpath = os.path.join(os.getcwd(), "README.md")
if os.path.exists(fpath):
    with open(fpath, "r") as fd:
        long_desc = fd.read()

setup(
    name='SeqMetrics',

    version="2.0.0",

    description='SeqMetrics: a unified library for performance metrics calculation in Python',
    long_description=long_desc,
    long_description_content_type="text/markdown",

    url='https://github.com/AtrCheema/SeqMetrics',

    author='Ather Abbas',
    author_email='ather_abbas786@yahoo.com',

    classifiers=[
        'Development Status :: 4 - Beta',

        'Natural Language :: English',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',

        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',



        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],

    packages=['SeqMetrics'],

    install_requires=[
        'numpy',
        'scipy',
    ],
    extras_require={
        'all': ["numpy", "scipy", "easy_mpl"], 
    }
)
