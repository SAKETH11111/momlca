#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="momlca",
    version="0.0.1",
    description="MoML-CA project scaffold based on lightning-hydra-template",
    author="Saketh",
    author_email="",
    url="https://github.com/saketh/momlca",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
