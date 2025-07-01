from setuptools import setup, find_packages
import os
import sys

PACKAGE_NAME = "grgapp"

THISDIR = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(THISDIR, "requirements.txt")) as f:
    requirements = list(map(str.strip, f))
with open(os.path.join(THISDIR, "README.md")) as f:
    long_description = f.read()

from grgapp.version import GRGAPP_VERSION

setup(
    name=PACKAGE_NAME,
    packages=find_packages(),
    package_data={
        "grgapp": [os.path.join(THISDIR, "requirements.txt")],
    },
    version=GRGAPP_VERSION,
    description="Statistical and population genetics methods on GRG",
    author="Drew DeHaas",
    author_email="",
    url="https://aprilweilab.github.io/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    install_requires=[
        requirements,
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": ["grgapp=grgapp.cli.main:main"],
    },
)
