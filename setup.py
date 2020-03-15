import io
import os
import re

from setuptools import find_packages, setup

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with io.open(os.path.join(HERE, *parts), encoding="utf-8") as f:
        return f.read()


VERSION = re.search(r'__version__ = "([^"]+)"', read("lindex", "__init__.py")).group(1)

setup(
    name="lindex",
    version=VERSION,
    description="Learned index structures for sorted, in-memory data",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="planckestein",
    license="MIT",
    packages=find_packages(exclude=("tests", "scripts")),
    python_requires=">=3.7",
    install_requires=["numpy>=1.18"],
    extras_require={
        "plot": ["matplotlib>=3.2"],
        "test": ["pytest>=5.4"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Database :: Database Engines/Servers",
    ],
)
