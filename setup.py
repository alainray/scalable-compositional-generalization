from glob import glob
from setuptools import find_packages, setup

setup(
    name="visgen",
    version="0.0",
    description="Visual Generalization study.",
    packages=find_packages(include=["visgen", "visgen.*"]),
    data_files=[("configs", glob("configs/*.json"))],
    python_requires=">=3.9",
)
