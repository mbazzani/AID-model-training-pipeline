"""Example setup file
"""
from setuptools import setup, find_packages

setup(
    name="AID_model_training",
    version="0.0.0.1",
    author="UCSD Engineers for Exploration",
    author_email="e4e@eng.ucsd.edu",
    entry_points={},
    packages=find_packages(),
    install_requires=[
        "torch==2.0.0",
        "torchaudio==2.0.1",
        "torchvision==0.15.1",
        "pandas==2.0.1",
        "librosa==0.10.0.post2",
        "scipy==1.10.1",
        "soundfile==0.12.1",
        "matplotlib==3.7.1",
    ],
    extras_require={
        "dev": [
            "pytest",
            "coverage",
            "pylint",
            "wheel",
        ]
    },
)
