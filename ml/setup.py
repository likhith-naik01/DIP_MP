from setuptools import setup, find_packages

setup(
    name="dip-core",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "scikit-image>=0.19.0",
        "opencv-python>=4.7.0",
        "scipy>=1.9.0",
        "matplotlib>=3.5.0",
        "pandas>=1.5.0",
        "pillow>=9.0.0",
    ],
)
