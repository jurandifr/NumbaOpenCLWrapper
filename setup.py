
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="numba-opencl",
    version="0.2.0",
    author="Replit User",
    author_email="user@replit.com",
    description="Extensão Numba para OpenCL que emula a API numba.cuda",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/user/numba-opencl",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.20.0",
        "pyopencl>=2022.1",
        "numba>=0.55.0",
        "siphash24>=1.0.0",
        "prettytable>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
        ],
    },
)
