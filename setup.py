# setup.py
from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="interro",
    version="0.1.0",
    author="Sparsh Chakraborty",
    author_email="sparsh.chakraborty07@gmail.com",
    description="AI-powered code understanding and querying tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/codexx07/interro",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "interro=interro.cli:app",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)