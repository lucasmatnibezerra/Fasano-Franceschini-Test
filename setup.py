from setuptools import setup, find_packages

setup(
    name="ff-pkg",
    version="0.1.0",
    description="Implementation of Fasano–Franceschini test in Python",
    author="Lucas Matni Bezerra", 
    author_email="lucas.matni@itec.ufpa.br",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20",
        "rtree",
        "joblib",
        "tqdm",
        "pandas",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "ff-test=ff_pkg.ff_test:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
