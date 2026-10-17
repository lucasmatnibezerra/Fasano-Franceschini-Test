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
        "numpy>=1.23",
        "joblib>=1.2",
        # Imported at module level by ff_pkg.range_tree, which ff_pkg.ff_core
        # reaches through ff_pkg.distance. Mandatory even for method="b".
        "numba>=0.58",
        "Rtree>=1.0.0",
        # Used by ff_pkg.dim_reduction for the PCA projection.
        "scikit-learn>=1.2",
        "tqdm>=4.64",
        "pandas>=2.0",
    ],
    python_requires=">=3.10",
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
