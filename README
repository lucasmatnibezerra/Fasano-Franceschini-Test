# Fasano-Franceschini Test Documentation

This document presents the *Python implementation* of the multivariate Kolmogorov–Smirnov test, known as the **Fasano & Franceschini** test, based on the original papers and the R implementation by Puritz, Ness-Cohn & Braun (2023).

<img src="src\figures\fasano-logo.png" alt="Fasano & Franceschini Logo" width="200" align="right"/>

---
## 1. Overview

The Fasano-Franceschini (FF) test is a multivariate generalization of the Kolmogorov–Smirnov test for comparing two independent samples of arbitrary size and assessing whether they are drawn from the same distribution. It is especially useful in high dimension, where univariate tests may not capture cyclical differences.

- **Code author**: Lucas Matni Bezerra
- **Original method**: [Fasano & Franceschini (1987)](https://watermark.silverchair.com/mnras225-0155.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA1gwggNUBgkqhkiG9w0BBwagggNFMIIDQQIBADCCAzoGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM26FczWrIyxWgBDelAgEQgIIDC1f0kYaduXEC9kFh6jeiLZ1-vjmUCHikTreOLj8thhsI7pABCLUKKt5syshvkzC7jzvTDNlgwKS32dEhdl7Jp6A5iGOnKCrjYhCkukOXOGd9nKVOfHUlPD6M_YK5UqhKbSbOwDhtMUmP83jUc5_x3U2v7rDr-7A1iIma32Ac3TB4QQqox8LnTlvLoy5CIDcIOPa2JcrdPa0OB18ll6CvUI7zQBuKyI6G3SvqoxhN0LJ14SQcbtoY_pILFkrixFPa4eU2OlonVdW-VijoROLOmRTtpi_BtZqj-u_iV3_GWg0zmAfjKsOKJXXHjZq7XkqQ-xiv_F1BoY5K_vI3ypfEY2fR4iUkEyPOKlF6IK8m0c2pkcK-Wnw4BYSk2JGyrR2nHUNtrYu4t829WYAbRNJkgsEtWM7habdNqumheh72cCUA8y_izCdOMdWzSv3brBPahGQHcFbsmBEryAr6OnJtDmQoQJexO8-G0I_Mg1WfISjvwwQehtw2HPWgwJpCVhqU1nZMvZ6sAk-1N4KPzV9HRiWPfXISsRejhcQlhtq8Fh2GSW1LD7c_t_ISBCkKj5Z_dYiu4uPM-2P9ubjX31tF1goeae7zjuEZzXgoWcXbFssqTib7-Do22p06cy5yTmXIibjx6kuUGVZh6zUS_7JGBlHtXs9IVTSsdjCs7KY1HkOUuVUS8-RAYOhZjtwe-s98OkvsxXWYxmYgqa3oEJ-87Wc4C3CjFpcXqKdRt-vZ89yGEjLrl-sqjpkvolD6KKwy4P7NKWgLXVaikwYQ721upSQOrk28MBcel-hIyQnfqlFbeZU5wsn3WQRKbdmcqSjd3WhTSelwbFnePwjTIdTThrczhjhHtxWd8nsQdg4ACNZRIA9K3ZOB35jJFK4DJe9I_QNQQ9IM1xr5V2TR5IntQf9V1Me1Vq_SRmaIB2Hu5B4deF9pAR3kimeUVU2YpRM_x3UbT1Wy4uXZMyglufzxvFSeihMmk7b4pI7OhPFROTpNOsWGz4QWwdUzqis_zeRZ4Y7WUGEPXEI9fqRQ)
- **Reference R implementation**: [Puritz, C., Ness-Cohn, E. & Braun, R. (2023). *fasano.franceschini.test: An Implementation of a Multivariate KS Test in R*. The R Journal, 15(3), 159–171.](https://journal.r-project.org/articles/RJ-2023-067/#ref-ff1987)


---

## 2. Theoretical Background

1. **Peacock (1983)**: Introduced the D statistic for KS testing in several dimensions, with emphasis on separation by orthogonal hyperplanes. 2. **Fasano & Franceschini (1987)**: Refined the approach, using partitioning into quadrants (or orthants) defined by combined reference points of the two samples.

2. **Puritz, Ness-Cohn & Braun (2023)**: Provided an efficient implementation in R, which serves as the basis for the Python version presented here.

--- 
## 3. Project Structure

The project is organized into modules (inspired bu the implementation in R) that implement the following functionalities:
```
$PROJECT_ROOT
├── setup.py
├── LICENSE
├── README       
└── src
    ├── examples
        └── script.py       # Example script demonstrating usage of the package with random data in csv format
    ├── ff_pkg.egg-info
    └── ff_pkg              # The main package containing the Fasano-Franceschini test implementation
        ├── __init__.py
        ├── distance.py     # Functions for calculating distances between points
        ├── ff_core.py      # Core functions for the Fasano-Franceschini test
        ├── matrix_util.py  # Utility functions for matrix operations
        └── range_tree.py   # Implementation of a range tree for efficient querying
```

## 4. How to use

To integrate and use the **ff-pkg** package in your Python projects, follow these steps:

1. **Installation**

   In directory of ROOT PROJECT (where is `setup.py` or `pyproject.toml`), run:
   ```bash
   git clone https://github.com/lucasmatnibezerra/Fasano-Franceschini-Test.git
   cd Fasano-Franceschini-Test
   # Then, install the package using pip:
   # Installation in edit mode (development)
   pip install -e .
   ```
2. **Import the package**

   In your Python script, import the Fasano-Franceschini test package:
   ```python
   from ff_pkg.ff_core import ff_test_statistic
   from ff_pkg.ff_core import permutation_test, permutation_test_parallel
   ```

3. **Example usage:**
   ```python
   import numpy as np
   from ff_pkg.ff_core import ff_test_statistic, permutation_test

   # Generate random data
   X = np.random.randn(100, 3)
   Y = np.random.randn(100, 3) + 0.5

   # D Statistic
   D = ff_test_statistic(X, Y, method='r')
   print(f"D = {D}")

   # Permutation test (p-value)
   zg, ze, p = permutation_test(
    X, Y,
    n_permutations=1000,
    method='r',
    seed=42,
    verbose=True, # You can set verbose=False to suppress the output
   )
   print(f"p-value ≈ {p:.4f}")
   ```