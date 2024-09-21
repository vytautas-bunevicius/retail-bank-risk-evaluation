"""
This module configures the setup for the `retail_bank_risk` package. It
utilizes `setuptools` to define the package metadata and dependencies for
distribution and installation.

The `retail_bank_risk` package is designed for evaluating risk in a retail
banking context. The setup script includes the following key configurations:

- `name`: The name of the package.
- `version`: The current version of the package.
- `packages`: Specifies the packages to include, located in the `src`
  directory.
- `package_dir`: Maps the package root to the `src` directory.
- `install_requires`: Lists the dependencies required for the package,
  as specified in the `requirements.txt` file.
- `description`: A brief description of the package.
- `long_description`: A detailed description of the package, read from the
  `README.md` file.
- `long_description_content_type`: Specifies that the long description is in
  Markdown format.

This module is intended to be run during the package installation process to
properly configure the `retail_bank_risk` package for use.
"""

import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
REQUIREMENTS = (HERE / "requirements.txt").read_text().splitlines()

setup(
    name="retail_bank_risk",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=REQUIREMENTS,
    description="Retail Bank Risk Evaluation",
    long_description=README,
    long_description_content_type="text/markdown",
)
