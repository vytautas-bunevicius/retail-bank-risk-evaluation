from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
REQUIREMENTS = (HERE / "requirements.txt").read_text().splitlines()

setup(
    name='retail_bank_risk',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=REQUIREMENTS,
    description="Retail Bank Risk Evaluation",
    long_description=README,
    long_description_content_type="text/markdown",
)
