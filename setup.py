from setuptools import setup, find_packages

version = {}
with open("neighbors/version.py") as f:
    exec(f.read(), version)

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

extra_setuptools_args = dict(tests_require=["pytest"])

setup(
    name="neighbors",
    version=version["__version__"],
    description="A Python package for performing collaborative filtering on social and emotion datasets",
    maintainer="Cosan Lab",
    maintainer_email="eshin.jolly@dartmouth.edu",
    url="http://github.com/cosanlab/neighbors",
    install_requires=requirements,
    packages=find_packages(exclude=["neighbors/tests"]),
    include_package_data=True,
    package_data={"": ["data/*.csv"]},
    license="MIT",
    keywords=["emotion", "collaborative filtering", "recommender", "machine-learning"],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
    ],
    **extra_setuptools_args
)
