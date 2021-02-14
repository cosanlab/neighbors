from setuptools import setup, find_packages

version = {}
with open("emotioncf/version.py") as f:
    exec(f.read(), version)

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

extra_setuptools_args = dict(tests_require=["pytest"])

setup(
    name="emotioncf",
    version=version["__version__"],
    description="A Python package for performing Collaborative Filtering on sparse emotion ratings",
    maintainer="Cosan Lab",
    maintainer_email="eshin.jolly@dartmouth.edu",
    url="http://github.com/cosanlab/emotionCF",
    install_requires=requirements,
    packages=find_packages(exclude=["emotioncf/tests"]),
    license="MIT",
    keywords=["emotion", "collaborative filtering", "recommender", "machine-learning"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
    ],
    **extra_setuptools_args
)
