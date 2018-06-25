from setuptools import setup, find_packages

version = {}
with open("emotioncf/version.py") as f:
    exec(f.read(), version)

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

extra_setuptools_args = dict(
    tests_require=['pytest']
)

setup(
    name="emotioncf",
    version=__version__,
    description='A Python package for performing Collaborative Filtering on ',
                'sparse emotion ratings",
    maintainer='Luke Chang',
    maintainer_email='luke.j.chang@dartmouth.edu',
    url='http://github.com/ljchang/emotionCF',
    install_requires=requirements,
    packages=find_packages(exclude=['emotioncf/tests']),
    license='MIT',
    keywords = ['emotion', 'collaborative filtering', 'recommender','machine-learning'],
    classifiers = [
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License"
    ],
    **extra_setuptools_args
)
