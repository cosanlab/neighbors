from emotioncf.version import __version__
from setuptools import setup, find_packages

extra_setuptools_args = dict(
    tests_require=['pytest']
)

setup(
    name="emotioncf",
    version=__version__,
    description="Emotion Rating Collaborative Filtering",
    maintainer='Luke Chang',
    maintainer_email='luke.j.chang@dartmouth.edu',
    url='http://github.com/ljchang/emotionCF',
    install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn'],
    packages=find_packages(exclude=['emotioncf/tests']),
    license='MIT',
    # download_url='https://github.com/ljchang/emotionCF/archive/%s.tar.gz' %
    # __version__,
    **extra_setuptools_args
)