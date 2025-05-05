from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()
    
setup(
    name="MLE2E SetUp",
    author="Maesak Delbar",
    author_email="maesak.delbar@gmail.com",
    version="0.1",
    packages=find_packages(),
    install_requires= requirements,
)