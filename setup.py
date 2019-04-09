from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='dataloader',
    version='0.0.1',
    description='Custom DataLoader for PyTorch',
    long_description=Path('README.md').read_text(encoding='utf-8'),
    author='ttivy',
    author_email='25240747+ttivy@users.noreply.github.com',
    url='https://github.com/ttivy/dataloader',
    license=Path('LICENSE').read_text(encoding='utf-8'),
    packages=find_packages(exclude=('tests',)),
    test_suite='tests',
    install_requires=['torch'],
)
