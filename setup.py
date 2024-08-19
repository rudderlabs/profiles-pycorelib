from setuptools import setup, find_packages
from version import version


def limit_major_minor_version(version: str) -> str:
    """
    Returns a modified version string that limits the major and minor version numbers.

    Args:
        version (str): The original version string.

    Returns:
        str: The modified version string with only the major and minor version numbers.

    Example:
        >>> limit_major_minor_version("1.2.3")
        '1.2.*'
    """
    return ".".join(version.split(".")[:2]).replace('v', '') + ".*"


setup(
    name='profiles-pycorelib',
    version=version,
    author='Shubham Mehra',
    author_email='shubhammehra@ruddersatck.com',
    description='A Python Native package that registers the core python models',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        f"profiles-rudderstack=={limit_major_minor_version(version)}",
        "cexprtk>=0.4.1",
        "seaborn>=0.13.1",
        "matplotlib>=3.7.5",
        "pandas>=2.0.3,<2.2.0",
        "numpy>=1.24.4",
        "plotly>=5.22.0",
        "scipy>=1.11.0,<=1.11.4",
    ]
)
