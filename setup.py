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
    return ".".join(version.split(".")[:2]) + ".*"


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
    ]
)
