from setuptools import setup, find_packages

setup(
    name='profiles-rudderstack-common-column-union',
    version='1.0.0',
    author='Shubham Mehra',
    author_email='shubhammehra@ruddersatck.com',
    description='A Python Native model package that implements the common_column_union model',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
)