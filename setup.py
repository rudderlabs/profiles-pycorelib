from setuptools import setup, find_packages

setup(
    name='profiles-pycorelib',
    version='0.2.0',
    author='Shubham Mehra',
    author_email='shubhammehra@ruddersatck.com',
    description='A Python Native package that registers the core python models',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        "profiles-rudderstack==0.10.4"
    ]
)