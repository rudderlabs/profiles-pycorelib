from setuptools import setup, find_packages

setup(
    name='profiles-pycorelib',
    version='0.2.2',
    author='Shubham Mehra',
    author_email='shubhammehra@ruddersatck.com',
    description='A Python Native package that registers the core python models',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
       "profiles-rudderstack==0.10.6",'pandas>=1.4.3','matplotlib>=3.7.1', 'cexprtk'
    ]
)
