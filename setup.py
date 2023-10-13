from setuptools import setup, find_packages

setup(
    name='profiles-pycorelib',
    version='0.1.0',
    author='Shubham Mehra',
    author_email='shubhammehra@ruddersatck.com',
    description='A Python Native package that registers the core python models',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        "profiles-rudderstack@git+https://github.com/rudderlabs/pywht@wht-grpc-0.1.1#subdirectory=profiles_rudderstack"
    ]
)