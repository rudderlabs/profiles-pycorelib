from setuptools import setup, find_packages

setup(
    name='profiles-pycorelib',
    version='0.3.0',
    author='Shubham Mehra',
    author_email='shubhammehra@ruddersatck.com',
    description='A Python Native package that registers the core python models',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        "profiles-rudderstack==0.10.5",
        "langchain>=0.0.352",
        "pandas>=2.0.3",
        "openai>=1.6.1",
        "langchain-google-genai>=0.0.5",
        "boto3>=1.34.12",
        "awscli>=1.32.12",
        "botocore>=1.34.12"
    ]
)