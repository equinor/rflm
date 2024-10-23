from setuptools import setup, find_packages

setup(
    name='rflm',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
    ],
    url='https://github.com/yourusername/my_package',
    description='A brief description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
)