# setup.py
from setuptools import setup, find_packages

setup(
    name='embeddingpaw',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'openai',
        'scikit-learn',
        'pyecharts'
    ],
    url='https://github.com/yourusername/embeddingpaw',
    license='MIT',
    author='Kira Pgr',
    author_email='your-email@example.com',
    description='A library for playing with embeddings',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)