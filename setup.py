from setuptools import setup, find_packages

setup(
    name='nontology',
    version='0.0.8',
    author='Cassandra Jacobs',
    author_email='jacobs.cassandra.l@gmail.com',
    license='MIT',
    url='https://github.com/BayesForDays/nontology',
    description='Functions for creating your own nontological `embeddings`',
    packages=find_packages(),
    long_description='Functions for creating your own nontological `embeddings`',
    keywords=['nontology', 'knowledge graph', 'api'],
    classifiers=[
        'Intended Audience :: Developers',
    ],
    install_requires=[
        'numpy>=1.14.5',
        'pandas>=0.20.2',
        'scikit-learn>=0.20.dev0', # not currently working w/o additional nonsense
        'nltk>=3.3',
        'scipy>=1.1.0'
    ]
)
