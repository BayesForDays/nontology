from setuptools import setup, find_packages

setup(
    name='nontology',
    version='0.0.7',
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
        'numpy',
        'pandas<=0.20.2',
        'sklearn<=0.18.1', # won't support 2.7 going forward
        'scipy'
    ]
)
