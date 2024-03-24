from setuptools import setup, find_packages

setup(
    name='TRACE',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas==2.1.4',
        'spotipy==2.23.0',
        'python-dotenv==1.0.0',
        'scikit-learn==1.3.2',
        'umap-learn==0.5.5',
        'plotly==5.18.0',
        'music-tag==0.4.3',
    ],
    entry_points={
        'console_scripts': [
            'trace=TRACE.module:main',
        ],
    },
)
