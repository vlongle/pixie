from setuptools import setup, find_packages


INSTALL_REQUIRES = [
    'google-generativeai',
    'ipykernel',
    'matplotlib',
    'markdown2',
    'scikit-learn',
    'opencv-python',
    'hydra-core',
    'termcolor',
    'python-dotenv',
    'plotly',
    'nbformat',
    'openai',
    'anthropic',
    'moviepy',
    'astor',
]




setup(
    name='vlmx',
    version='0.1.0',
    description='VLMX does everything and you just profit.',
    install_requires=INSTALL_REQUIRES, 
    packages=find_packages(),
    python_requires='>=3.8',
)
