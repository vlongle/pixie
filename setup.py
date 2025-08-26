from setuptools import setup
from setuptools import setup, find_packages


INSTALL_REQUIRES = [
    "objaverse",
    "sentence-transformers",
    # "torch",
    # "torchvision",
    "PyMCubes==0.1.4",
    "hydra-core",
    "omegaconf",
    "trimesh",
    "plyfile",
    "matplotlib",
    "numpy==1.24.4",
    "params-proto",
    "python-slugify",
    "warp_lang==0.10.1",
    "taichi==1.5.0",
    "dotenv",
    'timm==1.0.13',
    # for qwen
    'qwen-vl-utils[decord]',
    'accelerate',
    # https://github.com/QwenLM/Qwen2.5-VL/issues/723
    'transformers @ git+https://github.com/huggingface/transformers',
    'streamlit',
    'huggingface_hub',
    'colorlog',
    'seaborn',
    'umap-learn',
]

setup(
    name='pixie',
    version='0.0.0',
    description='pixie',
    author='Long Le',
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3.10',
    packages=find_packages(
        include=['pixie', 'pixie.*']),

)