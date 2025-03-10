from setuptools import setup, find_packages

setup(
    name="neural-rrt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "pygame",
        "matplotlib",
        "tqdm",
        "shapely",
        "pyyaml"
    ],
    python_requires=">=3.7",
    author="Huang xiaohai",
    author_email="huauangxiaohai99@126.com",
    description="Neural network enhanced RRT path planning",
    keywords="path planning, RRT, neural networks",
)
