from setuptools import setup, find_packages

setup(
    name="mujoco-robotics-playground",
    version="0.1.0",
    description="MuJoCo robotics simulation suite with PPO from scratch",
    author="",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    install_requires=[
        "mujoco>=3.0.0",
        "gymnasium>=1.0.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "video": ["imageio[ffmpeg]>=2.31.0"],
        "dev":   ["pytest>=7.4.0"],
    },
    entry_points={
        "console_scripts": [
            "mujoco-train    = scripts.train:main",
            "mujoco-evaluate = scripts.evaluate:main",
            "mujoco-demo     = scripts.demo:main",
        ],
    },
)
