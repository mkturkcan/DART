from setuptools import setup, find_packages

setup(
    name="dart-sam3",
    version="0.1.0",
    description="Detect Anything in Real Time: Real-time object detection using frontier object detection models",
    author="mkturkcan",
    url="https://github.com/mkturkcan/DART",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.7.0",
        "torchvision>=0.22.0",
        "tensorrt>=10.9.0",
        "onnx>=1.20.1",
        "onnxsim",
        "scipy",
        "opencv-python",
        "numpy",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
)
