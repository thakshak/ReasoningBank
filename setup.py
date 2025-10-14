import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="reasoningbank",
    version="0.0.1",
    author="Jules",
    author_email="jules@example.com",
    description="A Python library for implementing the ReasoningBank memory framework.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/reasoningbank",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
