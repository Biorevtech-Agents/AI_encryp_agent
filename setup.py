from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai_encryp_agent",
    version="0.1.0",
    author="Biorevtech-Agents",
    author_email="media@biorev.us",
    description="An autonomous AI agent for cryptocurrency trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Biorevtech-Agents/AI_encryp_agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires="==3.8.*",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ai_encryp_agent=src.main:main",
        ],
    },
) 