from setuptools import setup, find_packages

# Read the content of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="data_driven_linearization",
    version="1.0.0",
    description="A dara-driven model reduction tool using linearization techniques.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Bálint Kaszás",
    author_email="bkaszas@ethz.ch",
    url="https://github.com/haller-group/data_driven_linearization",  # Update with your repository URL
    packages=find_packages(),

    extras_require={
        "dev": [
            "pytest>=5.3.5",
            "flake8>=3.7.9",
            "black>=19.10b0",
        ],
    },
    python_requires='>=3.6',
)
