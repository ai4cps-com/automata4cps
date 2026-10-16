import io
import os
from setuptools import setup, find_packages
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
main_ns = {}
exec((BASE_DIR / "ml4cps" / "version.py").read_text(encoding="utf-8"), main_ns) # pylint: disable=exec-used, consider-using-with

def read_req_file(req_type):
    with open(os.path.join("requirements", f"{req_type}.txt"), encoding="utf-8") as fp:
        requires = (line.strip() for line in fp)
        return [req for req in requires if req and not req.startswith("#")]

setup(
    name="ml4cps",
    version=main_ns["__version__"],
    author="Nemanja Hranisavljevic & Tom Westermann",
    author_email="nemanja@ai4cps.com",
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
    license="MIT",
    description="Tools for learning, plotting, analyzing etc. of discrete, continuous, timed, and "
                "hybrid cyber-physical systems.",
    long_description=io.open("README.md", encoding="utf-8").read(), # pylint: disable=consider-using-with
    long_description_content_type="text/markdown",
    install_requires=read_req_file("install"),
    python_requires='>=3.6',
    extras_require={
        "dev": read_req_file("dev"),
    },
    entry_points={
        "console_scripts": [],
        "pytest11": [],
    },
    url="https://ml4cps.ai4cps.com",
    project_urls={
        "Documentation": "https://ml4cps.ai4cps.com",
        "Source": "https://github.com/ai4cps-com/ml4cps",
        "Issue Tracker": "https://github.com/ai4cps-com/ml4cps/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Manufacturing",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    data_files=[],

)
