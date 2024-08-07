from setuptools import setup, find_packages

setup(
    name="automata4cps",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here, for example:
        'dash','pandas', 'networkx', 'plotly', 'numpy', 'dash_daq', 'dash-bootstrap-components', 'pydotplus',
        'dash-cytoscape', 'simpy', 'mlflow', 'torch'
    ],
    author="Nemanja Hranisavljevic & Tom Westermann",
    author_email="nemanja@ai4cps.com",
    description="Tools for learning, plotting, analyzing etc of discrete, timed and hybrid automata.",
    url="https://github.com/ai4cps-com/automata4cps",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)