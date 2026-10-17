[![Sponsor](https://img.shields.io/badge/Sponsor-❤-ff69b4)](https://github.com/sponsors/ai4cps-com)

# ML4CPS
ML4CPS is a Python package for learning and analysis 
of hybrid dynamical systems, with the focus on 
Cyber-Physical Systems (CPS).
The code was developed for several research publications ([bibtex](docs/cite.bib)).

-   Website ([ml4cps.ai4cps.com](http://ml4cps.ai4cps.com))
-   Contact ([contact@ai4cps.com](mailto:contact@ai4cps.com))


## Installation

Install the core package without PyTorch:

```sh
pip install ml4cps
```

For DEBTA models and tensor-based tools, install the optional PyTorch dependency:

```sh
pip install "ml4cps[torch]"
```

The `dev` extra also includes PyTorch for development and tests.

## Data
In the folder "data" there are several datasets that can be easily loaded using the [`ml4cps.examples`](ml4cps/examples/examples.py) module.


## Bugs
If you find any bugs, please contact us at [bugs@ai4cps.com](mailto:bugs@ai4cps.com).


## License
See [LICENSE](LICENSE).  
If you use this code in your research, please [cite](docs/cite.bib) our work. 
