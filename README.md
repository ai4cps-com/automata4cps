# Automata4CPS 

Automata4CPS is a Python package for learning and analysis 
of the behavior of hybrid dynamical systems, with the focus on 
Cyber-Physical Systems (CPS).
The code was developed for several research publications ([bibtex](cite.bib)).

-   Website (www.ai4cps.com/automata4cps - under construction)
-   Contact ([contact@ai4cps.com](mailto:contact@ai4cps.com))


## Simple example

```python
import automata4cps as at

A = at.Automaton()
A.add_states_from(["s1", "s2", "s3"])
A.add_transitions_from([("s1", "s2", "e1"),
                  ("s2", "s3", "e1"),
                  ("s3", "s1", "e2")])

print(A)
A.view_plotly().show()
 ```

## Jupyter notebook examples

- [Conveyor system SFOWL discrete data analysis](notebooks/Conveyors_SFOWL_discrete.ipynb)
- [Conveyor system SFOWL continuous data analysis](notebooks/Conveyors_SFOWL_cont.ipynb)


## Install

To install Automata4CPS:

```
pip install ...
```

## Data

In folder "data" there are several datasets which can be easily loaded using examples module.
E.g.

```python

from automata4cps import examples

discrete_data, timestamp_col, discrete_vars = examples.conveyor_system_sfowl("discrete")
```
will load a dataset of a conveyor system from the SFOWL.

## Bugs
If you find any bugs, please contact us at [bugs@ai4cps.com](mailto:bugs@ai4cps.com).


## License

See [LICENSE](LICENSE).  
If you use this code in your research please [cite](cite.bib) our work. 
