# Automata4CPS 

Automata4CPS is a Python package for learning and analysis 
of the behavior of hybrid dynamical systems, with the focus on 
Cyber-Physical Systems (CPS).
The code was developed for several research publications:

>-To be added-


-   Website (to be provided)
-   Contact (to be provided)


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


## Bugs

## License

See [LICENSE](LICENSE).
