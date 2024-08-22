# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the path to your package
sys.path.insert(0, os.path.abspath('../../../'))


#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Automata4CPS'
copyright = '2024, Nemanja Hranisavljevic, Tom Westermann'
author = 'Nemanja Hranisavljevic, Tom Westermann'
release = '0.1.11'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # To automatically generate documentation from docstrings
    'sphinx.ext.napoleon',  # To support Google-style or NumPy-style docstrings
    'sphinx.ext.autosummary'
]


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

html_theme_options = {
    'page_width': 'auto',
    'sidebar_width': '300px',
    'body_max_width': 'auto',
}

# Automatically generate summary pages for autosummary directives
autosummary_generate = True