# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, "../../src/")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'qcd_ml'
copyright = '2024, Daniel Knüttel, et al.'
author = 'Daniel Knüttel, et al.'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.doctest',
        'sphinx.ext.coverage',
        'sphinx.ext.imgmath',
        'sphinx.ext.ifconfig',
        'sphinx.ext.viewcode',
        ]

templates_path = ['_templates']
exclude_patterns = []

source_suffix = '.rst'
todo_include_todos = True

autodoc_mock_imports = ["gpt"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
