HOWTO -- Testing
================

We have a test suite using `pytest <https://docs.pytest.org/en/stable/>`_ that
tests all important functions and classes.

Tests are functions organized in test files in the ``test/`` directory. 

To make sure tests run in a well-defined we recommend running the tests using
`tox <https://tox.wiki/>`_. The file ``tox.ini`` defines the standard testing
environment and a special environment that we use to build this documentation.


Running the Tests
-----------------

To run the tests, it is sufficient to install ``tox``, for instance using ``pip``::

    python -m pip install --user tox
    python -m tox --help

Then the tests can be run using::

    python -m tox

