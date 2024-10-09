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

In case you want to test against `gpt <https://github.com/lehner/gpt>`_,
``gpt`` must be installed on your system. Assume that it is available as the
module ``gpt``. Then one can run the tests using::

    module load gpt      # Assumes that gpt is available as a module
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src    # Assuming that you are in the root directory of the project
    pytest test/

Some tests are marked as slow. These tests are skipped by default, they can
take a long time to run. To run these tests, use::

    pytest --runslow test/
    # or
    tox -- --runslow

One can mark tests as *selected* by using the decorator
``@pytest.mark.selected``. Then one can run only the selected tests using::

    pytest --onlyselected test/
    # or
    tox -- --onlyselected
    
This may be useful to run only a subset of tests that are relevant for
a specific change.

How Tests are Organized
-----------------------


- All tests are in the ``test/`` directory. The tests are organized in files 
    that correspond either to a module or a topic. The test files are prefixed
    with ``test_XX`` where ``XX`` is a number that defines the order in which
    the tests are run.

- Some tests are benchmarks that can be used to evaluate the performance of
    the current code or some changes. These tests are prefixed with ``test_bench_XX``.

- There are tests that test functionality that is not strictly part of ``qcd_ml``.
    These tests are in the ``extra_tests/`` directory.

Writing Tests
-------------

The best way to get started with writing tests is to look at the existing tests.
Typically a test is a single function prefixed with ``test_`` that thests one feature.
There are several usefull assets available as fixtures. The most useful assets are

- ``config_1500``: A gauge configuration on a :math:`8^3 \times 16` lattice.
- ``config_1200``: A gauge configuration on a :math:`8^3 \times 16` lattice.
- ``config_1500_gtrans_1200mu0``: The gauge field ``config_1500`` gauge transformed 
  using  :math:`U_{0}(x)` from ``config_1200``.

A test should be self contained and easy to understand. Ideally, it explains a feature.
Typically the test should end with an assertion that checks the expected result.
