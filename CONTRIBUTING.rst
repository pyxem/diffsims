=================
Contributor Guide
=================

This guide is intended to get new developers started with contributing to diffsims.

Many potential contributors will be scientists with much expert knowledge but
potentially little experience with open-source code development. This guide is primarily
aimed at this audience, helping to reduce the barrier to contribution.

We have a `Code of Conduct
<https://github.com/pyxem/diffsims/blob/master/.github/CODE_OF_CONDUCT.md>`_ that must
be honoured by contributors.

Start using diffsims
====================

The best way to start understanding how diffsims works is to use it.

For developing the code the home of diffsims is on GitHub and you'll see that a lot of
this guide boils down to using that platform well. so visit the following link and poke
around the code, issues, and pull requests (PRs): `diffsims
on GitHub <https://github.com/pyxem/diffsims>`_.

It's probably also worth visiting the `GitHub guides <https://docs.github.com/en>`_ to
get a feel for the terminology.

In brief, to give you a hint on the terminology to search for, the contribution pattern
is:

1. Setup git/GitHub if you don't have it.
2. Fork diffsims on GitHub.
3. Checkout your fork on your local machine.
4. Create a new branch locally where you will make your changes.
5. Push the local changes to your own github fork.
6. Create a PR to the official diffsims repository.

Note: You cannot mess up the main diffsims project. So when you're starting out be
confident to play, get it wrong, and if it all goes wrong you can always get a fresh
install of diffsims!

PS: If you choose to develop in Windows/Mac you may find the `Github Desktop
<https://desktop.github.com>`_ useful.

Questions?
==========

Open source projects are all about community - we put in much effort to make good tools
available to all and most people are happy to help others start out. Everyone had to
start at some point and the philosophy of these projects centers around the fact that we
can do better by working together.

Much of the conversation happens in 'public' using the 'issues' pages on
`GitHub <https://github.com/pyxem/diffsims/issues>`_ -- doing things in public can be
scary but it ensures that issues are identified and logged until dealt with. This is
also a good place to make a proposal for some new feature or tool that you want to work
on.

Good coding practice
====================

The most important aspects of good coding practice are: (1) to work in manageable
branches, (2) develop a good code style, (3) write tests for new functions, and (4)
document what the code does. Tips on these points are provided below.

Use git to work in manageable branches
--------------------------------------

Git is an open source "version control" system that enables you to can separate out your
modifications to the code into many versions (called branches) and switch between them
easily. Later you can choose which version you want to have integrated into diffsims.

You can learn all about Git `here <https://www.git-scm.com/about>`_!

The most important thing is to separate your contributions so that each branch is a
small advancement on the "master" code or on another branch.

Get the style right
-------------------

diffsims closely follows the Style Guide for Python Code - these are just some rules for
consistency that you can read all about in the `Python Style Guide
<https://peps.python.org/pep-0008/>`_.

Please run the latest version of
`black <https://black.readthedocs.io/en/stable/the_black_code_style/index.html>`_ on
your newly added and modified files prior to each PR.

Run and write tests
-------------------

All functionality in diffsims is tested via the `pytest
<https://docs.pytest.org/en/stable/>`_ framework. The tests reside in the
``diffsims.tests`` module. Tests are short functions that call functions in diffsims and
compare resulting output values with known answers. Good tests should depend on as few
other features as possible so that when they break we know exactly what caused it.

Install necessary dependencies to run the tests::

   pip install --editable .[tests]

Some useful `fixtures <https://docs.pytest.org/en/latest/explanation/fixtures.html>`_
are available in the ``conftest.py`` file.

To run the tests::

   pytest --cov --pyargs diffsims

The ``--cov`` flag makes `coverage.py <https://coverage.readthedocs.io/en/latest/>`_
print a nice report in the terminal. For an even nicer presentation, you can use
``coverage.py`` directly::

   coverage html

Then, you can open the created ``htmlcov/index.html`` in the browser and inspect the
coverage in more detail.

Useful hints on testing:

- When comparing integers, it's fine to use ``==``. When comparing floats use something
  like assert ``np.allclose(shifts, shifts_expected, atol=0.2)``.
- ``@pytest.mark.parametrize()`` is a convenient decorator to test several parameters of
  the same function without having to write to much repetitive code, which is often
  error-prone. See `pytest documentation for more details
  <https://doc.pytest.org/en/latest/how-to/parametrize.html>`_.

Build and write documentation
-----------------------------

Docstrings -- written at the start of a function -- give essential information about how
it should be used, such as which arguments can be passed to it and what the syntax
should be. The docstrings mostly follow the `numpydoc
<https://numpydoc.readthedocs.io/en/latest/format.html>`_ standard.

We use `Sphinx <https://www.sphinx-doc.org/en/master>`_ for documenting functionality.
Install necessary dependencies to build the documentation::

    pip install -e .[doc]

Then, build the documentation from the ``doc`` directory::

    cd doc
    make html

The documentation's HTML pages are built in the ``doc/build/html`` directory from files
in the `reStructuredText (reST)
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
plaintext markup language. They should be accessible in the browser by typing
``file:///your-absolute/path/to/diffsims/doc/build/html/index.html`` in the address bar.

Continuous integration (CI)
===========================

We use `GitHub Actions <https://github.com/pyxem/diffsims/actions>`_ to ensure that
diffsims can be installed on Windows, macOS and Linux. After a successful installation,
the CI server runs the tests. After the tests return no errors, code coverage is
reported to `Coveralls <https://coveralls.io/github/pyxem/diffsims?branch=master>`_.

Learn more
==========

1. The Python programming language, `for beginners <https://www.python.org/about/gettingstarted/>`__.
