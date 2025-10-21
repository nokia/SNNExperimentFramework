Contributing
============

We welcome contributions to the SNN2 framework! This guide outlines how to contribute effectively to the project.

Getting Started
---------------

Development Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Fork and Clone**

   .. code-block:: bash

      # Fork the repository on GitHub, then:
      git clone https://github.com/your-username/SNN2.git
      cd SNN2

2. **Set Up Development Environment**

   .. code-block:: bash

      # Install the required packages if not done previously
      make isntall
      # Create virtual environment
      make virtualenv


Code Style and Standards
------------------------

The SNN2 project follows coding standards to maintain code quality and consistency. Or at least we try.
We adhere to established best practices and guidelines.

Python Code Style
~~~~~~~~~~~~~~~~~~

* **PEP 8**: Follow Python Enhancement Proposal 8 for code style
* **Line Length**: Maximum 100 characters per line
* **Indentation**: 4 spaces (no tabs)
* **Naming Conventions**:
  * Variables and functions: ``snake_case``
  * Classes: ``PascalCase``
  * Constants: ``UPPER_CASE``
  * Private methods: ``_leading_underscore``

Code Structure Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~

We follow principles inspired by the **Linux Kernel Coding Style** for code organization and structure:

* **Function Length**: Keep functions concise and focused on a single task
* **Code Comments**: Write clear, meaningful comments explaining *why*, not *what*
* **Documentation**: Every public function and class must have docstrings
* **Error Handling**: Use explicit error handling, avoid bare ``except`` clauses
* **Import Organization**: Group imports logically (standard library, third-party, local)

For detailed guidelines on code structure, formatting, and best practices, refer to the
`Linux Kernel Coding Style Guide <https://www.kernel.org/doc/html/latest/process/coding-style.html>`_,
which provides excellent principles for writing maintainable code.

Development Workflow
--------------------

Commit Messages
~~~~~~~~~~~~~~~

Follow conventional commit format:

.. code-block:: text

   type(scope): short description

   Longer description if needed.

   - Bullet points for details
   - Reference issues: Fixes #123

**Types**: ``feat``, ``fix``, ``docs``, ``style``, ``refactor``, ``test``, ``chore``

**Examples**:

.. code-block:: text

   feat(model): add support for custom embedding layers

   fix(preprocessing): handle edge case in data normalization

   docs(api): update configuration documentation

Testing
-------

A specific .coveragerc is porvided to help the usage of pytest.
At the moment the tests are not chomprehensive of every part of the code.

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   python -m pytest

   # Run specific test file
   python -m pytest tests/test_model.py

   # Run with coverage
   python -m pytest --cov=SNN2

Writing Tests
~~~~~~~~~~~~~

* Place tests in the ``tests/`` directory
* Use descriptive test names: ``test_model_handles_empty_input``
* Follow the Arrange-Act-Assert pattern
* Mock external dependencies
* Test both success and failure cases

Code Quality Tools
------------------

Linting and Formatting
~~~~~~~~~~~~~~~~~~~~~~

We use pylint to maintain code quality with a best effort approach.

.. code-block:: bash

   # find all py files and run pylint on those
   find SNN2/ -type f -name "*.py" | xargs pylint > pylint_output.txt

CI-CD
~~~~~~~~~~~~~~~

Pylint theoretically is also run thanks to CI-CD (this is work in progress)

Documentation
-------------

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~~

* **Docstrings**: Use numpy style for documentation, a reference is provided `here <https://numpydoc.readthedocs.io/en/latest/format.html>`_
* **Type Hints**: Include type hints for function parameters and return values
* **Examples**: Provide usage examples in docstrings
* **RST Format**: Documentation files use reStructuredText

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   make docs

Types of Contributions
----------------------

Bug Reports
~~~~~~~~~~~

When reporting bugs, include:

* **Environment**: OS, Python version, package versions
* **Steps to Reproduce**: Minimal example that triggers the bug
* **Expected vs Actual Behavior**: Clear description of the issue
* **Logs**: Relevant error messages and stack traces

Feature Requests
~~~~~~~~~~~~~~~~

For new features, provide:

* **Use Case**: Explain the problem this feature solves
* **Proposed Solution**: Describe the desired functionality
* **Alternatives**: Consider alternative approaches
* **Implementation Ideas**: Technical details if you have them

Code Contributions
~~~~~~~~~~~~~~~~~~

Before submitting code:

1. **Discuss First**: Open an issue to discuss major changes
2. **Write Tests**: Ensure new code is well-tested
3. **Update Documentation**: Keep docs synchronized with code changes
4. **Follow Standards**: Adhere to coding style and conventions

Pull Request Process
--------------------

Submission Checklist
~~~~~~~~~~~~~~~~~~~~

Before submitting a pull request:

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] Changes are focused and atomic

Review Process
~~~~~~~~~~~~~~

1. **Automated Checks**: CI/CD runs tests and quality checks
2. **Code Review**: A maintainer reviews the code for quality and design
3. **Discussion**: Address feedback and make requested changes
4. **Approval**: A maintainer approves the changes
5. **Merge**: Code is merged into the target branch

Communication
-------------

Getting Help
~~~~~~~~~~~~

* **GitHub Issues**: For bug reports and feature requests
* **GitHub Discussions**: For questions and general discussion

Thank you for contributing to SNN2! Your efforts help make this framework better for everyone.