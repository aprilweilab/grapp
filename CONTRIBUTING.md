# Contributing to grapp

We welcome pull requests (PRs) for [open issues](https://github.com/aprilweilab/grapp/issues)! If you have a request,
suggestion, or improvement, please open a [new issue](https://github.com/aprilweilab/grapp/issues/new) even if
you plan to implement the change yourself. That way you can get feedback prior to spending time in implementation.

## What should be in a PR?

A PR should be the minimal non-breaking changeset for a small feature/bugfix. Don't lump things together! It is fine
to have a branch where you have lots of changes that work together, but before submitting a PR they should be split
into reasonable sized "atomic changes".

If a PR is fixing a bug, it should include an automated regression test (unless there is a very good reason not to).

If a PR is adding a feature, there should be at least a few automated tests covering the new feature.

## Developer information

You can install an interactive version of `grapp` (meaning your code changes will be immediately
available in your environment/venv) via:
```
pip install -e .
```

We use the following tools in our CI pipeline:
* [black](https://pypi.org/project/black/) code formatter
  * You can format your changes via: `black grapp/ setup.py test/ examples/`
* [flake8](https://flake8.pycqa.org/en/latest/) code linter
* [pytest](https://pypi.org/project/pytest/) for running the tests in `test/`
  * You can run all tests via `pytest test/`
* [mypy](https://www.mypy-lang.org/) for type checking


Before creating a PR, run `./prep_for_commit.sh` on your changes, and it will simulate the CI
pipeline and tell you of any linting, formatting, or test errors.
