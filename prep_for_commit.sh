#!/bin/bash
set -ev

flake8 grapp aireml --count --select=E9,F63,F7,F82,F401 --show-source --statistics

black grapp/ aireml/ setup.py test/ examples/ --check 

mypy grapp aireml --no-namespace-packages --ignore-missing-imports

pytest -x test/

