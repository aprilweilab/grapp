#!/bin/bash
set -ev

flake8 grapp --count --select=E9,F63,F7,F82,F401 --show-source --statistics

black grapp/ setup.py test/ examples/ --check 

mypy grapp --no-namespace-packages --ignore-missing-imports

pytest test/

