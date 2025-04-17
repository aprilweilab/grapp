#!/bin/bash
set -ev

flake8 grgapp --count --select=E9,F63,F7,F82,F401 --show-source --statistics

black grgapp/ setup.py test/ examples/ --check 

mypy grgapp --no-namespace-packages --ignore-missing-imports

pytest test/

