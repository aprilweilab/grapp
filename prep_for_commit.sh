#!/bin/bash
set -ev

flake8 grgnn --count --select=E9,F63,F7,F82,F401 --show-source --statistics

black grgnn/ setup.py test/ examples/ --check 

mypy grgnn --no-namespace-packages --ignore-missing-imports

pytest test/

