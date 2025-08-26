# Generating a new release

1. Bump version number in `grapp/version.py`
2. Push and merge to main
3. Tag main with the version `vX.Y` and push
4. Push to PyPi:
```
rm -rf dist
python setup.py sdist
python setup.py bdist_wheel
python3 -m twine upload dist/*
```
