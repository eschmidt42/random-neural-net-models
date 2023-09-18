# publishing

Set up an account on TestPyPi and PyPi. Then after enabling two factor authentication set up a token. Once this is done add as described on (Test)PyPi to $HOME/.pypirc.

Then run

    python -m build

    twine check dist/*

    twine upload -r testpypi dist/*

    twine upload -r pypi dist/*

    rm -r dist

## References

https://realpython.com/pypi-publish-python-package/
