# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


## Requirements
* [Python](https://www.python.org/downloads/) >= 3.7
* [virtualenv](https://virtualenv.pypa.io/en/latest/) >= 16.4.3
* [GNU Make](https://www.gnu.org/software/make/) >= 3.8.1

## Build
* Create a new sandbox (aka venv) and install required python libraries into it
    ```bash
    virtualenv --python=python3.7 venv
    source venv/bin/activate

    # Install sm-dataflow as an editable package
    pip install -e .
  
    # Install additional Python packages for development
    pip install -r requirements-dev.txt
  
    # Download the spaCy model for tokenization
    python -m spacy download en_core_web_md-2.2.0 --direct
  
    # Add tests/ to PYTHONPATH
    export PYTHONPATH="./tests:$PYTHONPATH"
    ```
* Run `make test` to execute all tests.
    
## IntelliJ/PyCharm Setup
* Setup the Python Interpreter
    - For PyCharm, go to `Settings -> Project Settings -> Project Interpreter`.
    - For IntelliJ, go to `File -> Project Structure -> Project -> Project SDK`.
    - Add a `Virtualenv Environment` from an `Existing environment` and set the Interpreter to `YOUR_REPO_ROOT/venv/bin/python`.
- Configure `pytest`
    * In `Preferences -> Tools -> Python Integration Tools`, set the default test runner to `pytest`.
- Setup source folders so that the `pytest` in the IDE becomes aware of the source codes.
    * Right click the `src` folder and choose `Mark Directory As -> Sources Root`.
    * Right click the `tests` folder and choose `Mark Directory As -> Test Sources Root`.

## Pull Requests
We force the following checks before a pull request can be merged into master:
  [isort](https://pypi.org/project/isort/),
  [black](https://black.readthedocs.io/en/stable/),
  [pylint](https://www.pylint.org/),
  [mypy](http://mypy-lang.org/),
  and [pytest](https://docs.pytest.org/en/latest/).
  
* You can run `make test` to automatically execute these checks.
* You can run `make` to execute all checks except `pytest`.
* To fix any formatting or import errors, you can simply run `make format`.
* To fix `pylint` and `mypy` errors, you can skip prefix tasks by manually running `pylint src/ tests/` and `mypy src/ tests/`.
* To fix `pytest` errors, you can skip prefix tasks by manually running `python -m pytest tests/`.
You can also go to the test file and run a specific test (see htt s://www.jetbrains.com/help/pycharm/pytest.html#run-pytest-test).
* For more details, see the `test` task in the [Makefile](./Makefile) 
and the GitHub action [lint_code_and_run_tests.yml](.github/workflows/lint_code_and_run_tests.yml).
