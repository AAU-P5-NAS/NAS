# NAS
Neural architecture search

## Installation guide
1. Install 'uv' if not done already: 
    - MacOS / Linux: 
    curl -LsSf https://astral.sh/uv/install.sh | sh

    - Windows: 
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

    Ensure 'uv' is in path (follow instructions in terminal), check with uv --version, need help then here: 
    https://github.com/astral-sh/uv 

2. Install packages: 

    simply run 'uv sync'

    or manually do:
    - uv add pydantic torch pytest
    - uv add -- dev ruff pre-commit

    then run: 
    - uv run pre-commit install

    You should also install the pylance extension from vscode for type checking in your vscode language server.

3. Run programs

    To run a program, run 'uv run python name_of_script.py'.
    To run 