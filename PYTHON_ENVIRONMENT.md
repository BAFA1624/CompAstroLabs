# Python virtual environment

Run these commands from the repository root.

## Generate and install

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Activate and deactivate

Activate the environment for a new shell with:

```sh
source .venv/bin/activate
```

Leave it with:

```sh
deactivate
```

## Rebuild

Deactivate the environment if necessary, then recreate it from the tracked requirements:

```sh
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The `.venv` directory is machine-specific and ignored by Git. Track changes to
`requirements.txt` instead.
