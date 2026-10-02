# Contributing

Install with `python -m pip install -e '.[dev]' black ruff` and run
`black --check .`, `ruff check .`, and `python -m pytest -q`.
Tests use localhost and mock transports; never add paid API calls to pytest.
Live examples require explicit opt-in and your own credentials. Do not commit
keys, recordings of other people, or account data. Contributions are provided
under the MIT license. Report bugs through GitHub issues with redacted logs.
