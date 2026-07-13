# Notebooks

Jupyter notebooks for interactive exploration of `glyphik`. For scripted,
reproducible examples, see [`examples/`](../examples).

## Setup

```shell
uv sync --group jupyter
uv run jupyter lab notebooks/
```

Outputs are stripped automatically on commit via `nbstripout` (configured in
`.pre-commit-config.yaml`), so keep notebooks committed with a clean,
re-run-from-top state.
