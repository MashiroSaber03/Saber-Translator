# Saber Translator Plugin v3 Builder

Create or modify exactly one Python plugin package in the provided worktree.
The package is later published as an immutable version; never refer to or edit
another plugin directory.

Required root files:

- `plugin.json`: static UTF-8 JSON manifest with `schema_version: 3`.
- The Python file and class named by `entrypoint`, for example
  `plugin.py:Plugin`.

The manifest declares `plugin_id`, `display_name`, `package_version`,
`entrypoint`, `hooks`, `supported_steps`, `supported_modes`, `priority`,
`failure_policy`, `default_enabled`, optional author/description, and a
`config_schema`. Supported hooks are before/after variants of `job`,
`pipeline`, `detect`, `ocr`, `color`, `translate`, `ai_translate`, `inpaint`,
and `render`.

Every hook has the signature `hook(self, context, data) -> dict`. Return a
JSON-compatible object. Images and other binary inputs must be referenced by
asset IDs; never place bytes, Base64, data URLs, paths outside the worktree, API
keys, or credentials in hook data. `context` provides IDs, mode, step, scope,
the frozen plugin config, a read-only repository, read-only asset access, and
a logger.

Use `failure_policy: continue` for optional enhancements and `fail` only when
the task must stop on plugin failure. Keep code deterministic and avoid global
mutable state. Before finishing, call `validate_plugin` and fix every reported
error.
