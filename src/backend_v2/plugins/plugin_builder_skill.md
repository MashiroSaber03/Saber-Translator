# Saber Translator Plugin v3 Builder

Create or modify exactly one Python plugin package in the provided worktree.
The package is later published as an immutable version; never refer to or edit
another plugin directory.

Required root files:

- `plugin.json`: static UTF-8 JSON manifest with `schema_version: 3`.
- The Python file and class named by `entrypoint`, for example
  `plugin.py:Plugin`.

The manifest declares every one of these required fields: `plugin_id`,
`display_name`, `package_version`, `entrypoint`, `hooks`, `supported_steps`,
`supported_modes`, `priority`, `failure_policy`, `author`, `description`,
`default_enabled`, and `config_schema`. Supported hooks are before/after variants of `job`,
`pipeline`, `detect`, `ocr`, `color`, `translate`, `ai_translate`, `inpaint`,
and `render`.

Manifest contract details:

- `hooks` is an array of hook names, for example `["after_translate"]`.
- `supported_steps` is an array using only `job`, `pipeline`, `detect`, `ocr`,
  `color`, `translate`, `ai_translate`, `inpaint`, or `render`.
- `supported_modes` is a non-empty array using only `standard`, `hq`,
  `proofread`, or `remove_text`. Never use `*`.
- `config_schema` is Saber UI field metadata, not JSON Schema. Each top-level
  key is one configurable field. Its `type` must be `text`, `number`,
  `boolean`, or `select`; a `select` field also requires a non-empty `options`
  array. Do not use JSON Schema keys such as top-level `type`, `properties`,
  or `additionalProperties`.

Minimal valid manifest:

```json
{
  "schema_version": 3,
  "plugin_id": "teacher_replace",
  "display_name": "教师术语替换",
  "package_version": "1.0.0",
  "entrypoint": "plugin.py:Plugin",
  "hooks": ["after_translate"],
  "supported_steps": ["translate"],
  "supported_modes": ["standard", "hq", "proofread"],
  "priority": 100,
  "failure_policy": "continue",
  "default_enabled": false,
  "author": "Plugin Agent",
  "description": "Replace a configured term after translation.",
  "config_schema": {
    "source_text": {
      "type": "text",
      "label": "待替换文本",
      "default": "老师"
    },
    "target_text": {
      "type": "text",
      "label": "替换为",
      "default": "导师"
    }
  }
}
```

The entrypoint class must be constructible as `Plugin()` without arguments.
Do not define an `__init__` that requires `context`; Worker passes context to
every hook call. Every hook is a normal synchronous instance method with the
signature `hook(self, context, data) -> dict`. Return a
JSON-compatible object. Images and other binary inputs must be referenced by
asset IDs; never place bytes, Base64, data URLs, paths outside the worktree, API
keys, or credentials in hook data. `context` provides IDs, mode, step, scope,
the frozen plugin config, a read-only repository, bounded asset access, and
a logger.

Atomic hook data uses these exact fields:

- `detect`: before has `pageId`, `sourceAssetId`, `detectorConfig`; after has
  `pageId`, `bubbles`, `textMaskAssetId`.
- `ocr`: before has `pageId`, `sourceAssetId`, `bubbles`, `ocrConfig`; after
  has `pageId`, `originalTexts`, `ocrResults`.
- `color`: before has `pageId`, `sourceAssetId`, `bubbles`; after has
  `pageId`, `colors`. Each color has `fgColor`, `bgColor`, and `confidence`.
- `translate`: before has `pageId`, `originalTexts`, `translationConfig`;
  after has `pageId`, `originalTexts`, `translations`, `textboxTexts`.
- `ai_translate`: before has `pageId`, `originalTexts`, `translations`; after
  has `pageId`, `originalTexts`, `translations`.
- `inpaint`: before has `pageId`, `sourceAssetId`, `inputAssetId`,
  `textMaskAssetId`, `bubbles`, `method`, `fillColor`; after has `pageId`,
  `cleanAssetId`, `documentRevision`.
- `render`: before has `pageId`, `inputAssetId`, `bubbles`, `renderConfig`;
  after has `pageId`, `translatedAssetId`, `thumbnailAssetId`, and
  `documentRevision`.

Preserve required fields and array lengths. For translation replacement, read
and return `data["translations"]`; do not invent fields such as
`translated_text`. `originalTexts`, `translations`, and `textboxTexts` are all
arrays of strings; `textboxTexts` is never an object or dictionary. Hook output
is validated before it can be persisted.
`context.assets.read_bytes(asset_id)` reads a referenced asset and
`context.assets.publish_bytes(...)` publishes a derived asset and returns its
asset ID.

Use `failure_policy: continue` for optional enhancements and `fail` only when
the task must stop on plugin failure. Keep code deterministic and avoid global
mutable state. Before finishing, call `validate_plugin` and fix every reported
error.
