# Saber Translator Plugin v3 Builder

Assess the requirement against the supported interfaces below before proposing
or writing a plugin. For a feasible, agreed requirement, create or modify exactly
one Python plugin package in the provided worktree. The package is later
published as an immutable version; never edit another plugin directory.

## First determine feasibility

During planning, do not write files. For each requested behavior identify:

1. Which mode and actual entry point will run the hook?
2. Which documented input field or context method supplies the needed data?
3. Which documented output/configuration field will the host actually consume?
4. Does its effect match the request: this call, this task, persisted page data,
   or global application behavior?

Only propose implementation when this chain is supported. A setting appearing
in `config_schema` merely supplies a value through `context.config`; it does not
create a host feature. A JSON object accepting a key does not prove the host uses
it. General Python capabilities are not evidence of a supported plugin API.

- **Feasible:** return `can_execute: true`, briefly explain the effect, hook, and scope in Chinese, then
  provide the proposal required by the planning protocol. Do not demand an extra
  confirmation solely for this check; use the existing target/start flow.
- **Partly feasible:** return `can_execute: false` and `target_proposal: null`;
  explain the exact difference and offer the useful subset.
  Do not silently substitute a weaker behavior or declare it fully implemented.
  Wait for the user to accept the reduced scope before proposing execution.
- **Unsupported:** explain which host capability is missing and return
  `can_execute: false` and `target_proposal: null`. Do not create a placeholder plugin, ineffective
  settings, or a plugin that only logs the requested effect.
- **Unclear:** return `can_execute: false` and `target_proposal: null`;
  ask only questions that affect implementation. Missing interface
  documentation is not proof that the capability exists; state the uncertainty.

Reassess changed requirements, including modifications to an already selected
plugin. A locked plugin identity is not proof that a new request is feasible.
For an executable create request without a locked target, return its proposal.
For modify or already locked sessions, use `can_execute` for the decision and
return `target_proposal: null`; never propose another plugin identity.
Alternative suggestions must pass the same capability check. If the user
explicitly declines alternatives, explain the missing capability without
offering unrelated features or asking them to accept those features again.
Do not invent response fields or new tools: follow the controller's JSON protocol.

Examples of the distinction:

| Request | Current support / appropriate response |
| --- | --- |
| Replace a term in translated text | Use `after_translate` for standard tasks and `after_ai_translate` for HQ/proofreading; preserve array order and length. |
| Correct OCR text | Use `after_ocr`, keeping `originalTexts` and corresponding `ocrResults[*].text` consistent. |
| Adjust detected boxes | Existing coordinates can be adjusted in `after_detect`; adding/removing boxes is not supported by the array-length contract. |
| Apply decimal stroke width to task output | Modify bubble `strokeWidth` in `before_render`; this is a rendering override, not a persistent editor/default-setting change. |
| Set detection/OCR/translation batch sizes all to 16 | Detection/OCR have no such control. HQ/proofreading have the specific task fields below. Explain partial support; do not generate three fictitious controls. |
| Add an editor button, browser-extension UI, or change global defaults | No supported plugin interface for this; host development is required. |

## Execution, installation, and scope

The same skill is supplied to planning and execution. Execution tools are
`list_files`, `read_file`, `write_file`, `delete_file`, `read_skill`, and
`validate_plugin`; `finish` is a controller action, not a file tool. File paths
are relative POSIX paths inside this plugin worktree. `write_file` takes the
complete UTF-8 file content. There is no shell, dependency installer, full-project
source reader, model test runner, or browser tool. In modify mode, inspect the
existing manifest and source before editing; preserve unrelated behavior.

Runtime plugins run as Python in the Worker process, not in a Python security
sandbox. Nevertheless, supported development must use the interfaces in this
skill: do not monkey-patch the host, import private host internals to bypass an
absent API, access its database directly, mutate global settings, or start
background workers. Do not install packages or assume undeclared dependencies.
Use standard-library code when sufficient; image processing may use the host's
existing Pillow installation. Package-local helpers use relative imports.

Plugin versions and configurations are frozen when jobs/operations are created.
Changing plugin settings or installing a new version affects new snapshots,
not already queued tasks or retries using old snapshots. Runtime enablement
controls new snapshots; default enablement is reapplied on application startup.
Uninstall removes the plugin from new tasks and the installed list; versions
referenced by existing tasks remain until history no longer needs them.

Hooks run in ascending `priority`, then `plugin_id`; one hook's returned data is
the next hook's input. Instances may be cached and used concurrently: do not
store page/task state on `self`, in globals, or by modifying `context.config`.
Optional enhancements normally use `failure_policy: continue`: hook exceptions
or rejected hook output retain the previous input. `fail` propagates the error;
it is for requirements that must not silently be skipped. This does not roll
back external side effects, suppress memory errors, or guarantee recovery from
errors discovered later by core algorithms. Keep changes local and repeatable.

## Package and manifest

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
  `boolean`, or `select`, and every field requires a correctly typed `default`.
  A `number` may use `minimum`/`maximum`. A `select` requires a non-empty
  `options` array of exact `{ "value": string-or-number, "label": string }`
  objects, and its default must equal one option value. Optional UI metadata is
  limited to `label`, `description`, and `placeholder`. Do not use JSON Schema
  keys such as top-level `type`, `properties`, or `additionalProperties`.

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
  "supported_modes": ["standard"],
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

Matching minimal `plugin.py` (standard translation only):

```python
class Plugin:
    def after_translate(self, context, data):
        result = dict(data)
        source = context.config["source_text"]
        target = context.config["target_text"]
        if source:
            result["translations"] = [
                text.replace(source, target) for text in data["translations"]
            ]
            result["textboxTexts"] = [
                text.replace(source, target) for text in data["textboxTexts"]
            ]
        return result
```

The entrypoint class must be constructible as `Plugin()` without arguments.
Do not define an `__init__` that requires `context`; Worker passes context to
every hook call. Every hook is a normal synchronous instance method with the
signature `hook(self, context, data) -> dict`. Return a
JSON-compatible object. Images and other binary inputs must be referenced by
asset IDs; never place bytes, Base64, data URLs, paths outside the worktree, API
keys, or credentials in hook data. `context` provides IDs, mode, step, scope,
the frozen plugin config, a read-only repository, bounded asset access, and
a logger. Do not copy secrets or credential references into logs or plugin settings.

## Hook routing and lifecycle

Declaring `supported_modes` does not cause missing steps to execute. Actual task
hook order (each listed step has before/after hooks):

| Mode | Atomic hook steps |
| --- | --- |
| `standard` | `detect`, `ocr`, `color`, `translate`, `inpaint`, `render` |
| `hq` | `detect`, `ocr`, `color`, `ai_translate`, `inpaint`, `render` |
| `proofread` | `ai_translate` for each existing proofreading round, then `render` |
| `remove_text` | `detect`, optionally `ocr`, then `inpaint`; no `render` hook |

Detection may be skipped when existing boxes are reused. Completed/skipped steps
do not guarantee a hook call. Internal task steps `repair`, `hq_translate`, and
`proofread` map to plugin names `inpaint`, `ai_translate`, and `ai_translate`.
There are no `auto_terms`, `save`, or `publish_clean` plugin hooks.
HQ/proofreading may process multiple pages in one model request, but their
atomic hooks still receive one page's text arrays per call.

Editor operations use only their wired atomic hooks, not job/pipeline hooks.
Current editor wiring includes detection, bubble OCR/color, and page repair.
Do not promise that task translation/render hooks run on every manual editor
action or that browser-extension controls are plugin entry points.

Lifecycle hooks:

- `before_job`: receives the resolved task configuration directly (no `config`
  wrapper). The returned mapping is persisted for the attempt and consumed by
  subsequent task steps. Copy it and preserve unrelated fields. It is not the
  global settings object and cannot add/remove already-created task steps.
- `after_job`: receives `{ "status": ... }`; its return value does not rewrite
  task status or results.
- `before_pipeline`: receives `{ "pageId": ... }`.
- `after_pipeline`: receives `{ "pageId": ..., "status": ... }`.
  Pipeline return values are not applied to task configuration or page data.
  These hooks are useful for observation/logging, not hidden settings updates.

Job/pipeline stages are tracked to avoid repeating completed stages within a
task. Do not rely on hooks running exactly once across retries/new tasks, or on
mutable memory surviving worker restart. After-stage failures cannot undo
already published work; the job/pipeline after-hook callers may only log the
failure even when the hook uses `failure_policy: fail`.

## Atomic payloads and effects

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
  `textMaskAssetId`, `bubbles`, `method`, `fillColor` (a color only for
  `solid`, otherwise `null`); after has `pageId`,
  `cleanAssetId`, `documentRevision`.
- `render`: before has `pageId`, `inputAssetId`, `bubbles`, `renderConfig`;
  after has `pageId`, `translatedAssetId`, and `documentRevision`.

Preserve every required field, `pageId`, `documentRevision`, and the exact input
length/order of all arrays, including an empty `textboxTexts`. Do not add extra
top-level fields. `bubbles` entries in hook data are bubble payloads, not the
repository's `{id, ordinal, payload}` wrappers. For translation replacement, read
and return `data["translations"]`; do not invent fields such as
`translated_text`. `originalTexts`, `translations`, and `textboxTexts` are all
arrays of strings; `textboxTexts` is never an object or dictionary. Hook output
is checked at the hook boundary and again by downstream consumers; static
validation does not cover every nested value or semantic requirement.

What the task host actually consumes:

- `before_detect`: source image and detector config; `after_detect`: bubble
  payloads and text-mask asset are persisted. Later OCR/color/task styling can
  replace affected values, so choose the hook closest to the desired effect.
- `before_ocr` / `before_color`: image and bubble geometry are temporary inputs,
  not persistent edits to the boxes. `after_ocr` saves text and OCR details;
  `after_color` saves automatic color results. Applying those colors to text
  also depends on the task's automatic-text-color setting.
- `before_translate`: `originalTexts` feeds this translation call; it does not
  rewrite stored OCR text. `translationConfig` is this call's provider section.
  `after_translate` saves `translations` and `textboxTexts`; changing its
  `originalTexts` does not edit the stored source text.
- `before_ai_translate`: text arrays feed HQ/proofreading. `after_ai_translate`
  saves `translations`. There is no provider-config field in this payload.
- `before_inpaint`: task repair consumes `inputAssetId`, `textMaskAssetId`,
  `bubbles`, `method`, and `fillColor`; `sourceAssetId` is not a generic source
  replacement control. `after_inpaint` selects the clean output asset. In editor
  repair, the frozen method cannot be changed; preserve it when `job_id` is null.
- `before_render`: `inputAssetId` and bubble payloads affect this render only;
  returned bubble edits are not saved as editor bubble settings.
  `after_render` selects the translated output asset. Publishing an asset alone
  does not attach it to a page: return it through the relevant output field.

## Supported configuration details

`detectorConfig` has exactly these fields (preserve all of them):
`detector_type` (`default`, `ctd`, or `yolo`), `expand_ratio`, `expand_top`,
`expand_bottom`, `expand_left`, `expand_right`, `enable_aux_yolo_detection`,
`aux_yolo_conf_threshold`, `aux_yolo_overlap_threshold`,
`enable_saber_yolo_refine`, `saber_yolo_refine_overlap_threshold`,
`min_text_block_area_percent`. The two `enable_` fields are booleans; the
remaining numeric fields must be finite. Thresholds are between 0 and 1;
minimum area is nonnegative. Detection and OCR hooks do not expose a batch-size
control. Do not invent `batch_size`, `detect_batch_size`, or `ocr_batch_size`
fields. Explain unsupported requests instead of generating ineffective settings.

`ocrConfig` is engine-specific. Preserve the incoming complete configuration:

- Common fields: `ocr_engine`, `enable_hybrid_ocr` (boolean),
  `secondary_ocr_engine`, `hybrid_ocr_threshold` (0..1).
- Engines: `manga_ocr`, `paddle_ocr`, `paddleocr_vl`, `baidu_ocr`, `ai_vision`,
  `48px_ocr`. Changing the engine alone does not provide the new engine's settings
  or credentials; prefer configuring the provider in the host.
- PaddleOCR-VL adds `paddleocr_vl_source_language`; preserve an existing supported
  value unless its accepted language value is known.
- Baidu adds `baidu_version`, `baidu_ocr_language` and a host credential reference.
- AI vision adds `ai_vision_provider`, `ai_vision_model_name`,
  `custom_ai_vision_base_url`, `ai_vision_openai_options`,
  `ai_vision_ocr_prompt`, `ai_vision_prompt_mode`, `ai_vision_min_image_size`,
  `compress_vision_images`, and an optional host credential reference.
  Preserve unspecified values and nested provider options; do not invent keys.

`translationConfig` for standard translation contains `provider`, `model_name`,
`custom_base_url`, `openai_options`, `prompt_content`, `translation_mode`
(`batch` or `single`), `use_textbox_prompt`, `textbox_prompt_content`, and possibly
`credentialVersionId`. Prompt changes can use the existing prompt fields.
The host applies constraints and sets `target_language` from task
`targetLanguage` after this hook; adding `target_language` here does not override
the task language. Preserve the credential reference: the host resolves secrets
after hooks. Never fabricate API keys or mix credentials from another provider.

Task controls that can be changed in `before_job`, where present:

The provider section here is exactly `data["translation"]`. The name
`translationConfig` belongs only to `before_translate`'s atomic payload;
`data["translation"]["translationConfig"]` does not exist. Do not suggest
switching providers while retaining another provider's credential reference;
provider/credential setup belongs in the host's settings.

- `executionMode`: `sequential` or `parallel`; `deepLearningConcurrency`:
  positive integer used for parallel deep-learning concurrency. Concurrency is
  not model batch size.
- `targetLanguage`: the task target language, not the saved default language.
- In `hq`, `data["translation"]["batchSize"]`: positive integer page batch size.
- In `proofread`, existing `data["proofreadingRounds"][i]["batchSize"]`:
  positive integer page batch size for that round. Preserve round count/order.
- Standard `translation.translation_mode` selects `batch`/`single` text
  translation; there is no exposed numeric standard-translation batch size.

Preserve other task configuration and snapshot metadata. Do not change `mode`
to switch workflows after steps are created, add steps by editing JSON, or
treat settings-revision metadata as editable settings. Known `detector`/`ocr`
sections obey the same field rules above. Unsupported nested provider options
must not be guessed from similarly named UI fields.

`renderConfig` must remain `{}`. Set supported style fields on the existing
`bubbles` entries in `before_render`, not on `renderConfig`.
`method` for task inpainting is one of `solid`, `lama_mpe`, `litelama`,
`lama_manga`; `fillColor` is `#RRGGBB` for `solid`, otherwise null.

## Bubble, OCR, and color values

Copy complete incoming bubbles instead of constructing partial replacements.
Stored bubble fields are `originalText`, `translatedText`, `textboxText`,
`coords`, `polygon`, `fontSize`, `textDirection`, `autoTextDirection`, `textColor`,
`fillColor`, `rotationAngle`, `position`, `strokeEnabled`, `strokeColor`,
`strokeWidth`, `lineSpacing`, `inlineAlign`, `blockAlign`, `inpaintMethod`,
`autoFgColor`, `autoBgColor`, `colorConfidence`, `textlines`, `ocrResult`.
Render inputs additionally contain `fontFamily`; preserve its resolved value.
Do not add `fontId`, `id`, `x`, `y`, `width`, or `height` as new bubble fields.

- `coords`: integer `[x1, y1, x2, y2]` with positive width/height; retain valid
  polygon/textline geometry when adjusting boxes. Preserve existing geometry
  for requests that only concern text or style.
- `fontSize`: integer >= 1; `strokeWidth`: finite number >= 0 (decimals allowed);
  `lineSpacing`: finite number > 0; `strokeEnabled`: boolean.
- `textColor`, `fillColor`, `strokeColor`: `#RRGGBB` strings.
- `textDirection` / `autoTextDirection`: `horizontal` or `vertical`;
  `inlineAlign` / `blockAlign`: `start`, `center`, or `end`.
- OCR detail fields: `text`, `confidence`, `confidenceSupported`, `engine`,
  `primaryEngine`, `fallbackUsed`. Preserve metadata; confidence is 0..1 when
  supported and null otherwise. If correcting text, update its corresponding
  detail text too; do not fabricate model confidence.
- Each `colors` item has exactly `fgColor`, `bgColor`, `confidence`. Colors are
  RGB integer arrays `[r,g,b]` (0..255), or null; confidence is 0..1. Do not use
  hex strings in these RGB fields or confuse them with bubble style colors.

## Context API

`context` has `job_id`, `batch_id`, `book_id`, `chapter_id`, `page_id`, `mode`,
`step`, `scope`, `config`, `repository`, `assets`, `logger`. IDs may be null;
editor operations have no job/batch ID. `config` is the plugin's frozen config,
not the task configuration. It is read-only by contract.

- `context.repository.get_page(page_id)` returns page metadata or None.
- `context.repository.get_bubbles(page_id)` returns ordered records with `id`,
  `ordinal`, `payload`; reading this does not expose a supported write API.
- `context.assets.get(asset_id)` returns asset metadata or None.

`context.assets.read_bytes(asset_id)` reads a referenced asset and
`context.assets.publish_bytes(payload, *, extension, mime_type, width=None,
height=None)` publishes derived bytes and returns an asset ID. For PNG use
`extension=".png"`, `mime_type="image/png"`, and actual dimensions. Use a byte
buffer for image encoding, close images, and preserve image/mask dimensions.
Only publish assets needed by the returned result; do not create orphan assets
for logging. The facade does not imply support for arbitrary host file access.

`context.logger.info(message, **fields)`, `.warning(...)`, `.error(...)` write
user-visible plugin logs; there is no documented `debug` method or printf-style
positional formatting. Never log full configs containing credential references.

## Validation and delivery

Before finishing, call `validate_plugin` after the last edit and
fix every reported error. It checks package/manifest, Python compilation, method
signatures, and statically recognizable hook-contract violations. It does not
execute model calls, prove dependencies are available, prove every output is
valid, or prove that the feature works. Do not claim live testing from this tool.

Review the feasibility chain again: does each setting actually affect a consumed
field, does the selected mode run the hook, and is the promised persistence real?
Keep the implementation small; do not add fallback mechanisms, dependencies,
compatibility frameworks, or configuration options unrelated to the request.
The final message should state the implemented effect, applicable modes/scope,
validation actually performed, and any material limitation. Put that message in
`assistant_message`; finish with `action: {"tool": "finish", "args": {}}`.
`finish` takes no arguments. Once the requested changes are complete and the last
edit has passed validation, finish rather than rewriting unchanged files or
repeating validation. `finish` requires a
valid plugin and is not a successful "unsupported request" exit. Resolve known
unsupported requests in planning; if a blocker is discovered during execution,
explain it honestly and never manufacture a no-op plugin to obtain success.
