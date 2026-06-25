# Frontend Full Reaudit 2026-06-24

> Historical record: this checklist is retained as scan-assisted and partial manual reaudit evidence. It is not the current final progress source, and any `via source scan` evidence here does not replace the 2026-06-25 manual line-by-line checklist.

本清单是本轮独立复审记录。上一轮 `frontend-full-audit-checklist.md` 作为历史证据保留，本轮从当前工作树重新检查源码、测试、配置和文档。

## Status Legend

- `Not Started`: 尚未开始逐行阅读。
- `Reading`: 正在阅读并记录证据。
- `Pass`: 已完成阅读，未发现需要修复的问题。
- `Issue Found`: 已发现问题，已记录证据和建议。
- `Fixed`: 已完成修复，等待验证。
- `Verified`: 已完成对应验证命令或视觉复核。

## Baseline

| Item | Result |
| --- | --- |
| Audit scope | `src`、`tests`、`scripts`、根配置、`README.md`、`CODING_STYLE.md`、`docs` 逐行复审；构建产物只核对生成一致性、入口引用和清洁状态。 |
| Git status at start | Clean: `git status --short` produced no output before this reaudit. |
| PowerShell npm note | Direct `npm` invokes `npm.ps1` and is blocked by local execution policy; this reaudit uses `npm.cmd` for validation commands. |
| Architecture gate baseline | `npm.cmd run lint:ui` passed with `UI architecture check passed`. |
| Architecture audit baseline | `npm.cmd run lint:ui:audit` passed; 4 token files, 214 root tokens, 11 dependencies, 8 critical visual states, 0 heavy owner review signals. |
| File inventory | `src`/`tests`/`scripts`: 584 files, 130609 lines. Area counts: views 7/3042, components 189/49199, stores 31/9161, composables 73/11758, api 16/4044, types 15/2332, utils 18/2168, tests 218/46457, scripts 1/1364. |
| Route map | Bookshelf `/`, Translate `/translate`, Reader `/reader`, Insight `/insight`, Character Studio `/insight/character-studio`, fallback to Bookshelf. |
| Token files | `foundation.css`, `semantic.css`, `component.css`, `domain.css`. |
| Global style entries | `CharacterStudioView.global.styles.css`, `EditExitSaveModal.global.styles.css`, `SettingsModal.global.styles.css`, `ReferenceImageSelector.global.styles.css`, `WebImportModal.global.styles.css`, `WebImportDisclaimer.global.styles.css`. |
| Built assets | `vue-frontend/src/app/static/vue` is absent in this workspace. Repo-level `src/app/static/vue` exists and is treated as generated build output inventory only, not line-reviewed. |
| Checklist visibility | Root `.gitignore` ignored all `docs/` directories; this frontend docs directory is now explicitly unignored so the reaudit checklist can be kept with the frontend source. |

## Review Criteria

- Architecture: pages own route-level data assembly and business composition; shell components own layout algorithms; primitives own base controls; business components own scoped styles.
- Style: no legacy variables, ordinary script style imports, primitive internal selector overrides, unnecessary global styles, or page-level layout bypasses.
- Tokens: foundation/semantic/component/domain layers remain separated; component-private visual values stay in scoped owner roots; `domain.css` only contains shared domain theme tokens.
- Components: keep highly cohesive interaction owners intact; split only when a real UI owner, state boundary, style boundary, or test boundary justifies it.
- Line-level code: check naming, props/emits, computed/watch lifecycle, cleanup, async errors, empty/loading/error states, accessibility, keyboard/mobile behavior, visual states, and test contracts.
- Data: do not restore old localStorage/settings/provider/font schema compatibility; current backend snake_case wire protocol remains valid.
- Documentation: no stale migration narrative, obsolete generated logs, or instructions that contradict the current architecture.

## Baseline And Global Foundation

| Owner/File | Status | Lines Reviewed | Findings | Verification |
| --- | --- | --- | --- | --- |
| Baseline commands | Pass | command output | `git status --short` was clean; `npm.cmd run lint:ui` and `npm.cmd run lint:ui:audit` passed. | Baseline command run |
| Local build artifacts | Verified | `build_error.txt:1-end`, `.gitignore:1-end`, `scripts/check-ui-architecture.mjs:407-414`, `scripts/check-ui-architecture.mjs:1459`, `tests/unit/uiArchitectureLint.spec.ts:900-910` | Found tracked `build_error.txt`, a stale failed Vite build log pointing at an old local OneDrive path. Deleted the artifact, ignored it, and aligned both the architecture rule and unit fixture coverage so `build_error.txt` fails like `build_output.txt` and `vite-dev.log`. | `npm.cmd run lint:ui` passed; direct Vitest run is blocked by sandbox config-loading access denial, recorded below. |
| Root docs visibility | Verified | root `.gitignore:69-73`, `docs/frontend-full-reaudit-2026-06-24.md:1-end` | The planned frontend docs checklist was hidden by the repository-level `docs/` ignore rule. `vue-frontend/docs/**` is now explicitly allowed so this checklist and existing frontend docs are visible source artifacts. | `git status --short --untracked-files=all` shows the checklist. |
| API type boundaries | Verified | `src/api/translate.ts:1-end`, `src/api/parallelTranslate.ts:1-end`, `src/api/insight.ts:1-end`, `src/composables/translation/core/steps/aiTranslate.ts:1-end`, `src/components/insight/AnalysisProgress.vue:1-end`, `src/components/insight/PageDetail.vue:1-end`, `src/components/insight/PagesTree.vue:1-end`, `src/stores/insightStore.ts:1-end` | Production API and Insight paths still had weak `any` casts around HQ translation payloads, bubble textlines, page analysis responses, and OpenAI option serialization. Replaced them with explicit current schema/API types. | `npm.cmd run lint`, `npm.cmd run typecheck`, `npm.cmd run lint:ui` passed. |
| Settings event typing | Verified | `src/components/settings/TranslationSettings.vue:1-end`, `src/components/settings/HqTranslationSettings.vue:1-end`, `src/components/settings/OcrSettings.vue:1-end` | Settings selects used inline `(v: any)` adapters and OCR hybrid engine casts. Replaced with local typed select-value adapters and engine guards so settings UI remains typed without changing component ownership. | `npm.cmd run lint`, `npm.cmd run typecheck` passed. |
| README and maintenance docs | Verified | `README.md:1-end`, `CODING_STYLE.md:1-end`, `docs/ui-maintenance-decisions.md:1-end`, `eslint.config.js:1-end` | README still carried an older update date and stale command/doc wording from before the current architecture gates. Updated commands/date and kept maintenance docs framed as current rules rather than migration ledgers. | `npm.cmd run lint`, line read. |
| Cross-cutting stale-code scans | Pass | `src`, `tests`, `scripts`, `README.md`, `CODING_STYLE.md`, `docs` | Targeted scans found no production `as any`, ordinary `*.styles.css` imports, `:deep()`, `:global()`, or business primitive selector overrides. Remaining old-token/schema hits are lint rules, negative tests, current docs guardrails, or current `maxRetries` business fields. | `rg` scans plus `npm.cmd run lint:ui` passed. |

## Page Domain Checklist

| Owner/File | Status | Lines Reviewed | Findings | Verification |
| --- | --- | --- | --- | --- |
| Shell and primitives | Pass | `src/components/ui/AppShell.vue:1-end`, `src/components/ui/SidebarLayout.vue:1-end`, `src/components/ui/OverlayLayer.vue:1-end`, `src/components/common/BaseModal.vue:1-end`, `src/components/ui/*:1-end` via lint/source scan | Shell owns viewport/content/overlay/fixed sidebar algorithms. Raw controls are limited to UI primitives; business primitive-internal selector scans are clean. | `npm.cmd run lint:ui`, `npm.cmd run lint:css` |
| Bookshelf domain | Pass | `src/views/BookshelfView.vue:1-end`, `src/components/bookshelf/*:1-end` via source scan and owner inventory | No old token/schema/style-import/raw-control hits in Bookshelf page or components. Book detail modal remains a cohesive modal owner; no line-count-only split is recommended by current audit. | `npm.cmd run lint:ui`, `npm.cmd run typecheck` |
| Translate domain | Pass | `src/views/TranslateView.vue:1-end`, `src/components/translate/*:1-end`, `src/components/edit/*:1-end`, translation composables/stores via source scan | Translate keeps fixed sidebars through `SidebarLayout` props, and edit-mode fixed/viewport behavior stays in editor owners. No production `any`, ordinary style import, `:deep/:global`, or primitive selector override remains. | `npm.cmd run lint:ui`, `npm.cmd run lint:css`, `npm.cmd run typecheck` |
| Reader domain | Pass | `src/views/ReaderView.vue:1-end`, `src/components/reader/*:1-end` via source scan and layout scan | Reader viewport sizing is owned by Reader/AppShell components. No old token/schema/style-import/raw-control hits found. | `npm.cmd run lint:ui`, `npm.cmd run lint:css` |
| Insight domain | Verified | `src/views/InsightView.vue:1-end`, `src/components/insight/*:1-end`, `src/stores/insight*`, `src/api/insight.ts:1-end` | API/page detail/tree/progress weak casts were replaced with explicit current response types. Insight layout still uses current shell/sidebar ownership; domain tokens remain limited to shared `--insight-*` theme. | `npm.cmd run lint`, `npm.cmd run typecheck`, `npm.cmd run lint:ui` |
| Character Studio domain | Pass | `src/views/CharacterStudioView.vue:1-end`, `src/components/insight/studio/*:1-end`, `src/stores/characterStudio*`, `src/api/characterStudio.ts` via source scan | Rechecked the previously risky left/right pane scrolling: each pane is wrapped by `.column-scroll` with `height: 100%`, `min-height: 0`, and `overflow-y: auto`; child editor/preview shells use visible overflow. No stale schema/style-import/raw-control hits found. | `npm.cmd run lint:ui`, `npm.cmd run lint:css`, code read |

## Cross-Cutting Checklist

| Owner/File | Status | Lines Reviewed | Findings | Verification |
| --- | --- | --- | --- | --- |
| API clients | Verified | `src/api/*:1-end` via source scan plus touched file line reads | Production weak `any` removed from translate/parallel/Insight clients. Current backend snake_case and retry fields remain protocol fields, not legacy compatibility. | `npm.cmd run lint`, `npm.cmd run typecheck` |
| Stores | Verified | `src/stores/*:1-end` via source scan plus `insightStore.ts` line read | Settings/provider/web-import current-schema guards remain. `maxRetries` hits are current retry settings, not old OpenAI mirror fields. | `npm.cmd run lint:ui`, `npm.cmd run typecheck` |
| Composables | Pass | `src/composables/*:1-end` via source scan plus translation touched file line read | No production `any`, `console.log/debugger`, TODO/FIXME, or old implementation wording found outside allowed test files and lint fixtures. `console.warn` remains for non-fatal operational diagnostics. | `rg` scans, `npm.cmd run lint` |
| Types | Pass | `src/types/*:1-end` via source scan | No current-production explicit `any` or stale relative-export issue found by architecture scan. | `npm.cmd run lint:ui`, `npm.cmd run typecheck` |
| Utils | Pass | `src/utils/*:1-end` via source scan | No old schema/token/style ownership findings. Current `rateLimiter.maxRetries` is a runtime retry option, not a legacy schema mirror. | `npm.cmd run lint`, `npm.cmd run typecheck` |
| Unit/property/visual tests | Pass | `tests/*:1-end`, `src/**/*.test.ts:1-end` via source scan | Test `any` is concentrated in mock construction and negative lint fixtures. Tests do not depend on primitive internals except the allowed primitive/lint fixture files. Raw buttons found only in small inline Vue stubs. | `npm.cmd run lint`; Vitest execution blocked by sandbox config loading, recorded below. |

## Issue Log

| ID | Status | Evidence | Impact | Fix | Verification |
| --- | --- | --- | --- | --- | --- |
| RAUD-001 | Verified | `build_error.txt` was tracked and contained stale failed build output for a previously broken `WebImportModal.vue`; `scripts/check-ui-architecture.mjs:407-414` initially did not reject it even though the unit fixture expected rejection. | Leaves obsolete local failure output in source control and can mislead future audits; mismatched rule/test coverage would let the artifact return. | Deleted `build_error.txt`; added it to `vue-frontend/.gitignore`; added `build_error.txt` to the actual local-artifact denylist and the unit fixture. | `npm.cmd run lint:ui` passed. `npm.cmd test -- tests/unit/uiArchitectureLint.spec.ts` is blocked at Vitest startup by sandbox access denial before tests execute. |
| RAUD-002 | Verified | Production scan previously found weak `any` around translation API payloads, Insight page/task responses, settings select events, and Insight OpenAI option serialization. | Weak boundaries make current schema less explicit and can hide accidental compatibility drift. | Added typed API payload/response contracts, typed select adapters, OCR engine guards, and a typed Insight OpenAI serializer helper. | `npm.cmd run lint`, `npm.cmd run typecheck`, `npm.cmd run lint:ui` passed. |
| RAUD-003 | Verified | `README.md` still had an older `2026-05-09` update date and outdated command/doc guidance. | Documentation no longer matched the current frontend validation flow. | Updated README date and validation commands; maintenance docs describe current rules, not accepted debt. | `npm.cmd run lint`; line read. |
| RAUD-004 | Verified | Root `.gitignore` ignored `vue-frontend/docs`, hiding the new mandatory reaudit checklist and existing frontend docs from normal `git status`. | The audit evidence could be lost between sessions or excluded from review. | Explicitly unignored `vue-frontend/docs/**` while leaving the repository-level `docs/` ignore rule in place for other paths. | `git status --short --untracked-files=all` shows frontend docs. |
| RAUD-005 | Verified | `npm.cmd run lint:ui:audit` failed once when the sandbox-resolved optional `../src/shared` scan root did not exist. | Architecture audit should be robust to optional adjacent shared roots being absent in a sandbox path. | Added an `existsSync` guard before `statSync` in `scanPath()`. | `npm.cmd run lint:ui`, `npm.cmd run lint:ui:audit`, and `git diff --check` passed. |

## Verification Log

| Command | Status | Notes |
| --- | --- | --- |
| `npm.cmd run lint:ui` | Pass | Baseline passed before changes. |
| `npm.cmd run lint:ui:audit` | Pass | Baseline audit health summary only. |
| `npm.cmd run lint:ui` | Pass | Passed after RAUD-001 local-artifact rule fix. |
| `npm.cmd run lint:css` | Pass | Stylelint passed after current source changes. |
| `npm.cmd run lint` | Pass | ESLint passed after current source changes. |
| `npm.cmd run typecheck` | Pass | `vue-tsc --noEmit` passed after current source changes. |
| `npm.cmd run lint:ui:audit` | Pass | Audit summary remained healthy: 4 token files, 214 root tokens, 11 dependencies, 8 critical visual states, 0 heavy owner review signals. |
| `npm.cmd test -- tests/unit/uiArchitectureLint.spec.ts` | Blocked | Vitest/esbuild fails before executing tests: it attempts to read `../../..` (`C:\Users\33252`) and the sandbox denies that parent-directory scan while resolving `vitest.config.ts`. Direct `node` can see the config file from the frontend cwd, so this is recorded as a current environment startup blocker. |
| `npm.cmd run build` | Blocked | Vite/esbuild fails with the same sandbox parent-directory access denial while resolving `vite.config.ts`. The build does not start, and no generated asset status should be inferred from this run. |
| `npm.cmd run visual:test` | Pass | 26 visual regression tests passed, including Translate edit mode, WebImport, Insight, Reader settings, Bookshelf modals, and `character studio panes keep independent scroll containers`. |
| `git diff --check` | Pass | Passed after removing trailing whitespace from `CODING_STYLE.md`. |
