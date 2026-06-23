import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'

const ROOTS = [
  '.stylelintignore',
  '.stylelintrc.json',
  'eslint.config.js',
  'package.json',
  'playwright.config.ts',
  'postcss.config.mjs',
  'src/App.vue',
  'src/main.ts',
  'src/api',
  'src/components',
  'src/config',
  'src/composables',
  'src/router',
  'src/stores',
  'src/styles',
  'src/types',
  'src/utils',
  'src/views',
  'tests',
  'tsconfig.app.json',
  'tsconfig.json',
  'tsconfig.node.json',
  'vite.config.ts',
  'vitest.config.ts',
  '../src/shared',
]
const IS_AUDIT = process.argv.includes('--audit')
const SKIP_SOURCE_SCAN = process.argv.includes('--skip-source-scan')
const TOKENS_FIXTURE_INDEX = process.argv.indexOf('--tokens-fixture')
const TOKENS_FIXTURE_PATH_INDEX = process.argv.indexOf('--tokens-fixture-path')
const TOKENS_FIXTURE_PATH = TOKENS_FIXTURE_PATH_INDEX >= 0
  ? (process.argv[TOKENS_FIXTURE_PATH_INDEX + 1] || '').replaceAll('\\', '/')
  : null
const SOURCE_FIXTURE_INDEX = process.argv.indexOf('--source-fixture')
const SOURCE_FIXTURE = SOURCE_FIXTURE_INDEX >= 0
  ? resolve(process.cwd(), process.argv[SOURCE_FIXTURE_INDEX + 1] || '')
  : null
const SOURCE_FIXTURE_PATH_INDEX = process.argv.indexOf('--source-fixture-path')
const SOURCE_FIXTURE_PATH = SOURCE_FIXTURE_PATH_INDEX >= 0
  ? (process.argv[SOURCE_FIXTURE_PATH_INDEX + 1] || '').replaceAll('\\', '/')
  : null
const TOKEN_FILE_ORDER = [
  'src/styles/tokens/foundation.css',
  'src/styles/tokens/semantic.css',
  'src/styles/tokens/component.css',
  'src/styles/tokens/domain.css',
]
const TOKEN_FILES = TOKENS_FIXTURE_INDEX >= 0
  ? [resolve(process.cwd(), process.argv[TOKENS_FIXTURE_INDEX + 1] || '')]
  : TOKEN_FILE_ORDER.map(file => join(process.cwd(), file))
const SFC_MAX_LINES = 900
const SFC_SHELL_MAX_LINES = 1200
const SFC_MAX_STYLE_LINES = 1000
const CSS_MAX_LINES = 450
const CSS_OWNER_REVIEW_LINES = 450
const DOMAIN_TOKEN_MAX_DEFINITIONS = 50
const GLOBAL_STYLE_MAX_LINES = 20
const VUE_STYLE_RE = /<style(?![^>]*\bscoped\b)[^>]*>/g
const CSS_ID_SELECTOR_RE = /^\s*#[A-Za-z0-9_-]+/m
const IMPORTANT_RE = /!important\b/
const LEGACY_IMPORT_RE = /legacy-components(?:\.css)?/
const PART_STYLE_IMPORT_RE = /@import\s+['"][^'"]*\.part\d+\.css['"]/
const SCOPED_SHARED_STYLE_RE = /<style\b[^>]*\bsrc=["'][^"']*settings-shared\.styles\.css["'][^>]*\bscoped\b[^>]*>/
const LEGACY_UI_COLOR_TOKEN_RE = /--ui-color-/
const COMPAT_COLOR_TOKEN_RE = new RegExp(`--${'color'}-${'token'}-`)
const VISUAL_COMPAT_COLOR_TOKEN_RE = /--color-visual-/
const LEGACY_RADIUS_TOKEN_RE = /--border-radius-/
const LEGACY_RADIUS_COMPAT_TOKEN_RE = /--radius-[a-z0-9-]*-legacy/
const GENERATED_CSS_RE = /\.generated-\d+\.css/
const SETTINGS_SHARED_STYLE_RE = /settings-shared\.styles\.css/
const SETTINGS_LEGACY_CLASS_RE = /\bsettings-(?:group|item)\b/
const UNSTYLED_BUTTON_VARIANT_RE = /variant\s*=\s*["']unstyled["']/
const TOOL_BUTTON_VARIANT_RE = /variant\s*=\s*["']tool["']/
const RAW_INPUT_RE = /<input\b/
const RAW_TEXTAREA_RE = /<textarea\b/
const RAW_SELECT_RE = /<select\b/
const STYLE_SRC_RE = /<style[^>]+src=/
const SCRIPT_STYLE_IMPORT_RE = /import\s+['"]\.\/([^'"]+\.styles\.css)['"]/g
const HORIZONTAL_STYLE_SLICE_RE = /\.(?:base|layout|panels|responsive)\.styles\.css$/
const CSS_IMPORT_RE = /@import\s+/
const BARE_Z_INDEX_RE = /z-index\s*:\s*\d+\s*;/
const Z_INDEX_NUMERIC_FALLBACK_RE = /z-index\s*:\s*var\([^)]*,\s*\d+[^)]*\)/
const BARE_MEDIA_BREAKPOINT_RE = /@media\s*\([^)]*\b\d{3,4}px\b[^)]*\)/
const LAYOUT_BYPASS_RE = /position\s*:\s*fixed|(?:min-|max-)?height\s*:\s*(?:100vh|calc\(100vh)|margin-(?:left|right)\s*:\s*(?:240|340)px/
const BASE_MODAL_LEGACY_RE = /\bclass\s*=\s*["'][^"']*\bmodal-(?:overlay|container|header|title|close-btn|body|footer|small|medium|large|full)\b/
const BASE_MODAL_CUSTOM_CLASS_RE = /<BaseModal\b[\s\S]*?\bcustom-class="([^"]+)"/g
const STATIC_CLASS_RE = /\bclass\s*=\s*["']([^"']+)["']/g
const LEGACY_CLASS_TOKENS = new Set(['btn', 'card', 'form-group', 'modal'])
const LEGACY_BUTTON_CLASS_TOKENS = new Set([
  'ui-action-btn',
  'btn-primary',
  'btn-secondary',
  'btn-danger',
  'btn-sm',
  'btn-icon',
  'primary-btn',
  'secondary-btn',
  'ghost-btn',
  'danger-btn',
])
const GENERIC_FORM_CLASS_TOKENS = new Set(['form-input', 'form-textarea'])
const CSS_LEGACY_SELECTOR_RE = /(^|[^A-Za-z0-9_-])\.(btn|card|form-group|modal)(?![A-Za-z0-9_-])/g
const CSS_LEGACY_BUTTON_SELECTOR_RE = /(^|[^A-Za-z0-9_-])\.(ui-action-btn|btn-primary|btn-secondary|btn-danger|btn-sm|btn-icon|primary-btn|secondary-btn|ghost-btn|danger-btn)(?![A-Za-z0-9_-])/g
const CSS_GENERIC_FORM_SELECTOR_RE = /(^|[^A-Za-z0-9_-])\.(form-input|form-textarea)(?![A-Za-z0-9_-])/g
const CSS_UNOWNED_GLOBAL_SELECTOR_RE = /(^|[^A-Za-z0-9_-])\.(header-content|logo-container|app-logo|app-name|header-links|tutorial-link|github-link|donate-link|header-btn|mode-btn|upload-card|thumbnail-item|status-icon|ui-form-field)(?![A-Za-z0-9_-])/g
const CSS_UI_PRIMITIVE_SELECTOR_RE = /(^|[^A-Za-z0-9_-])\.(ui-button--[a-z0-9-]+|ui-icon-button--[a-z0-9-]+|ui-form-field|ui-input|ui-select|ui-textarea)(?![A-Za-z0-9_-])/g
const TEST_UI_PRIMITIVE_INTERNAL_SELECTOR_RE = /\.(ui-button--[a-z0-9-]+|ui-icon-button--[a-z0-9-]+|ui-form-field|ui-input|ui-select|ui-textarea|ui-modal__[a-z0-9-]+)(?![A-Za-z0-9_-])/g
const CSS_BARE_UI_MODAL_SELECTOR_RE = /^\s*\.ui-modal__/m
const GLOBAL_STYLE_MODAL_DETAIL_SELECTOR_RE = /\.ui-modal__(?:body|header|footer|title|close)\b/
const GLOBAL_FORM_SKIN_SELECTOR_RE = /(\.ui-settings-field\s+\.ui-(?:input|select|textarea)|\.ui-checkbox-label\s+input(?:\[[^\]]+\])?)/g
const BUSINESS_PROVIDE_INJECT_RE = /\b(?:provide|inject)\s*\(/g
const CUSTOM_STYLE_ATTR_RE = /(?::custom-style|custom-style)\s*=\s*(["'])([\s\S]*?)\1/g
const UI_PRIMITIVE_BUSINESS_OWNER_TOKEN_RE = /--(?:bookshelf|book-card|character-studio|edit|insight|reader|studio|translate|web-import)-[A-Za-z0-9_-]+/g
const PUBLIC_PRIMITIVE_CUSTOM_PROPERTY_RE = /^--ui-/
const RAW_BUTTON_RE = /<button\b/
const RAW_BUTTON_ALLOWED_FILES = new Set([
  'src/components/ui/UiButton.vue',
  'src/components/ui/UiIconButton.vue',
])
const RAW_FORM_CONTROL_ALLOWED_FILES = new Set([
  'src/components/ui/UiCheckbox.vue',
  'src/components/ui/UiFileInput.vue',
  'src/components/ui/UiInput.vue',
  'src/components/ui/UiSelect.vue',
  'src/components/ui/UiTextarea.vue',
  'tests/unit/ui-primitives.spec.ts',
])
const TEST_PRIMITIVE_SELECTOR_ALLOWED_FILES = new Set([
  'tests/unit/baseModal.spec.ts',
  'tests/unit/ui-primitives.spec.ts',
  'tests/unit/uiArchitectureLint.spec.ts',
])
const UNSTYLED_BUTTON_ALLOWED_FILES = new Set([
  'src/components/ui/UiButton.vue',
])
const UI_PRIMITIVE_STYLE_ALLOWED_PREFIX = 'src/components/ui/'
const TELEPORT_STYLE_OWNER_ALLOWED_FILES = new Set([
  'src/components/common/BaseModal.vue',
])
const STYLE_BLOCK_RE = /<style\b[^>]*>([\s\S]*?)<\/style>/g
const SCOPED_STYLE_BLOCK_RE = /<style\b(?=[^>]*\bscoped\b)[^>]*>([\s\S]*?)<\/style>/g
const SCOPED_UI_MODAL_SELECTOR_RE = /\.ui-modal__/
const HARDCODED_COLOR_RE = /#[0-9a-fA-F]{3,8}\b|(?:rgb|rgba|hsl|hsla)\([^)]*\)/g
const LEGACY_VARIABLE_DEFINITION_RE = /--(?:text-color|bg-color|card-bg-color|shadow-color|border-color|input-bg-color|input-border-color|button-primary-bg|button-hover-bg|card-bg|hover-bg|input-bg)\s*:/
const GLOBAL_SELECTOR_RE = /:global\(([^)]*)\)/g
const VALUE_NAMED_TOKEN_RE = new RegExp(
  `--${'semantic'}-[a-z-]+-(?:hex|rgb|rgba)-[a-z0-9-]+|--${'color'}-${'token'}-[a-z0-9-]+`,
  'g',
)
const GENERATED_PALETTE_TOKEN_NAME_RE = /^--palette-(?:(?:border|color|shadow|surface|text)-\d{3}|[a-z]+-\d{2,3}-(?:[a-z]|bright|calm|clear|cool|dark|deep|dusty|fresh|light|muted|pale|quiet|rich|soft|solid|subtle|vivid|warm)(?:-[a-z]+)?)$/
const GENERATED_OWNER_TOKEN_RE = /--color-[a-z0-9-]+-(?:surface|text|border|shadow|accent)-\d{3}/g
const DOMAIN_SHARED_TOKEN_RE = /^--insight-/
const GENERATED_DOMAIN_TOKEN_RE = /^--[a-z0-9-]+-variant-\d{3}$/
const DOMAIN_SPECIFIC_GLOBAL_TOKEN_RE = /^--(?:color-edit-|color-(?:surface|text|border|shadow)-studio|shadow-edit-|shadow-studio-)/
const LEGACY_INSIGHT_DOMAIN_ALIAS_RE = /^--insight-(?:bg-[a-z0-9-]+|color-(?:primary|warning|error|danger)|primary(?:-[a-z0-9-]+)?|(?:success|warning|error|danger)(?:-color)?)$/
const VAGUE_COMPONENT_TOKEN_RE = /^--(?:app-header|base-modal|custom-select|toast-notification|ui-button|ui-icon-button|ui-input|ui-panel|ui-select|ui-textarea)-(?:(?:surface|text|border|shadow)-(?:base|primary|secondary|default|raised|muted|subtle|hover|floating|strong|inverse|selected|overlay)|(?:border|shadow)-(?:muted|subtle|strong))$/
const VALUE_NAMED_SEMANTIC_TOKEN_NAME_RE = /^--(?!palette-)[a-z0-9-]+-(?:base[0-9a-f]+|(?=[a-z0-9]*\d)(?=[a-z0-9]*[a-f])[a-z]+[a-z0-9]*|(?:light|soft|tint)\d+|[a-z]+(?:333|444|555|666|777|888|999))$/
const PALETTE_TOKEN_REFERENCE_RE = /--palette-[A-Za-z0-9_-]+/g
const FRONTEND_SCHEMA_COMPAT_RE = /\b(?:custom_openai|custom_openai_vision|legacyIds|LEGACY_STORAGE_KEY|providerSettings|deepMerge|(?:strip|sync)Legacy[A-Za-z0-9_]*|coerceLegacy[A-Za-z0-9_]*|threshold(?:48px|MangaOcr|PaddleOcr)|isJsonMode|forceJson)\b/g
const FRONTEND_SCHEMA_MAX_RETRIES_RE = /\bmaxRetries\b/g
const OPTIONAL_CURRENT_SCHEMA_VERSION_RE = /\b(?:settingsSchemaVersion|webImportSettingsSchemaVersion)\s*\?:/g
const OPENAI_MIRROR_FIELD_PATH_RE = /(?:^|\/)openaiOptions\.ts$|src\/stores\/insightStore\.ts$|src\/stores\/insight\/useInsightConfigManager\.ts$/
const OLD_IMPLEMENTATION_MINDSET_RE = /保持既有|保持当前视觉|当前视觉|复刻原版|复刻旧版|复刻自|整理自既有|完整样式(?:\s*-\s*从[^*\n\r]+)?|从\s+[^*\n\r]+\.css\s+迁移|迁移自\s+[^*\n\r]+|对应原\s+[^*\n\r]+\.js|旧版[^*\n\r]*|原版[^*\n\r]*|【(?:简化设计|增强版|优化)[^】]*】|关键修复|修复问题\d*|修复\s*P\d+|本地兼容 API|\b(?:bookshelf|edit_mode|main|events)\.js\b|\b(?:global|style|reader|manga-insight)\.css\b|迁移自旧 CSS|已迁移到 global\.css|Source:\s*[^*]*\.styles\.css|legacy UI|legacy CSS/gi
const COMPOSABLE_IMPLEMENTATION_HISTORY_RE = /从\s+[^*\n\r]+提取|【(?:简化设计|增强版|优化)[^】]*】/g
const EXPLICIT_ANY_RE = /\bas\s+any\b|:\s*any\b|\bany\[\]/g
const RELATIVE_EXPORT_RE = /\bexport\s+(?:type\s+)?(?:\*|\{[\s\S]*?\})\s+from\s+['"](\.[^'"]+)['"]/g
const TYPE_BARREL_REQUIRED_EXPORTS = new Set([
  './characterStudio',
  './webImport',
])
const CSS_BLOCK_RE = /([^{}]+)\{([^{}]*)\}/g
const CUSTOM_PROPERTY_RE = /(--[A-Za-z0-9_-]+)\s*:\s*([^;]+);/g
const VAR_REFERENCE_RE = /var\(\s*(--[A-Za-z0-9_-]+)/g
const CUSTOM_PROPERTY_MUTATION_RE = /\.(?:setProperty|removeProperty)\(\s*['"](--[A-Za-z0-9_-]+)['"]/g
const LEGACY_SHORT_ALIAS_TOKEN_RE = /^--(?:bg-[a-z0-9-]+|btn-[a-z0-9-]+|primary(?:-[a-z0-9-]+)?|danger(?:-[a-z0-9-]+)?|success(?:-[a-z0-9-]+)?|warning(?:-[a-z0-9-]+)?|error(?:-[a-z0-9-]+)?|reader-bg-color)$/
const LEGACY_COMPATIBILITY_VARIABLES = new Set([
  '--text-color',
  '--bg-color',
  '--card-bg-color',
  '--shadow-color',
  '--border-color',
  '--input-bg-color',
  '--input-border-color',
  '--button-primary-bg',
  '--button-hover-bg',
  '--card-bg',
  '--hover-bg',
  '--input-bg',
])
const LEGACY_GLOBAL_VARIABLES = new Set([
  '--text-primary',
  '--text-secondary',
  '--text-muted',
  '--color-primary',
  '--success-color',
  '--error-color',
  '--warning-color',
])
const LEGACY_SHORT_ALIAS_VARIABLES = new Set([
  '--bg-primary',
  '--bg-secondary',
  '--bg-tertiary',
  '--bg-hover',
  '--primary',
  '--primary-light',
  '--primary-dark',
  '--primary-hover',
  '--danger',
  '--danger-color',
  '--danger-hover-color',
  '--success',
  '--success-bg',
  '--success-text',
  '--success-border',
  '--warning',
  '--warning-bg',
  '--warning-text',
  '--warning-border',
  '--error',
  '--error-bg',
  '--error-text',
  '--error-border',
  '--reader-bg-color',
])
const RESERVED_GLOBAL_CUSTOM_PROPERTIES = new Set([
  '--text-primary',
  '--text-secondary',
  '--text-muted',
  '--text-color',
  '--bg-primary',
  '--bg-secondary',
  '--bg-tertiary',
  '--bg-hover',
  '--bg-color',
  '--card-bg-color',
  '--shadow-color',
  '--border-color',
  '--input-bg-color',
  '--input-border-color',
  '--button-primary-bg',
  '--button-hover-bg',
  '--card-bg',
  '--hover-bg',
  '--input-bg',
  '--color-primary',
  '--primary',
  '--primary-light',
  '--primary-dark',
  '--success-color',
  '--success',
  '--warning-color',
  '--warning',
  '--error-color',
  '--error',
  '--danger',
  ...LEGACY_SHORT_ALIAS_VARIABLES,
])
const UI_MINDSET_SCAN_ROOTS = [
  '.stylelintignore',
  '.stylelintrc.json',
  'eslint.config.js',
  'package.json',
  'playwright.config.ts',
  'postcss.config.mjs',
  'src/api/',
  'src/config/',
  'src/components/',
  'src/composables/',
  'src/stores/',
  'src/styles/',
  'src/types/',
  'src/utils/',
  'src/views/',
  'tests/',
  'tsconfig.app.json',
  'tsconfig.json',
  'tsconfig.node.json',
  'vite.config.ts',
  'vitest.config.ts',
]
const PAGE_CUSTOM_PROPERTY_PREFIXES = new Map([
  ['src/views/BookshelfView.vue', '--bookshelf-'],
  ['src/views/CharacterStudioView.vue', '--studio-'],
  ['src/views/CharacterStudioView.global.styles.css', '--studio-'],
  ['src/views/InsightView.vue', '--insight-'],
  ['src/views/InsightView.styles.css', '--insight-'],
  ['src/views/ReaderView.vue', '--reader-'],
  ['src/views/TranslateView.vue', '--translate-'],
  ['src/views/TranslateView.styles.css', '--translate-'],
])
const CRITICAL_VISUAL_COVERAGE = new Map([
  ['TranslateView empty shell', 'translate workspace empty state keeps its layout contract'],
  ['TranslateView loaded sidebars', 'translate loaded workspace keeps fixed sidebar sizing contract'],
  ['EditWorkspace dark shell', 'translate edit workspace keeps dark editor shell contract'],
  ['EditWorkspace selected bubble editor', 'translate edit workspace selected bubble keeps editor panel contract'],
  ['InsightView selected-book sidebars', 'insight selected-book sidebars keep their gutter contract'],
  ['ReaderView immersive shell', 'reader loaded state keeps its layout contract'],
  ['CharacterStudioView empty shell', 'character studio empty workspace keeps its layout contract'],
  ['CharacterStudioView editor preview shell', 'character studio editor and preview keep split workspace contract'],
])
const VISUAL_REGRESSION_SPEC = join(process.cwd(), 'tests/visual/ui-regression.spec.ts')
const COLOR_LITERAL_ALLOWED_FILES = new Set([
  ...TOKEN_FILE_ORDER,
])
const LINE_BUDGET_EXCLUDED_CSS = new Set([
  ...TOKEN_FILE_ORDER,
])
const ARCHITECTURE_DEBT_BUDGETS = {
  businessVisualColorTokenReferences: 0,
  legacyRadiusCompatTokenReferences: 0,
  rawInputs: 0,
  rawTextareas: 0,
  rawSelects: 0,
  toolButtonVariants: 0,
  styleSrcEntries: 0,
  cssImports: 0,
  generatedCssReferences: 0,
  colorTokenReferencesOutsideTokens: 0,
  globalSelectors: 0,
  settingsSharedReferences: 0,
  valueNamedTokenReferences: 0,
}

const LAYOUT_OWNER_ALLOWED_FILES = new Set([
  'src/components/common/AppHeader.vue',
  'src/components/common/BaseModal.vue',
  'src/components/common/CustomSelect.vue',
  'src/components/common/ToastNotification.vue',
  'src/components/ui/AppShell.vue',
  'src/components/ui/OverlayLayer.vue',
  'src/components/ui/SidebarLayout.vue',
  'src/components/edit/EditWorkspace.vue',
])

const LARGE_INTERACTION_OWNER_FILES = new Set([
  'src/components/edit/BubbleEditor.vue',
  'src/components/edit/BubbleOverlay.vue',
  'src/components/edit/EditToolbar.vue',
  'src/components/edit/EditWorkspace.vue',
])

const failures = []
const valueNamedTokenCandidates = new Map()
const tokenArchitectureStats = {
  rootTokens: 0,
  bodyTokens: 0,
  rootDependencies: 0,
}
const visualCoverageEntries = []
const heavyOwnerReviewSignals = []
const architectureDebtUsage = Object.fromEntries(
  Object.keys(ARCHITECTURE_DEBT_BUDGETS).map(key => [key, 0])
)

function scanPath(path) {
  const stat = statSync(path)
  if (stat.isDirectory()) {
    walk(path)
    return
  }
  if (path.endsWith('.vue') || path.endsWith('.css')) {
    checkFile(path)
    return
  }
  if (/\.(?:js|jsx|ts|tsx|json)$/.test(path) || path.endsWith('.stylelintignore')) {
    checkScriptFile(path)
  }
}

function walk(dir) {
  for (const entry of readdirSync(dir)) {
    scanPath(join(dir, entry))
  }
}

function addFailure(path, message) {
  failures.push(`${relative(process.cwd(), path)}: ${message}`)
}

function normalizePath(path) {
  if (TOKENS_FIXTURE_INDEX >= 0 && TOKENS_FIXTURE_PATH && resolve(path) === resolve(process.cwd(), process.argv[TOKENS_FIXTURE_INDEX + 1] || '')) {
    return TOKENS_FIXTURE_PATH
  }
  if (SOURCE_FIXTURE && resolve(path) === SOURCE_FIXTURE && SOURCE_FIXTURE_PATH) {
    return SOURCE_FIXTURE_PATH
  }
  return relative(process.cwd(), path).replaceAll('\\', '/')
}

function isTokenFile(normalizedPath) {
  return TOKEN_FILE_ORDER.includes(normalizedPath)
}

function isDomainTokenFile(normalizedPath) {
  return normalizedPath === 'src/styles/tokens/domain.css'
}

function isComponentTokenFile(normalizedPath) {
  return normalizedPath === 'src/styles/tokens/component.css'
}

function addHeavyOwnerReviewSignal(path, kind, lines, limit, reason) {
  heavyOwnerReviewSignals.push(`${normalizePath(path)}: ${kind} ${lines}/${limit} (${reason})`)
}

function checkLocalBuildArtifact(path, normalizedPath) {
  if (/^(?:build_output\.txt|vite-dev\.log)$/.test(normalizedPath)) {
    addFailure(path, 'local build/dev log files are not allowed in frontend source; remove the artifact and keep it ignored')
  }
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function stripCssComments(content) {
  return content.replace(/\/\*[\s\S]*?\*\//g, '')
}

function countRegexMatches(content, regex) {
  return [...content.matchAll(new RegExp(regex.source, regex.flags.includes('g') ? regex.flags : `${regex.flags}g`))].length
}

function parseCustomPropertyScopes(content) {
  const scopes = {
    root: new Map(),
    body: new Map(),
    other: new Map(),
  }
  const css = stripCssComments(content)

  for (const block of css.matchAll(CSS_BLOCK_RE)) {
    const selectorText = block[1].trim().split(/\r?\n/).pop()?.trim() || ''
    const selectors = selectorText.split(',').map(selector => selector.trim())
    const declarations = block[2]
    let scope = 'other'
    if (selectors.includes(':root')) {
      scope = 'root'
    } else if (selectors.includes('body')) {
      scope = 'body'
    }

    for (const declaration of declarations.matchAll(CUSTOM_PROPERTY_RE)) {
      const [, token, value] = declaration
      const target = scopes[scope]
      if (!target.has(token)) {
        target.set(token, [])
      }
      target.get(token).push(value.trim())
    }
  }

  return scopes
}

function referencedTokens(values) {
  const refs = new Set()
  for (const value of values) {
    for (const match of value.matchAll(VAR_REFERENCE_RE)) {
      refs.add(match[1])
    }
  }
  return refs
}

function referencedCustomProperties(content, tokens) {
  const refs = new Set()
  for (const match of content.matchAll(VAR_REFERENCE_RE)) {
    const token = match[1]
    if (tokens.has(token)) {
      refs.add(token)
    }
  }
  return refs
}

function isLegacyShortAliasVariable(token) {
  return LEGACY_SHORT_ALIAS_VARIABLES.has(token) || LEGACY_SHORT_ALIAS_TOKEN_RE.test(token)
}

function referencedOrMutatedCustomProperties(content, tokens) {
  const refs = referencedCustomProperties(content, tokens)
  for (const match of content.matchAll(CUSTOM_PROPERTY_MUTATION_RE)) {
    const token = match[1]
    if (tokens.has(token)) {
      refs.add(token)
    }
  }
  return refs
}

function referencedOrMutatedLegacyShortAliases(content) {
  const refs = new Set()
  for (const match of content.matchAll(VAR_REFERENCE_RE)) {
    const token = match[1]
    if (isLegacyShortAliasVariable(token)) {
      refs.add(token)
    }
  }
  for (const match of content.matchAll(CUSTOM_PROPERTY_MUTATION_RE)) {
    const token = match[1]
    if (isLegacyShortAliasVariable(token)) {
      refs.add(token)
    }
  }
  return refs
}

function isReservedGlobalCustomProperty(token) {
  return RESERVED_GLOBAL_CUSTOM_PROPERTIES.has(token) || isLegacyShortAliasVariable(token)
}

function matchedTokens(content, regex) {
  return new Set([...content.matchAll(regex)].map(match => match[0]))
}

function customPropertyDefinitions(content, path) {
  const styleContents = extractStyleContents(content, path)
  const definitions = []
  for (const styleContent of styleContents) {
    const css = stripCssComments(styleContent)
    for (const block of css.matchAll(CSS_BLOCK_RE)) {
      for (const declaration of block[2].matchAll(CUSTOM_PROPERTY_RE)) {
        definitions.push(declaration[1])
      }
    }
  }
  return definitions
}

function shouldScanOldImplementationMindset(normalizedPath) {
  if (SOURCE_FIXTURE) {
    return true
  }
  return UI_MINDSET_SCAN_ROOTS.some(prefix => normalizedPath.startsWith(prefix))
}

function checkOldImplementationMindset(path, normalizedPath, content) {
  if (!shouldScanOldImplementationMindset(normalizedPath)) {
    return
  }

  const phrases = new Set([...content.matchAll(OLD_IMPLEMENTATION_MINDSET_RE)].map(match => match[0]))
  if (phrases.size === 0) {
    return
  }

  addFailure(
    path,
    `old implementation mindset comment/text ${[...phrases].join(', ')} is not allowed in UI source; describe the current behavior contract instead`
  )
}

function isProductionComposable(normalizedPath) {
  return normalizedPath.startsWith('src/composables/')
    && !normalizedPath.endsWith('.test.ts')
    && !normalizedPath.endsWith('.spec.ts')
}

function isProductionTypeFile(normalizedPath) {
  return normalizedPath.startsWith('src/types/')
    && normalizedPath.endsWith('.ts')
    && !normalizedPath.endsWith('.test.ts')
    && !normalizedPath.endsWith('.spec.ts')
}

function isProductionUtilFile(normalizedPath) {
  return normalizedPath.startsWith('src/utils/')
    && normalizedPath.endsWith('.ts')
    && !normalizedPath.endsWith('.test.ts')
    && !normalizedPath.endsWith('.spec.ts')
}

function checkComposableSourceHygiene(path, normalizedPath, content, contentWithoutComments) {
  if (!isProductionComposable(normalizedPath)) {
    return
  }

  const historyPhrases = new Set([...content.matchAll(COMPOSABLE_IMPLEMENTATION_HISTORY_RE)].map(match => match[0]))
  if (historyPhrases.size > 0) {
    addFailure(
      path,
      `composable implementation-history wording ${[...historyPhrases].join(', ')} is not allowed; describe the current behavior contract instead`
    )
  }

  const explicitAny = new Set([...contentWithoutComments.matchAll(EXPLICIT_ANY_RE)].map(match => match[0]))
  if (explicitAny.size > 0) {
    addFailure(
      path,
      `explicit any in production composable ${[...explicitAny].join(', ')} is not allowed; use the current frontend types or an unknown boundary guard`
    )
  }
}

function checkTypeSourceHygiene(path, normalizedPath, contentWithoutComments) {
  if (!isProductionTypeFile(normalizedPath)) {
    return
  }

  const explicitAny = new Set([...contentWithoutComments.matchAll(EXPLICIT_ANY_RE)].map(match => match[0]))
  if (explicitAny.size > 0) {
    addFailure(
      path,
      `explicit any in production type ${[...explicitAny].join(', ')} is not allowed; use current frontend contracts or an unknown boundary guard`
    )
  }
}

function checkUtilSourceHygiene(path, normalizedPath, contentWithoutComments) {
  if (!isProductionUtilFile(normalizedPath)) {
    return
  }

  const explicitAny = new Set([...contentWithoutComments.matchAll(EXPLICIT_ANY_RE)].map(match => match[0]))
  if (explicitAny.size > 0) {
    addFailure(
      path,
      `explicit any in production util ${[...explicitAny].join(', ')} is not allowed; use current frontend types or an unknown boundary guard`
    )
  }
}

function relativeExportExists(path, specifier) {
  const base = resolve(dirname(path), specifier)
  const candidates = [
    base,
    `${base}.ts`,
    `${base}.tsx`,
    `${base}.vue`,
    `${base}.js`,
    `${base}.mjs`,
    `${base}.css`,
    join(base, 'index.ts'),
    join(base, 'index.tsx'),
    join(base, 'index.vue'),
    join(base, 'index.js'),
    join(base, 'index.mjs'),
  ]

  return candidates.some(candidate => existsSync(candidate))
}

function relativeExportSpecifiers(contentWithoutComments) {
  return [...contentWithoutComments.matchAll(RELATIVE_EXPORT_RE)]
    .map(match => match[1])
    .filter(Boolean)
}

function checkRelativeExports(path, contentWithoutComments) {
  const unresolved = []
  for (const specifier of relativeExportSpecifiers(contentWithoutComments)) {
    if (!relativeExportExists(path, specifier)) {
      unresolved.push(specifier)
    }
  }

  if (unresolved.length > 0) {
    addFailure(
      path,
      `unresolved relative export(s) ${[...new Set(unresolved)].join(', ')} are not allowed; remove stale barrel exports or create the exported module`
    )
  }
}

function checkTypesBarrelExports(path, normalizedPath, contentWithoutComments) {
  if (normalizedPath !== 'src/types/index.ts') {
    return
  }

  const exports = new Set(relativeExportSpecifiers(contentWithoutComments))
  const missing = [...TYPE_BARREL_REQUIRED_EXPORTS].filter(specifier => !exports.has(specifier))
  if (missing.length > 0) {
    addFailure(
      path,
      `types barrel missing export(s) ${missing.join(', ')}; keep current page-domain type modules available from @/types`
    )
  }
}

function checkFrontendSchemaCompatibility(path, normalizedPath, contentWithoutComments) {
  const compatMatches = new Set([...contentWithoutComments.matchAll(FRONTEND_SCHEMA_COMPAT_RE)].map(match => match[0]))
  if (SOURCE_FIXTURE || OPENAI_MIRROR_FIELD_PATH_RE.test(normalizedPath)) {
    for (const match of contentWithoutComments.matchAll(FRONTEND_SCHEMA_MAX_RETRIES_RE)) {
      compatMatches.add(match[0])
    }
  }

  if (compatMatches.size > 0) {
    addFailure(
      path,
      `legacy frontend schema/provider reference(s) ${[...compatMatches].join(', ')} are not allowed; use the current provider ids and current nested settings schema`
    )
  }

  const optionalSchemaVersions = new Set([...contentWithoutComments.matchAll(OPTIONAL_CURRENT_SCHEMA_VERSION_RE)].map(match => match[0]))
  if (optionalSchemaVersions.size > 0) {
    addFailure(
      path,
      `current schema version fields must be required, not optional: ${[...optionalSchemaVersions].join(', ')}`
    )
  }
}

function checkCustomPropertyOwnership(path, normalizedPath, content) {
  if (isTokenFile(normalizedPath)) {
    return
  }

  const definitions = customPropertyDefinitions(content, path)
  if (definitions.length === 0) {
    return
  }

  for (const token of definitions) {
    if (isReservedGlobalCustomProperty(token)) {
      addFailure(
        path,
        `reserved global CSS variable ${token} may only be defined in src/styles/tokens/*; use an owner-namespaced variable such as --insight-* in business CSS`
      )
    }
  }

  const pagePrefix = PAGE_CUSTOM_PROPERTY_PREFIXES.get(normalizedPath)
  if (!pagePrefix) {
    return
  }

  const unownedTokens = definitions.filter(token => (
    !token.startsWith(pagePrefix)
    && !PUBLIC_PRIMITIVE_CUSTOM_PROPERTY_RE.test(token)
  ))
  if (unownedTokens.length > 0) {
    addFailure(
      path,
      `page CSS custom property definition(s) ${unownedTokens.join(', ')} must use the owner namespace ${pagePrefix}* or move to src/styles/tokens/*`
    )
  }
}

function appendTokenScopes(target, source) {
  for (const scopeName of ['root', 'body', 'other']) {
    for (const [token, values] of source[scopeName]) {
      if (!target[scopeName].has(token)) {
        target[scopeName].set(token, [])
      }
      target[scopeName].get(token).push(...values)
    }
  }
}

function checkTokenCycles(path, rootTokens) {
  const visiting = new Set()
  const visited = new Set()

  function visit(token, stack) {
    if (visited.has(token)) {
      return
    }
    if (visiting.has(token)) {
      const cycle = [...stack.slice(stack.indexOf(token)), token].join(' -> ')
      addFailure(path, `token dependency cycle detected: ${cycle}`)
      return
    }

    visiting.add(token)
    for (const referencedToken of referencedTokens(rootTokens.get(token) || [])) {
      if (rootTokens.has(referencedToken)) {
        visit(referencedToken, [...stack, referencedToken])
      }
    }
    visiting.delete(token)
    visited.add(token)
  }

  for (const token of rootTokens.keys()) {
    visit(token, [token])
  }
}

function checkTokenDependencyArchitecture(paths) {
  const scopes = {
    root: new Map(),
    body: new Map(),
    other: new Map(),
  }

  for (const path of paths) {
    const content = readFileSync(path, 'utf8')
    const normalizedPath = normalizePath(path)
    const rootTokenDefinitions = [...parseCustomPropertyScopes(content).root.keys()]
    const domainSpecificGlobalTokens = rootTokenDefinitions.filter(token => DOMAIN_SPECIFIC_GLOBAL_TOKEN_RE.test(token))
    if (domainSpecificGlobalTokens.length > 0) {
      addFailure(
        path,
        `domain-specific semantic token definition(s) ${domainSpecificGlobalTokens.join(', ')} are not allowed in global token files; move them into domain tokens or scoped owner variables`
      )
    }
    if (isDomainTokenFile(normalizedPath)) {
      const domainTokens = rootTokenDefinitions
      const domainTokenCount = domainTokens.length
      if (domainTokenCount > DOMAIN_TOKEN_MAX_DEFINITIONS) {
        addFailure(
          path,
          `domain.css defines ${domainTokenCount} tokens; keep domain tokens below ${DOMAIN_TOKEN_MAX_DEFINITIONS} by moving component-private tokens into scoped owners`
        )
      }
      const componentPrivateTokens = domainTokens.filter(token => !DOMAIN_SHARED_TOKEN_RE.test(token))
      if (componentPrivateTokens.length > 0) {
        addFailure(
          path,
          `component-private domain token definition(s) ${componentPrivateTokens.join(', ')} are not allowed; move them into the owning component scoped style`
        )
      }
      const generatedDomainTokens = domainTokens.filter(token => GENERATED_DOMAIN_TOKEN_RE.test(token))
      if (generatedDomainTokens.length > 0) {
        addFailure(
          path,
          `generated domain token definition(s) ${generatedDomainTokens.join(', ')} are not allowed; name shared domain tokens by role`
        )
      }
      const legacyInsightAliasTokens = domainTokens.filter(token => LEGACY_INSIGHT_DOMAIN_ALIAS_RE.test(token))
      if (legacyInsightAliasTokens.length > 0) {
        addFailure(
          path,
          `legacy-style insight domain alias token definition(s) ${legacyInsightAliasTokens.join(', ')} are not allowed; use --insight-surface-*, --insight-action-*, or --insight-status-* names`
        )
      }
    }
    if (isComponentTokenFile(normalizedPath)) {
      const vagueComponentTokens = rootTokenDefinitions.filter(token => VAGUE_COMPONENT_TOKEN_RE.test(token))
      if (vagueComponentTokens.length > 0) {
        addFailure(
          path,
          `vague component token definition(s) ${vagueComponentTokens.join(', ')} are not allowed; name component tokens by their concrete UI role`
        )
      }
    }
    appendTokenScopes(scopes, parseCustomPropertyScopes(content))
  }

  const reportPath = paths[0]
  const rootTokens = scopes.root
  const bodyTokens = scopes.body
  const rootTokenNames = new Set(rootTokens.keys())
  const bodyTokenNames = new Set(bodyTokens.keys())

  tokenArchitectureStats.rootTokens = rootTokens.size
  tokenArchitectureStats.bodyTokens = bodyTokens.size

  for (const token of bodyTokenNames) {
    addFailure(reportPath, `body defines ${token}; body compatibility aliases are no longer allowed`)
  }

  const generatedPaletteDefinitions = [...rootTokenNames].filter(token => GENERATED_PALETTE_TOKEN_NAME_RE.test(token))
  if (generatedPaletteDefinitions.length > 0) {
    addFailure(
      reportPath,
      `generated palette token definition(s) ${generatedPaletteDefinitions.join(', ')} are not allowed; define the final semantic token value directly in src/styles/tokens/*`
    )
  }

  const valueNamedDefinitions = [...rootTokenNames].filter(token => VALUE_NAMED_SEMANTIC_TOKEN_NAME_RE.test(token))
  if (valueNamedDefinitions.length > 0) {
    addFailure(
      reportPath,
      `value-named token definition(s) ${valueNamedDefinitions.join(', ')} are not allowed; name tokens by role, not raw value fragments`
    )
  }

  for (const [token, values] of rootTokens) {
    for (const referencedToken of referencedTokens(values)) {
      tokenArchitectureStats.rootDependencies += 1
      if (bodyTokenNames.has(referencedToken)) {
        addFailure(reportPath, `:root token ${token} cannot reference body-scoped token ${referencedToken}; define dependencies in :root`)
        continue
      }
      if (!rootTokenNames.has(referencedToken)) {
        addFailure(reportPath, `:root token ${token} references undefined token ${referencedToken}`)
      }
    }
  }

  for (const [token, values] of bodyTokens) {
    for (const referencedToken of referencedTokens(values)) {
      if (!rootTokenNames.has(referencedToken) && !bodyTokenNames.has(referencedToken)) {
        addFailure(reportPath, `body token ${token} references undefined token ${referencedToken}`)
      }
    }
  }

  checkTokenCycles(reportPath, rootTokens)
}

function checkCriticalVisualCoverage() {
  const content = readFileSync(VISUAL_REGRESSION_SPEC, 'utf8')
  for (const [state, testName] of CRITICAL_VISUAL_COVERAGE) {
    if (!content.includes(testName)) {
      addFailure(VISUAL_REGRESSION_SPEC, `critical UI visual coverage missing for ${state}: expected test "${testName}"`)
      continue
    }
    visualCoverageEntries.push(`${state}: ${testName}`)
  }
}

function extractStyleContents(content, path) {
  if (!path.endsWith('.vue')) {
    return [content]
  }
  return [...content.matchAll(STYLE_BLOCK_RE)].map(match => match[1])
}

function stripCustomPropertyDeclarations(css) {
  return css.replace(CUSTOM_PROPERTY_RE, '')
}

function checkHardcodedColors(path, normalizedPath, content) {
  if (COLOR_LITERAL_ALLOWED_FILES.has(normalizedPath)) {
    return
  }

  const styleContents = extractStyleContents(content, path)
  const colorMatches = new Set()
  for (const styleContent of styleContents) {
    const css = stripCustomPropertyDeclarations(stripCssComments(styleContent))
    for (const match of css.matchAll(HARDCODED_COLOR_RE)) {
      colorMatches.add(match[0])
    }
  }

  if (colorMatches.size > 0) {
    addFailure(path, `style color literal(s) ${[...colorMatches].slice(0, 6).join(', ')} are not allowed outside owner custom property definitions`)
  }
}

function checkUiPrimitiveSelectors(path, normalizedPath, content) {
  if (normalizedPath.startsWith(UI_PRIMITIVE_STYLE_ALLOWED_PREFIX)) {
    return
  }

  const styleContents = extractStyleContents(content, path)
  const primitiveSelectors = new Set()
  for (const styleContent of styleContents) {
    for (const match of stripCssComments(styleContent).matchAll(CSS_UI_PRIMITIVE_SELECTOR_RE)) {
      primitiveSelectors.add(match[2])
    }
  }

  if (primitiveSelectors.size > 0) {
    addFailure(path, `UI primitive selector(s) ${[...primitiveSelectors].map(token => `.${token}`).join(', ')} are not allowed in business CSS; use primitive props/classes or a business-owned class`)
  }
}

function checkUiPrimitiveBusinessOwnerTokens(path, normalizedPath, contentWithoutComments) {
  if (!normalizedPath.startsWith(UI_PRIMITIVE_STYLE_ALLOWED_PREFIX)) {
    return
  }

  const businessOwnerTokens = matchedTokens(contentWithoutComments, UI_PRIMITIVE_BUSINESS_OWNER_TOKEN_RE)
  if (businessOwnerTokens.size > 0) {
    addFailure(
      path,
      `business owner token reference(s) ${[...businessOwnerTokens].join(', ')} are not allowed in UI primitives; expose a --ui-* primitive variable and let business owners override it`
    )
  }
}

function checkGlobalFormSkinSelectors(path, content) {
  if (!normalizePath(path).endsWith('form-primitives.css')) {
    return
  }
  const selectors = new Set()
  for (const styleContent of extractStyleContents(content, path)) {
    for (const match of stripCssComments(styleContent).matchAll(GLOBAL_FORM_SKIN_SELECTOR_RE)) {
      selectors.add(match[1].replace(/\s+/g, ' '))
    }
  }
  if (selectors.size > 0) {
    addFailure(
      path,
      `global form skin selector(s) ${[...selectors].join(', ')} are not allowed; express settings form layout through UiField/UiPanel/Form primitives`
    )
  }
}

function checkBusinessProvideInject(path, normalizedPath, contentWithoutComments) {
  if (!path.endsWith('.vue') || normalizedPath.startsWith('src/components/ui/')) {
    return
  }
  const matches = new Set([...contentWithoutComments.matchAll(BUSINESS_PROVIDE_INJECT_RE)].map(match => match[0].replace(/\s*\($/, '')))
  if (matches.size > 0) {
    addFailure(
      path,
      `business provide/inject usage ${[...matches].join(', ')} is not allowed for UI component splitting; pass typed props/emits or move shared workflow state into a composable owner`
    )
  }
}

function checkRawCustomStyleValues(path, content) {
  const rawValues = new Set()
  for (const match of content.matchAll(CUSTOM_STYLE_ATTR_RE)) {
    const attr = match[0]
    for (const color of attr.matchAll(HARDCODED_COLOR_RE)) {
      rawValues.add(color[0])
    }
  }
  if (rawValues.size > 0) {
    addFailure(
      path,
      `raw visual customStyle value(s) ${[...rawValues].join(', ')} are not allowed; use BaseModal layout props, semantic tokens, or scoped owner variables`
    )
  }
}

function checkBaseModalCustomStyleUsage(path, normalizedPath, content) {
  if (normalizedPath === 'src/components/common/BaseModal.vue') {
    return
  }
  if (CUSTOM_STYLE_ATTR_RE.test(content)) {
    addFailure(
      path,
      'BaseModal customStyle is not allowed in business UI; use explicit BaseModal layout props or scoped owner variables'
    )
  }
}

function isTestFile(normalizedPath) {
  return normalizedPath.startsWith('tests/')
    || /\.test\.[jt]sx?$/.test(normalizedPath)
    || /\.spec\.[jt]sx?$/.test(normalizedPath)
}

function checkTestPrimitiveInternalSelectors(path, normalizedPath, contentWithoutComments) {
  if (!isTestFile(normalizedPath) || TEST_PRIMITIVE_SELECTOR_ALLOWED_FILES.has(normalizedPath)) {
    return
  }

  const selectors = new Set()
  for (const match of contentWithoutComments.matchAll(TEST_UI_PRIMITIVE_INTERNAL_SELECTOR_RE)) {
    selectors.add(`.${match[1]}`)
  }

  if (selectors.size > 0) {
    addFailure(
      path,
      `primitive internal class selector(s) ${[...selectors].join(', ')} are not allowed in tests; query by role, label, text, data-testid, or a business-owned public class`
    )
  }
}

function checkFile(path) {
  const content = readFileSync(path, 'utf8')
  const normalizedPath = normalizePath(path)
  const contentWithoutComments = stripCssComments(content)

  checkLocalBuildArtifact(path, normalizedPath)
  checkTestPrimitiveInternalSelectors(path, normalizedPath, contentWithoutComments)
  checkOldImplementationMindset(path, normalizedPath, content)
  checkComposableSourceHygiene(path, normalizedPath, content, contentWithoutComments)
  checkTypeSourceHygiene(path, normalizedPath, contentWithoutComments)
  checkUtilSourceHygiene(path, normalizedPath, contentWithoutComments)
  checkRelativeExports(path, contentWithoutComments)
  checkTypesBarrelExports(path, normalizedPath, contentWithoutComments)
  checkFrontendSchemaCompatibility(path, normalizedPath, contentWithoutComments)
  checkCustomPropertyOwnership(path, normalizedPath, content)
  checkGlobalFormSkinSelectors(path, content)
  checkBusinessProvideInject(path, normalizedPath, contentWithoutComments)
  checkRawCustomStyleValues(path, content)
  checkBaseModalCustomStyleUsage(path, normalizedPath, content)
  checkUiPrimitiveBusinessOwnerTokens(path, normalizedPath, contentWithoutComments)

  if (GENERATED_CSS_RE.test(normalizedPath)) {
    addFailure(path, 'generated CSS files are not allowed; use co-located named .styles.css or component scoped styles')
  }

  if (HORIZONTAL_STYLE_SLICE_RE.test(normalizedPath)) {
    addFailure(path, 'horizontal style slice names (.base/.layout/.panels/.responsive.styles.css) are not allowed; merge small owners or split by real UI responsibility')
  }

  if (LAYOUT_BYPASS_RE.test(contentWithoutComments) && !isTokenFile(normalizedPath)) {
    if (!LAYOUT_OWNER_ALLOWED_FILES.has(normalizedPath)) {
      addFailure(path, 'layout bypass detected (fixed viewport, 100vh calc, or sidebar margin); move the layout algorithm to AppShell/SidebarLayout or a permanent shell/overlay owner')
    }
  }

  if (path.endsWith('.vue') && VUE_STYLE_RE.test(content)) {
    addFailure(path, 'component styles must be scoped; use :global(...) only for explicit Teleport/BaseModal reach-through')
  }

  if (path.endsWith('.vue')) {
    const contentWithoutStyleBlocks = content.replace(STYLE_BLOCK_RE, '')
    const lineCount = contentWithoutStyleBlocks.split(/\r?\n/).length
    const styleLineCount = [...content.matchAll(/<style\b[\s\S]*?<\/style>/g)]
      .map(match => match[0].split(/\r?\n/).length)
      .reduce((sum, count) => sum + count, 0)

    const sfcLineLimit = LARGE_INTERACTION_OWNER_FILES.has(normalizedPath)
      ? SFC_SHELL_MAX_LINES
      : SFC_MAX_LINES
    if (lineCount > sfcLineLimit) {
      addHeavyOwnerReviewSignal(path, 'SFC owner lines', lineCount, sfcLineLimit, 'review owner boundaries; do not split solely for line count')
    }
    if (styleLineCount > SFC_MAX_STYLE_LINES) {
      addHeavyOwnerReviewSignal(path, 'style owner lines', styleLineCount, SFC_MAX_STYLE_LINES, 'style owner is heavy; split only by real UI owner')
    }
    if (!TELEPORT_STYLE_OWNER_ALLOWED_FILES.has(normalizedPath)) {
      for (const match of content.matchAll(SCOPED_STYLE_BLOCK_RE)) {
        if (SCOPED_UI_MODAL_SELECTOR_RE.test(stripCssComments(match[1]))) {
          addFailure(path, 'scoped styles must not target BaseModal .ui-modal__* internals; move Teleport/slot reach-through to an explicit *.global.styles.css owner')
          break
        }
      }
    }
    if (RAW_BUTTON_RE.test(content) && !RAW_BUTTON_ALLOWED_FILES.has(normalizedPath)) {
      addFailure(path, 'raw <button> is not allowed in UI source; use UiButton or UiIconButton')
    }
    if (!RAW_FORM_CONTROL_ALLOWED_FILES.has(normalizedPath)) {
      architectureDebtUsage.rawInputs += countRegexMatches(contentWithoutComments, RAW_INPUT_RE)
      architectureDebtUsage.rawTextareas += countRegexMatches(contentWithoutComments, RAW_TEXTAREA_RE)
      architectureDebtUsage.rawSelects += countRegexMatches(contentWithoutComments, RAW_SELECT_RE)
    }
    architectureDebtUsage.toolButtonVariants += countRegexMatches(contentWithoutComments, TOOL_BUTTON_VARIANT_RE)
    architectureDebtUsage.styleSrcEntries += countRegexMatches(contentWithoutComments, STYLE_SRC_RE)
    architectureDebtUsage.generatedCssReferences += countRegexMatches(contentWithoutComments, GENERATED_CSS_RE)

    for (const match of contentWithoutComments.matchAll(SCRIPT_STYLE_IMPORT_RE)) {
      const importedStyle = match[1]
      if (!/^[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*\.styles\.css$/.test(importedStyle)) {
        addFailure(path, `script CSS import "./${importedStyle}" must be a co-located, owner-named *.styles.css file`)
      }
      if (!importedStyle.endsWith('.global.styles.css')) {
        addFailure(path, `ordinary script CSS imports are not allowed ("./${importedStyle}"); keep styles in the owning component <style scoped> or use an explicit *.global.styles.css Teleport/slot owner`)
      }
      if (HORIZONTAL_STYLE_SLICE_RE.test(importedStyle)) {
        addFailure(path, `script CSS import "./${importedStyle}" uses a horizontal slice name; use one owner file or a real responsibility name`)
      }
    }

    if (UNSTYLED_BUTTON_VARIANT_RE.test(content) && !UNSTYLED_BUTTON_ALLOWED_FILES.has(normalizedPath)) {
      addFailure(path, 'variant="unstyled" is not allowed in business UI; use a named UiButton variant or UiIconButton')
    }
    if (SCOPED_SHARED_STYLE_RE.test(content)) {
      addFailure(path, 'settings-shared.styles.css must not be injected through scoped component style; use UI primitives or the single global style entry')
    }
  }

  if (path.endsWith('.css') && !LINE_BUDGET_EXCLUDED_CSS.has(normalizedPath)) {
    const lineCount = content.replace(/\r?\n$/, '').split(/\r?\n/).length
    if (normalizedPath.endsWith('.global.styles.css')) {
      if (lineCount > GLOBAL_STYLE_MAX_LINES) {
        addFailure(path, `global style owner has ${lineCount} lines; Teleport shell styles must stay at or below ${GLOBAL_STYLE_MAX_LINES} lines`)
      }
      if (GLOBAL_STYLE_MODAL_DETAIL_SELECTOR_RE.test(content)) {
        addFailure(path, 'global style owners may only target Teleport shell/container state; do not override .ui-modal__body/header/footer/title/close')
      }
    }
    if (lineCount > CSS_MAX_LINES) {
      addHeavyOwnerReviewSignal(path, 'CSS owner lines', lineCount, CSS_MAX_LINES, 'style owner is heavy; split only by real UI owner')
    } else if (lineCount >= CSS_OWNER_REVIEW_LINES) {
      addHeavyOwnerReviewSignal(path, 'CSS owner lines', lineCount, CSS_OWNER_REVIEW_LINES, 'style owner is near review threshold')
    }
  }

  if (path.endsWith('.css')) {
    architectureDebtUsage.cssImports += countRegexMatches(contentWithoutComments, CSS_IMPORT_RE)
  }

  architectureDebtUsage.generatedCssReferences += countRegexMatches(contentWithoutComments, GENERATED_CSS_RE)
  architectureDebtUsage.settingsSharedReferences += countRegexMatches(contentWithoutComments, SETTINGS_SHARED_STYLE_RE)
  architectureDebtUsage.settingsSharedReferences += countRegexMatches(contentWithoutComments, SETTINGS_LEGACY_CLASS_RE)

  if (path.endsWith('.vue') || path.endsWith('.css')) {
    for (const match of content.matchAll(STATIC_CLASS_RE)) {
      const legacyTokens = match[1].split(/\s+/).filter(token => LEGACY_CLASS_TOKENS.has(token))
      if (legacyTokens.length > 0) {
        addFailure(path, `legacy class token(s) ${legacyTokens.map(token => `.${token}`).join(', ')} are not allowed; use UI primitives or namespaced classes`)
      }
      const legacyButtonTokens = match[1].split(/\s+/).filter(token => LEGACY_BUTTON_CLASS_TOKENS.has(token))
      if (legacyButtonTokens.length > 0) {
        addFailure(path, `legacy button class token(s) ${legacyButtonTokens.map(token => `.${token}`).join(', ')} are not allowed; use UiButton/UiIconButton props or namespaced component classes`)
      }
      const genericFormTokens = match[1].split(/\s+/).filter(token => GENERIC_FORM_CLASS_TOKENS.has(token))
      if (genericFormTokens.length > 0) {
        addFailure(path, `generic form class token(s) ${genericFormTokens.map(token => `.${token}`).join(', ')} are not allowed; use UI primitives or component-namespaced classes`)
      }
    }

    const legacySelectors = new Set([...content.matchAll(CSS_LEGACY_SELECTOR_RE)].map(match => match[2]))
    if (legacySelectors.size > 0) {
      addFailure(path, `legacy selector(s) ${[...legacySelectors].map(token => `.${token}`).join(', ')} are not allowed; use UI primitives or namespaced classes`)
    }
    const legacyButtonSelectors = new Set([...content.matchAll(CSS_LEGACY_BUTTON_SELECTOR_RE)].map(match => match[2]))
    if (legacyButtonSelectors.size > 0) {
      addFailure(path, `legacy button selector(s) ${[...legacyButtonSelectors].map(token => `.${token}`).join(', ')} are not allowed; use UiButton/UiIconButton props or namespaced component classes`)
    }
    const genericFormSelectors = new Set([...content.matchAll(CSS_GENERIC_FORM_SELECTOR_RE)].map(match => match[2]))
    if (genericFormSelectors.size > 0) {
      addFailure(path, `generic form selector(s) ${[...genericFormSelectors].map(token => `.${token}`).join(', ')} are not allowed; use UI primitives or component-namespaced classes`)
    }

    if (path.endsWith('.css')) {
      const unownedSelectors = new Set([...content.matchAll(CSS_UNOWNED_GLOBAL_SELECTOR_RE)].map(match => match[2]))
      if (unownedSelectors.size > 0) {
        addFailure(path, `unowned global selector(s) ${[...unownedSelectors].map(token => `.${token}`).join(', ')} are not allowed in imported CSS; use the owning component/page namespace`)
      }

      if (!TELEPORT_STYLE_OWNER_ALLOWED_FILES.has(normalizedPath) && CSS_BARE_UI_MODAL_SELECTOR_RE.test(content)) {
        addFailure(path, 'bare .ui-modal__* selectors are not allowed outside BaseModal; scope modal customization behind a business modal class')
      }
    }

    checkUiPrimitiveSelectors(path, normalizedPath, content)
  }

  if (CSS_ID_SELECTOR_RE.test(content)) {
    addFailure(path, 'CSS ID selectors are not allowed in frontend UI source')
  }

  if (IMPORTANT_RE.test(content)) {
    addFailure(path, 'avoid !important; fix specificity or component boundaries instead')
  }

  if (PART_STYLE_IMPORT_RE.test(contentWithoutComments)) {
    addFailure(path, 'mechanical part CSS imports are not allowed; split by named component or style responsibility')
  }

  if (LEGACY_UI_COLOR_TOKEN_RE.test(contentWithoutComments)) {
    addFailure(path, 'legacy --ui-color-* tokens are not allowed; use semantic or visual compatibility tokens')
  }

  if (isTokenFile(normalizedPath)) {
    const definitions = customPropertyDefinitions(content, path)
    const generatedPaletteDefinitions = definitions.filter(token => GENERATED_PALETTE_TOKEN_NAME_RE.test(token))
    if (generatedPaletteDefinitions.length > 0) {
      addFailure(
        path,
        `generated palette token definition(s) ${[...new Set(generatedPaletteDefinitions)].join(', ')} are not allowed; define the final semantic token value directly in src/styles/tokens/*`
      )
    }

    const valueNamedDefinitions = definitions.filter(token => VALUE_NAMED_SEMANTIC_TOKEN_NAME_RE.test(token))
    if (valueNamedDefinitions.length > 0) {
      addFailure(
        path,
        `value-named token definition(s) ${[...new Set(valueNamedDefinitions)].join(', ')} are not allowed; name tokens by role, not raw value fragments`
      )
    }

    const legacyDefinitions = definitions.filter(token => (
      LEGACY_COMPATIBILITY_VARIABLES.has(token) || LEGACY_GLOBAL_VARIABLES.has(token) || LEGACY_SHORT_ALIAS_VARIABLES.has(token)
    ))
    if (legacyDefinitions.length > 0) {
      addFailure(
        path,
        `legacy CSS variable definition(s) ${[...new Set(legacyDefinitions)].join(', ')} are not allowed; use semantic tokens such as --color-text-* and --color-action-*`
      )
    }
  } else {
    const legacyCompatibilityRefs = referencedOrMutatedCustomProperties(contentWithoutComments, LEGACY_COMPATIBILITY_VARIABLES)
    if (legacyCompatibilityRefs.size > 0) {
      addFailure(
        path,
        `legacy compatibility CSS variable reference(s) ${[...legacyCompatibilityRefs].join(', ')} are not allowed; use semantic tokens such as --color-text-* and --color-surface-*`
      )
    }

    const legacyGlobalRefs = referencedOrMutatedCustomProperties(contentWithoutComments, LEGACY_GLOBAL_VARIABLES)
    if (legacyGlobalRefs.size > 0) {
      addFailure(
        path,
        `legacy global CSS variable reference(s) ${[...legacyGlobalRefs].join(', ')} are not allowed; use semantic tokens such as --color-text-* and --color-action-*`
      )
    }

    const legacyShortAliasRefs = referencedOrMutatedLegacyShortAliases(contentWithoutComments)
    if (legacyShortAliasRefs.size > 0) {
      addFailure(
        path,
        `legacy short CSS variable reference(s) ${[...legacyShortAliasRefs].join(', ')} are not allowed; use semantic tokens such as --color-surface-*, --color-action-*, or --color-status-*`
      )
    }
  }

  const generatedOwnerTokenRefs = matchedTokens(contentWithoutComments, GENERATED_OWNER_TOKEN_RE)
  if (generatedOwnerTokenRefs.size > 0) {
    addFailure(
      path,
      `generated owner token reference(s) ${[...generatedOwnerTokenRefs].join(', ')} are not allowed; use named semantic or owner-scoped tokens`
    )
  }

  const valueNamedTokenCount = countRegexMatches(contentWithoutComments, VALUE_NAMED_TOKEN_RE)
  architectureDebtUsage.valueNamedTokenReferences += valueNamedTokenCount
  if (valueNamedTokenCount > 0) {
    valueNamedTokenCandidates.set(normalizedPath, valueNamedTokenCount)
  }

  if (!isTokenFile(normalizedPath)) {
    architectureDebtUsage.businessVisualColorTokenReferences += countRegexMatches(contentWithoutComments, VISUAL_COMPAT_COLOR_TOKEN_RE)
    architectureDebtUsage.colorTokenReferencesOutsideTokens += countRegexMatches(contentWithoutComments, COMPAT_COLOR_TOKEN_RE)
    const paletteRefs = matchedTokens(contentWithoutComments, PALETTE_TOKEN_REFERENCE_RE)
    if (paletteRefs.size > 0) {
      addFailure(
        path,
        `private palette token reference(s) ${[...paletteRefs].join(', ')} are not allowed outside src/styles/tokens/*; use semantic or owner tokens`
      )
    }
  }

  if (LEGACY_RADIUS_TOKEN_RE.test(contentWithoutComments)) {
    addFailure(path, 'legacy --border-radius-* tokens are not allowed; use --radius-* tokens')
  }

  architectureDebtUsage.legacyRadiusCompatTokenReferences += countRegexMatches(contentWithoutComments, LEGACY_RADIUS_COMPAT_TOKEN_RE)

  if (!isTokenFile(normalizedPath) && BARE_Z_INDEX_RE.test(contentWithoutComments)) {
    addFailure(path, 'bare z-index numbers are not allowed; use var(--z-*) tokens')
  }

  if (!isTokenFile(normalizedPath) && Z_INDEX_NUMERIC_FALLBACK_RE.test(contentWithoutComments)) {
    addFailure(path, 'numeric z-index fallbacks are not allowed; define the required --z-* token in src/styles/tokens/*')
  }

  if (!isTokenFile(normalizedPath) && BARE_MEDIA_BREAKPOINT_RE.test(contentWithoutComments)) {
    addFailure(path, 'bare media breakpoint numbers are not allowed; use @custom-media breakpoint tokens from src/styles/tokens/foundation.css')
  }

  if (contentWithoutComments.includes(':deep(')) {
    addFailure(path, ':deep() is not allowed in UI source; move ownership to the child component or a primitive')
  }

  architectureDebtUsage.globalSelectors += countRegexMatches(contentWithoutComments, GLOBAL_SELECTOR_RE)

  if (!isTokenFile(normalizedPath) && LEGACY_VARIABLE_DEFINITION_RE.test(contentWithoutComments)) {
    addFailure(path, 'legacy compatibility CSS variables may not be defined in business UI; create a semantic token in src/styles/tokens/* instead')
  }

  checkHardcodedColors(path, normalizedPath, content)

  if (LEGACY_IMPORT_RE.test(content)) {
    addFailure(path, 'legacy-components.css must not be imported or referenced')
  }

  if (path.endsWith('BaseModal.vue') && BASE_MODAL_LEGACY_RE.test(content)) {
    addFailure(path, 'BaseModal must expose only ui-modal__* structure classes')
  }

  if (path.endsWith('.vue')) {
    for (const match of content.matchAll(BASE_MODAL_CUSTOM_CLASS_RE)) {
      const customClasses = match[1].split(/\s+/).filter(Boolean)
      for (const customClass of customClasses) {
        const teleportedDeepSelectorRe = new RegExp(`:{1,2}deep\\(\\.${escapeRegExp(customClass)}(?:[\\s.)#>:]|$)`)
        if (teleportedDeepSelectorRe.test(content)) {
          addFailure(
            path,
            `BaseModal custom class .${customClass} is teleported; use :global(...) instead of scoped :deep(...) for modal container/body styling`
          )
        }
      }
    }
  }
}

function checkScriptFile(path) {
  const content = readFileSync(path, 'utf8')
  const normalizedPath = normalizePath(path)
  const contentWithoutComments = stripCssComments(content)

  checkLocalBuildArtifact(path, normalizedPath)
  checkTestPrimitiveInternalSelectors(path, normalizedPath, contentWithoutComments)
  checkOldImplementationMindset(path, normalizedPath, content)
  checkComposableSourceHygiene(path, normalizedPath, content, contentWithoutComments)
  checkTypeSourceHygiene(path, normalizedPath, contentWithoutComments)
  checkUtilSourceHygiene(path, normalizedPath, contentWithoutComments)
  checkRelativeExports(path, contentWithoutComments)
  checkTypesBarrelExports(path, normalizedPath, contentWithoutComments)
  checkFrontendSchemaCompatibility(path, normalizedPath, contentWithoutComments)

  architectureDebtUsage.generatedCssReferences += countRegexMatches(contentWithoutComments, GENERATED_CSS_RE)
  architectureDebtUsage.settingsSharedReferences += countRegexMatches(contentWithoutComments, SETTINGS_SHARED_STYLE_RE)
  architectureDebtUsage.settingsSharedReferences += countRegexMatches(contentWithoutComments, SETTINGS_LEGACY_CLASS_RE)
  const valueNamedTokenCount = countRegexMatches(contentWithoutComments, VALUE_NAMED_TOKEN_RE)
  architectureDebtUsage.valueNamedTokenReferences += valueNamedTokenCount
  if (valueNamedTokenCount > 0) {
    valueNamedTokenCandidates.set(normalizedPath, valueNamedTokenCount)
  }

  if (!isTokenFile(normalizedPath)) {
    architectureDebtUsage.colorTokenReferencesOutsideTokens += countRegexMatches(contentWithoutComments, COMPAT_COLOR_TOKEN_RE)
    const paletteRefs = matchedTokens(contentWithoutComments, PALETTE_TOKEN_REFERENCE_RE)
    if (paletteRefs.size > 0) {
      addFailure(
        path,
        `private palette token reference(s) ${[...paletteRefs].join(', ')} are not allowed outside src/styles/tokens/*; use semantic or owner tokens`
      )
    }
  }

  const legacyCompatibilityRefs = referencedOrMutatedCustomProperties(contentWithoutComments, LEGACY_COMPATIBILITY_VARIABLES)
  if (legacyCompatibilityRefs.size > 0) {
    addFailure(
      path,
      `legacy compatibility CSS variable reference(s) ${[...legacyCompatibilityRefs].join(', ')} are not allowed; use semantic tokens such as --color-text-* and --color-surface-*`
    )
  }

  const legacyGlobalRefs = referencedOrMutatedCustomProperties(contentWithoutComments, LEGACY_GLOBAL_VARIABLES)
  if (legacyGlobalRefs.size > 0) {
    addFailure(
      path,
      `legacy global CSS variable reference(s) ${[...legacyGlobalRefs].join(', ')} are not allowed; use semantic tokens such as --color-text-* and --color-action-*`
    )
  }

  const legacyShortAliasRefs = referencedOrMutatedLegacyShortAliases(contentWithoutComments)
  if (legacyShortAliasRefs.size > 0) {
    addFailure(
      path,
      `legacy short CSS variable reference(s) ${[...legacyShortAliasRefs].join(', ')} are not allowed; use semantic tokens such as --color-surface-*, --color-action-*, or --color-status-*`
    )
  }

  const generatedOwnerTokenRefs = matchedTokens(contentWithoutComments, GENERATED_OWNER_TOKEN_RE)
  if (generatedOwnerTokenRefs.size > 0) {
    addFailure(
      path,
      `generated owner token reference(s) ${[...generatedOwnerTokenRefs].join(', ')} are not allowed; use named semantic or owner-scoped tokens`
    )
  }

  if (!RAW_FORM_CONTROL_ALLOWED_FILES.has(normalizedPath)) {
    architectureDebtUsage.rawInputs += countRegexMatches(contentWithoutComments, RAW_INPUT_RE)
    architectureDebtUsage.rawTextareas += countRegexMatches(contentWithoutComments, RAW_TEXTAREA_RE)
    architectureDebtUsage.rawSelects += countRegexMatches(contentWithoutComments, RAW_SELECT_RE)
  }
  architectureDebtUsage.toolButtonVariants += countRegexMatches(contentWithoutComments, TOOL_BUTTON_VARIANT_RE)

  if (UNSTYLED_BUTTON_VARIANT_RE.test(content)) {
    addFailure(path, 'variant="unstyled" is not allowed in UI tests or business source; use a named UiButton variant or UiIconButton')
  }

  if (BARE_Z_INDEX_RE.test(contentWithoutComments)) {
    addFailure(path, 'bare z-index numbers in inline style strings are not allowed; use var(--z-*) tokens')
  }

  if (Z_INDEX_NUMERIC_FALLBACK_RE.test(contentWithoutComments)) {
    addFailure(path, 'numeric z-index fallbacks in inline style strings are not allowed; define the required --z-* token in src/styles/tokens/*')
  }
}

function checkRepositoryLocalArtifacts() {
  if (SOURCE_FIXTURE || SKIP_SOURCE_SCAN) {
    return
  }
  for (const artifact of ['build_output.txt', 'vite-dev.log']) {
    const path = join(process.cwd(), artifact)
    if (existsSync(path)) {
      checkScriptFile(path)
    }
  }
}

checkTokenDependencyArchitecture(TOKEN_FILES)
checkCriticalVisualCoverage()

if (SOURCE_FIXTURE) {
  if (SOURCE_FIXTURE.endsWith('.vue') || SOURCE_FIXTURE.endsWith('.css')) {
    checkFile(SOURCE_FIXTURE)
  } else {
    checkScriptFile(SOURCE_FIXTURE)
  }
} else if (!SKIP_SOURCE_SCAN) {
  for (const root of ROOTS) {
    scanPath(root)
  }
  checkRepositoryLocalArtifacts()
}

for (const [key, budget] of Object.entries(ARCHITECTURE_DEBT_BUDGETS)) {
  const usage = architectureDebtUsage[key]
  if (usage > budget) {
    failures.push(`UI architecture debt budget exceeded for ${key}: ${usage} > ${budget}; migrate to semantic tokens/primitives or lower the registered budget after cleanup`)
  }
}

if (failures.length > 0) {
  console.error('UI architecture check failed:')
  for (const failure of failures) {
    console.error(`- ${failure}`)
  }
  process.exit(1)
}

if (IS_AUDIT && valueNamedTokenCandidates.size > 0) {
  console.warn('UI architecture value-named token migration candidates:')
  for (const [file, count] of [...valueNamedTokenCandidates.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12)) {
    console.warn(`- ${file}: ${count} references`)
  }
}

if (IS_AUDIT) {
  console.warn('UI architecture audit summary:')
  console.warn(`- token files: ${TOKEN_FILES.length}`)
  console.warn(`- :root tokens: ${tokenArchitectureStats.rootTokens}`)
  console.warn(`- :root token dependencies: ${tokenArchitectureStats.rootDependencies}`)
  console.warn(`- critical visual states covered: ${visualCoverageEntries.length}`)
  console.warn(`- heavy owner review signals: ${heavyOwnerReviewSignals.length}`)
  for (const signal of heavyOwnerReviewSignals.slice(0, 12)) {
    console.warn(`  - ${signal}`)
  }
}

console.log('UI architecture check passed')
