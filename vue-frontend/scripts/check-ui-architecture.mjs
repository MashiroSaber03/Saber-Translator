import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join, relative, resolve } from 'node:path'

const ROOTS = ['src/components', 'src/views', 'src/styles', 'src/composables', 'tests']
const IS_AUDIT = process.argv.includes('--audit')
const SKIP_SOURCE_SCAN = process.argv.includes('--skip-source-scan')
const TOKENS_FIXTURE_INDEX = process.argv.indexOf('--tokens-fixture')
const SOURCE_FIXTURE_INDEX = process.argv.indexOf('--source-fixture')
const SOURCE_FIXTURE = SOURCE_FIXTURE_INDEX >= 0
  ? resolve(process.cwd(), process.argv[SOURCE_FIXTURE_INDEX + 1] || '')
  : null
const TOKEN_FILE_ORDER = [
  'src/styles/tokens/palette.css',
  'src/styles/tokens/semantic.css',
  'src/styles/tokens/component.css',
  'src/styles/tokens/domain.css',
]
const TOKEN_FILES = TOKENS_FIXTURE_INDEX >= 0
  ? [resolve(process.cwd(), process.argv[TOKENS_FIXTURE_INDEX + 1] || '')]
  : TOKEN_FILE_ORDER.map(file => join(process.cwd(), file))
const SFC_MAX_LINES = 1800
const SFC_REVIEW_LINES = 1400
const SFC_MAX_STYLE_LINES = 1200
const CSS_MAX_LINES = 1200
const CSS_REVIEW_LINES = 900
const CSS_OWNER_REVIEW_LINES = 800
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
const LAYOUT_BYPASS_RE = /position\s*:\s*fixed|(?:min-|max-)?height\s*:\s*calc\(100vh|margin-(?:left|right)\s*:\s*(?:240|340)px/
const BASE_MODAL_LEGACY_RE = /\bmodal-(?:overlay|container|header|title|close-btn|body|footer|small|medium|large|full)\b/
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
const CSS_UI_PRIMITIVE_SELECTOR_RE = /(^|[^A-Za-z0-9_-])\.(ui-form-field|ui-input|ui-select|ui-textarea)(?![A-Za-z0-9_-])/g
const CSS_BARE_UI_MODAL_SELECTOR_RE = /^\s*\.ui-modal__/m
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
const GENERATED_OWNER_TOKEN_RE = /--color-[a-z0-9-]+-(?:surface|text|border|shadow|accent)-\d{3}/g
const OLD_IMPLEMENTATION_MINDSET_RE = /保持既有|保持当前视觉|复刻原版|复刻旧版|复刻自|整理自既有|完整样式(?:\s*-\s*从[^*\n\r]+)?|从\s+[^*\n\r]+\.css\s+迁移|迁移自\s+[^*\n\r]+|旧版[^*\n\r]*|原版[^*\n\r]*|\b(?:bookshelf|edit_mode|main|events)\.js\b|\b(?:global|style|reader|manga-insight)\.css\b|迁移自旧 CSS|已迁移到 global\.css|Source:\s*[^*]*\.styles\.css|legacy UI|legacy CSS/gi
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
  'src/components/',
  'src/views/',
  'src/composables/',
  'src/styles/',
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

const LARGE_SFC_DECISIONS = new Map([
  ['src/components/insight/studio/CharacterStudioPreview.vue', 'split by workspace areas: chat/session/agent/prompt preview/modal concerns are independent UI owners'],
  ['src/components/bookshelf/BookDetailModal.vue', 'split by modal sections when editing this flow: summary/chapters/tags/danger actions are separate UI regions'],
  ['src/components/insight/studio/CharacterStudioEditor.vue', 'split by editor panels when the next studio UI change lands: identity/tabs/lorebook/diagnostics/footer are separate'],
  ['src/components/settings/TranslationSettings.vue', 'extract sections/composables only; provider state is cohesive and should remain centralized'],
  ['src/components/settings/OcrSettings.vue', 'extract sections/composables only; OCR provider switching and model fetching share state'],
  ['src/components/translate/WebImportModal.vue', 'split by modal workflow: extract bar/settings/results/logs/footer can stand alone'],
  ['src/components/insight/QAPanel.vue', 'keep chat flow cohesive; extract note modal/rebuild controls when touched'],
  ['src/views/InsightView.vue', 'split shell regions after layout migration; page still owns analysis orchestration'],
  ['src/components/insight/PagesTree.vue', 'keep cohesive: recursion and selection state are tightly coupled'],
  ['src/components/insight/AnalysisProgress.vue', 'extract API state composable only; UI is one workflow'],
  ['src/components/edit/BubbleEditor.vue', 'keep cohesive: selected bubble state and editing controls must be read together'],
  ['src/components/insight/ContinuationPanel.vue', 'keep cohesive tab orchestrator; sub-flows already live in dialogs'],
  ['src/components/insight/TimelinePanel.vue', 'keep cohesive timeline surface after style ownership merge; filters, summaries, entities, and event list share one timeline state'],
  ['src/views/CharacterStudioView.vue', 'keep until layout shell migration leaves clear subcomponents'],
  ['src/components/insight/PageDetail.vue', 'keep cohesive read/edit panel'],
  ['src/components/edit/BubbleOverlay.vue', 'keep cohesive: pointer geometry and overlay rendering are regression-prone'],
])

const LARGE_CSS_OWNER_DECISIONS = new Map([
  ['src/components/insight/NotesPanel.styles.css', 'split only with Notes modal extraction; list/toolbar/editor modal/empty state can become real owners'],
  ['src/components/edit/BubbleEditor.styles.css', 'keep while BubbleEditor remains one cohesive editor form'],
  ['src/components/insight/QAPanel.styles.css', 'split only with note modal/rebuild extraction; chat surface remains cohesive'],
  ['src/components/edit/EditToolbar.styles.css', 'keep for now: toolbar controls are visually cohesive and tiny style files would reduce clarity'],
  ['src/components/insight/studio/CharacterStudioEditor.styles.css', 'accepted studio editor owner; split leaf panels only with a concrete ownership gain'],
  ['src/components/settings/PluginAgentModal.styles.css', 'keep for now: three-column modal styling is one cohesive owner'],
])

const LAYOUT_BYPASS_DECISIONS = new Map([
  ['src/views/BookshelfView.vue', 'page content still owns historical header-height offset; migrate to AppShell content sizing when bookshelf layout is next touched'],
  ['src/views/InsightView.vue', 'Insight page owns the analysis viewport until the multi-pane shell fully absorbs mobile drawers'],
  ['src/views/InsightView.styles.css', 'mobile sidebar overlay remains page-owned until SidebarLayout overlay mode owns analysis mobile panes'],
  ['src/views/ReaderView.vue', 'reader header is a distinct immersive mode; keep until AppShell reader variant owns fixed chrome'],
  ['src/views/CharacterStudioView.vue', 'studio page keeps viewport editor shell until SidebarLayout studio mode owns pane heights'],
  ['src/views/TranslateView.vue', 'Translate page owns workspace side gutters until SidebarLayout fixed-sidebar mode covers every edit-state constraint'],
  ['src/views/TranslateView.styles.css', 'translate workspace still has legacy sidebar margin offsets; migrate after SidebarLayout fixed-sidebar mode is fully verified'],
  ['src/components/common/AppHeader.vue', 'AppHeader owns its mobile fixed navigation menu'],
  ['src/components/common/AppHeader.styles.css', 'AppHeader owns its mobile fixed navigation menu'],
  ['src/components/common/BaseModal.vue', 'BaseModal owns Teleport overlay positioning'],
  ['src/components/common/CustomSelect.vue', 'CustomSelect owns Teleport dropdown positioning'],
  ['src/components/common/ToastNotification.vue', 'Toast owns fixed notification positioning'],
  ['src/components/reader/ReaderCanvas.vue', 'reader canvas owns viewport fit for immersive reading mode'],
  ['src/components/reader/ReaderControls.vue', 'ReaderControls owns immersive fixed controls'],
  ['src/components/reader/ReaderControls.styles.css', 'reader controls own fixed immersive controls until reader shell is thickened'],
  ['src/components/translate/SettingsSidebar.vue', 'SettingsSidebar owns the fixed translated-page pane while SidebarLayout hosts the page shell'],
  ['src/components/translate/SettingsSidebar.shell.styles.css', 'settings sidebar owns its fixed translated-page pane until SidebarLayout can express the exact chrome contract'],
  ['src/components/translate/ThumbnailSidebar.vue', 'ThumbnailSidebar owns the fixed translated-page pane while SidebarLayout hosts the page shell'],
  ['src/components/translate/ThumbnailSidebar.styles.css', 'thumbnail sidebar owns its fixed translated-page pane until SidebarLayout can express the exact chrome contract'],
  ['src/components/ui/AppShell.vue', 'AppShell owns page viewport, chrome, and header-offset algorithms'],
  ['src/components/ui/SidebarLayout.vue', 'SidebarLayout owns fixed/sticky/overlay sidebar algorithms for page migration'],
  ['src/components/edit/EditToolbar.vue', 'EditToolbar owns fixed mobile toolbar behavior'],
  ['src/components/edit/EditToolbar.styles.css', 'edit toolbar owns fixed mobile toolbar behavior'],
  ['src/components/edit/EditWorkspace.vue', 'EditWorkspace is a full-screen editor shell by design'],
  ['src/components/edit/EditWorkspace.canvas.styles.css', 'edit canvas owns floating mini-map/compare controls'],
  ['src/components/edit/EditWorkspace.shell.styles.css', 'edit workspace is a full-screen editor shell by design'],
  ['src/components/settings/PluginManager.vue', 'PluginManager owns scoped loading overlay positioning'],
  ['src/components/settings/PluginManager.styles.css', 'plugin manager owns scoped loading overlay positioning'],
  ['src/components/insight/PageDetail.vue', 'PageDetail owns its scoped image preview overlay'],
  ['src/components/insight/PageDetail.styles.css', 'page detail image preview overlay is scoped to the panel'],
  ['src/components/insight/TimelinePanel.vue', 'TimelinePanel owns internal event-list scrolling'],
  ['src/components/insight/TimelinePanel.timeline.styles.css', 'timeline panel owns internal list max-height; page shell should not manage event list scrolling'],
])

const failures = []
const cssReviewCandidates = []
const sfcReviewCandidates = []
const layoutBypassCandidates = []
const valueNamedTokenCandidates = new Map()
const tokenArchitectureStats = {
  rootTokens: 0,
  bodyTokens: 0,
  rootDependencies: 0,
}
const visualCoverageEntries = []
const architectureDebtUsage = Object.fromEntries(
  Object.keys(ARCHITECTURE_DEBT_BUDGETS).map(key => [key, 0])
)

function walk(dir) {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry)
    const stat = statSync(path)
    if (stat.isDirectory()) {
      walk(path)
      continue
    }
    if (path.endsWith('.vue') || path.endsWith('.css')) {
      checkFile(path)
      continue
    }
    if (/\.(?:js|jsx|ts|tsx)$/.test(path)) {
      checkScriptFile(path)
    }
  }
}

function addFailure(path, message) {
  failures.push(`${relative(process.cwd(), path)}: ${message}`)
}

function normalizePath(path) {
  return relative(process.cwd(), path).replaceAll('\\', '/')
}

function isTokenFile(normalizedPath) {
  return TOKEN_FILE_ORDER.includes(normalizedPath)
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

  const unownedTokens = definitions.filter(token => !token.startsWith(pagePrefix))
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

function checkHardcodedColors(path, normalizedPath, content) {
  if (COLOR_LITERAL_ALLOWED_FILES.has(normalizedPath)) {
    return
  }

  const styleContents = extractStyleContents(content, path)
  const colorMatches = new Set()
  for (const styleContent of styleContents) {
    for (const match of stripCssComments(styleContent).matchAll(HARDCODED_COLOR_RE)) {
      colorMatches.add(match[0])
    }
  }

  if (colorMatches.size > 0) {
    addFailure(path, `style color literal(s) ${[...colorMatches].slice(0, 6).join(', ')} are not allowed; add/use tokens from src/styles/tokens/*`)
  }
}

function checkFile(path) {
  const content = readFileSync(path, 'utf8')
  const normalizedPath = normalizePath(path)
  const contentWithoutComments = stripCssComments(content)

  checkOldImplementationMindset(path, normalizedPath, content)
  checkCustomPropertyOwnership(path, normalizedPath, content)

  if (GENERATED_CSS_RE.test(normalizedPath)) {
    addFailure(path, 'generated CSS files are not allowed; use co-located named .styles.css or component scoped styles')
  }

  if (HORIZONTAL_STYLE_SLICE_RE.test(normalizedPath)) {
    addFailure(path, 'horizontal style slice names (.base/.layout/.panels/.responsive.styles.css) are not allowed; merge small owners or split by real UI responsibility')
  }

  if (LAYOUT_BYPASS_RE.test(contentWithoutComments) && !isTokenFile(normalizedPath)) {
    const decision = LAYOUT_BYPASS_DECISIONS.get(normalizedPath)
    if (!decision) {
      addFailure(path, 'layout bypass detected (fixed viewport, 100vh calc, or legacy sidebar margin) without a maintenance decision; move the layout algorithm to AppShell/SidebarLayout or register why this owner must keep it')
    } else {
      layoutBypassCandidates.push(`${normalizedPath} (${decision})`)
    }
  }

  if (path.endsWith('.vue') && VUE_STYLE_RE.test(content)) {
    addFailure(path, 'component styles must be scoped; use :global(...) only for explicit Teleport/BaseModal reach-through')
  }

  if (path.endsWith('.vue')) {
    const lineCount = content.split(/\r?\n/).length
    const styleLineCount = [...content.matchAll(/<style\b[\s\S]*?<\/style>/g)]
      .map(match => match[0].split(/\r?\n/).length)
      .reduce((sum, count) => sum + count, 0)

    if (lineCount > SFC_MAX_LINES) {
      addFailure(path, `SFC has ${lineCount} lines; split UI sections or move isolated UI behavior into composables`)
    } else if (lineCount >= SFC_REVIEW_LINES) {
      const decision = LARGE_SFC_DECISIONS.get(normalizedPath)
      if (!decision) {
        addFailure(path, `SFC has ${lineCount} lines and no maintenance decision; add a real owner-boundary decision before keeping or splitting it`)
      } else {
        sfcReviewCandidates.push(`${normalizedPath}: ${lineCount} lines (${decision})`)
      }
    }
    if (styleLineCount > SFC_MAX_STYLE_LINES) {
      addFailure(path, `SFC has ${styleLineCount} style lines; move repeated UI styling into primitives or split the component`)
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
    const lineCount = content.split(/\r?\n/).length
    if (lineCount > CSS_MAX_LINES) {
      addFailure(path, `CSS file has ${lineCount} lines; split by component boundary or move repeated rules into primitives`)
    } else if (lineCount >= CSS_OWNER_REVIEW_LINES) {
      const decision = LARGE_CSS_OWNER_DECISIONS.get(normalizedPath)
      if (!decision) {
        addFailure(path, `CSS owner has ${lineCount} lines and no maintenance decision; do not split by line count, register the owner decision or extract a real subcomponent`)
      } else if (lineCount >= CSS_REVIEW_LINES) {
        cssReviewCandidates.push(`${normalizedPath}: ${lineCount} lines (${decision})`)
      }
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

      if (!normalizedPath.startsWith(UI_PRIMITIVE_STYLE_ALLOWED_PREFIX)) {
        const primitiveSelectors = new Set([...content.matchAll(CSS_UI_PRIMITIVE_SELECTOR_RE)].map(match => match[2]))
        if (primitiveSelectors.size > 0) {
          addFailure(path, `UI primitive selector(s) ${[...primitiveSelectors].map(token => `.${token}`).join(', ')} are not allowed in business CSS; use primitive props/classes or a business-owned class`)
        }
      }

      if (!TELEPORT_STYLE_OWNER_ALLOWED_FILES.has(normalizedPath) && CSS_BARE_UI_MODAL_SELECTOR_RE.test(content)) {
        addFailure(path, 'bare .ui-modal__* selectors are not allowed outside BaseModal; scope modal customization behind a business modal class')
      }
    }
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
    addFailure(path, 'bare media breakpoint numbers are not allowed; use @custom-media breakpoint tokens from src/styles/tokens/palette.css')
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

  checkOldImplementationMindset(path, normalizedPath, content)

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

checkTokenDependencyArchitecture(TOKEN_FILES)
checkCriticalVisualCoverage()

if (SOURCE_FIXTURE) {
  checkFile(SOURCE_FIXTURE)
} else if (!SKIP_SOURCE_SCAN) {
  for (const root of ROOTS) {
    walk(root)
  }
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
}

console.log('UI architecture check passed')
