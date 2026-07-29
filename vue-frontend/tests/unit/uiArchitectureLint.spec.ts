import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { spawnSync } from 'node:child_process'
import { describe, expect, it } from 'vitest'

const frontendRoot = resolve(__dirname, '..', '..')
const legacyBgColor = '--bg' + '-color'
const legacyTextColor = '--text' + '-color'
const legacyCardBg = '--card' + '-bg'
const legacyBorderColor = '--border' + '-color'
const legacyTextPrimary = '--text' + '-primary'
const legacyColorPrimary = '--color' + '-primary'
const legacyShortBgSecondary = '--bg' + '-secondary'
const legacyShortBgActive = '--bg' + '-active'
const legacyShortPrimary = '--primary'
const legacyReaderBgColor = '--reader' + '-bg-color'
const generatedOwnerToken = '--color-' + 'book-card-surface-001'
const generatedPaletteToken = '--palette-' + 'surface-001'
const generatedPaletteVariantToken = '--palette-' + 'blue-400-clear'
const privatePaletteToken = '--palette-' + 'blue-500'
const paletteSurfaceEditorToken = '--palette-' + 'surface-editor'
const valueNamedSemanticToken = '--color-surface-' + 'basefff'
const migrationMindsetKeepExisting = '保持' + '既有'
const migrationMindsetReplica = '复刻' + '原' + '版'
const migrationMindsetStyleSource = 'Source' + ': Panel' + '.styles.css'
const migrationMindsetCompleteStyles = '完整' + '样式 - ' + '从 reader' + '.css 迁' + '移'
const migrationMindsetOldVersion = '旧' + '版 handleBubbleMouseDown'
const migrationMindsetMigratedFrom = '迁移' + '自 main' + '.js'
const migrationMindsetOriginalReference = '对应' + '原' + '版 edit_mode' + '.js'
const migrationMindsetOriginalCore = '原' + '版 edit_mode' + '.js'
const migrationMindsetOriginalJsReference = '对应' + '原 image_viewer' + '.js'
const migrationMindsetOldFileName = '当前行为 bookshelf' + '.js'
const migrationMindsetOldFileNameCore = 'bookshelf' + '.js'
const migrationMindsetSimplifiedDesign = '【简化' + '设计】'
const composableHistoryExtracted = '从 TranslateView 提' + '取'
const composableHistorySimplified = '简化' + '设计'
const composableExplicitAny = 'as ' + 'any'
const relativeExportStatement = 'export * fr' + 'om'
const missingRelativeExport = './missing' + 'Type'
const characterStudioTypeExport = './character' + 'Studio'
const webImportTypeExport = './web' + 'Import'
const legacyProviderCustomOpenAi = 'custom' + '_openai'
const legacyProviderCustomOpenAiVision = 'custom' + '_openai_vision'
const legacyIdsField = 'legacy' + 'Ids'
const legacyStorageKey = 'LEGACY' + '_STORAGE_KEY'
const oldStripMirrorHelper = 'strip' + 'LegacyOpenAiMirrorFields'
const oldSyncMirrorHelper = 'sync' + 'LegacyOpenAiMirrorFields'
const oldCoerceRetryHelper = 'coerce' + 'LegacyRetryValue'
const oldSchemaMergeHelper = 'deep' + 'Merge'
const staleTestFeatureLabel = 'Feature' + ': frontend-behavior'
const staleTestPropertyLabel = 'Property ' + '42'
const staleTestRequirementLabel = 'Validates' + ': Requirements'
const threshold48pxField = 'threshold' + '48px'
const thresholdMangaOcrField = 'threshold' + 'MangaOcr'
const thresholdPaddleOcrField = 'threshold' + 'PaddleOcr'
const oldIsJsonModeField = 'is' + 'JsonMode'
const oldForceJsonField = 'force' + 'Json'
const oldMaxRetriesField = 'max' + 'Retries'
const webImportSchemaVersionField = 'webImportSettings' + 'SchemaVersion'
const partialWebImportSettings = 'Partial<' + 'WebImportSettings>'
const partialWebImportProviderConfigs = 'Partial<' + 'WebImportProviderConfigs>'
const partialWebImportSettingsPayload = 'Partial<' + 'WebImportSettingsPayload>'
const primitiveButtonInternalSelector = '.ui-button' + '--primary'
const primitiveModalBodySelector = '.ui-modal' + '__body'
const primitiveFieldVariantSelector = '.ui-field' + '--settings'
const primitiveFieldLabelActionSelector = '.ui-field' + '__label-actions'
const componentPrivateDomainToken = '--character-studio-preview-shell-surface-base'
const genericComponentDomainToken = '--book-card-surface-base'
const generatedInsightVariantToken = '--insight-view-accent-variant-012'
const insightSharedThemeToken = '--insight-surface-page'
const insightLegacyBgToken = '--insight-bg-primary'
const insightLegacyPrimaryToken = '--insight-primary-dark'
const insightActionPrimaryToken = '--insight-action-primary'
const editDomainSemanticToken = '--color-edit-shell-start'
const studioDomainSemanticToken = '--color-text-studio-strong'
const semanticGrayScaleToken = '--color-gray-100'
const semanticEditorSurfaceToken = '--color-surface-editor-original'
const semanticAccentPurpleToken = '--color-accent-purple'
const semanticGradientStartToken = '--color-surface-brand-gradient-start'
const semanticSurfacePlainToken = '--color-surface-plain'
const semanticTextPrimaryStrongToken = '--color-text-primary-strong'
const semanticWarningTintToken = '--color-surface-warning-tint'
const vagueComponentToken = '--base-modal-surface-base'
const roleComponentToken = '--base-modal-overlay-background'
const domainTokenLimit = 50
const studioOwnerToken = '--studio-border-default'
const unusedOwnerToken = '--settings-sidebar-panel-background'

function runUiArchitectureTokenFixture(tokensCss: string, tokenPath = 'src/styles/tokens/domain.css') {
  const fixtureDir = mkdtempSync(join(tmpdir(), 'ui-architecture-tokens-'))
  const fixturePath = join(fixtureDir, 'tokens.css')
  writeFileSync(fixturePath, tokensCss)

  return spawnSync(
    process.execPath,
    [
      'scripts/check-ui-architecture.mjs',
      '--tokens-fixture',
      fixturePath,
      '--tokens-fixture-path',
      tokenPath,
      '--skip-source-scan',
    ],
    {
      cwd: frontendRoot,
      encoding: 'utf8',
    }
  )
}

function runUiArchitectureSourceFixture(relativePath: string, content: string) {
  const fixtureDir = mkdtempSync(join(tmpdir(), 'ui-architecture-source-'))
  const fixturePath = join(fixtureDir, relativePath)
  mkdirSync(join(fixturePath, '..'), { recursive: true })
  writeFileSync(fixturePath, content)

  return spawnSync(
    process.execPath,
    [
      'scripts/check-ui-architecture.mjs',
      '--source-fixture',
      fixturePath,
      '--source-fixture-path',
      relativePath,
    ],
    {
      cwd: frontendRoot,
      encoding: 'utf8',
    }
  )
}

function runUiArchitectureTokenUsageFixture(tokensCss: string, relativePath: string, content: string) {
  const fixtureDir = mkdtempSync(join(tmpdir(), 'ui-architecture-token-usage-'))
  const tokenFixturePath = join(fixtureDir, 'tokens.css')
  const sourceFixturePath = join(fixtureDir, relativePath)
  mkdirSync(join(sourceFixturePath, '..'), { recursive: true })
  writeFileSync(tokenFixturePath, tokensCss)
  writeFileSync(sourceFixturePath, content)

  return spawnSync(
    process.execPath,
    [
      'scripts/check-ui-architecture.mjs',
      '--tokens-fixture',
      tokenFixturePath,
      '--tokens-fixture-path',
      'src/styles/tokens/semantic.css',
      '--source-fixture',
      sourceFixturePath,
      '--source-fixture-path',
      relativePath,
    ],
    {
      cwd: frontendRoot,
      encoding: 'utf8',
    }
  )
}

function runUiArchitectureAudit() {
  return spawnSync(
    process.execPath,
    [
      'scripts/check-ui-architecture.mjs',
      '--audit',
    ],
    {
      cwd: frontendRoot,
      encoding: 'utf8',
    }
  )
}

describe('UI architecture token dependency lint', () => {
  it('rejects component-private tokens in domain token files', () => {
    const result = runUiArchitectureTokenFixture(`
      :root {
        ${componentPrivateDomainToken}: var(--color-surface-panel);
        --color-surface-panel: #fff;
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('component-private domain token definition(s)')
    expect(result.stderr).toContain(componentPrivateDomainToken)
  })

  it('rejects component owner tokens in domain token files', () => {
    const result = runUiArchitectureTokenFixture(`
      :root {
        ${genericComponentDomainToken}: rgba(0, 0, 0, .6);
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('component-private domain token definition(s)')
    expect(result.stderr).toContain(genericComponentDomainToken)
  })

  it('allows insight shared theme tokens in domain token files', () => {
    const result = runUiArchitectureTokenFixture(`
      :root {
        ${insightSharedThemeToken}: #fff;
      }
    `)

    expect(result.status).toBe(0)
    expect(result.stdout).toContain('UI architecture check passed')
  })

  it('rejects legacy-style insight domain aliases in domain token files', () => {
    const result = runUiArchitectureTokenFixture(`
      :root {
        ${insightLegacyBgToken}: #fff;
        ${insightLegacyPrimaryToken}: #4f46e5;
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('legacy-style insight domain alias token definition(s)')
    expect(result.stderr).toContain(insightLegacyBgToken)
    expect(result.stderr).toContain(insightLegacyPrimaryToken)
  })

  it('allows role-named insight action tokens in domain token files', () => {
    const result = runUiArchitectureTokenFixture(`
      :root {
        ${insightActionPrimaryToken}: #6366f1;
      }
    `)

    expect(result.status).toBe(0)
    expect(result.stdout).toContain('UI architecture check passed')
  })

  it('rejects generated insight variant token names in domain token files', () => {
    const result = runUiArchitectureTokenFixture(`
      :root {
        ${generatedInsightVariantToken}: #f6f8fb;
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('generated domain token definition(s)')
    expect(result.stderr).toContain(generatedInsightVariantToken)
  })

  it('rejects domain token files over the final owner budget', () => {
    const tokenDefinitions = Array.from(
      { length: domainTokenLimit + 1 },
      (_, index) => `--insight-domain-token-${index}: #fff;`
    ).join('\n')
    const result = runUiArchitectureTokenFixture(`
      :root {
        ${tokenDefinitions}
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain(`domain.css defines ${domainTokenLimit + 1} tokens`)
    expect(result.stderr).toContain(`below ${domainTokenLimit}`)
  })

  it('rejects generated palette token names in token files', () => {
    const result = runUiArchitectureTokenFixture(`
      :root {
        ${generatedPaletteToken}: #fff;
        --color-surface-page: var(${generatedPaletteToken});
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('generated palette token definition(s)')
    expect(result.stderr).toContain(generatedPaletteToken)
  })

  it('rejects generated palette variant names in token files', () => {
    const result = runUiArchitectureTokenFixture(`
      :root {
        ${generatedPaletteVariantToken}: #818cf8;
        --color-action-primary: var(${generatedPaletteVariantToken});
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('generated palette token definition(s)')
    expect(result.stderr).toContain(generatedPaletteVariantToken)
  })

  it('rejects value-named semantic token names in token files', () => {
    const result = runUiArchitectureTokenFixture(`
      :root {
        ${valueNamedSemanticToken}: #fff;
        --color-surface-page: var(${valueNamedSemanticToken});
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('value-named token definition(s)')
    expect(result.stderr).toContain(valueNamedSemanticToken)
  })

  it('rejects domain-specific token definitions in the global semantic layer', () => {
    const result = runUiArchitectureTokenFixture(`
      :root {
        ${editDomainSemanticToken}: #16213e;
        ${studioDomainSemanticToken}: #183351;
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('domain-specific semantic token definition(s)')
    expect(result.stderr).toContain(editDomainSemanticToken)
    expect(result.stderr).toContain(studioDomainSemanticToken)
  })

  it('rejects implementation-shaped token names in the global semantic layer', () => {
    const result = runUiArchitectureTokenFixture(`
      :root {
        ${semanticGrayScaleToken}: #f8fafc;
        ${semanticEditorSurfaceToken}: #f8f8f8;
        ${semanticAccentPurpleToken}: #9b59b6;
        ${semanticGradientStartToken}: #667eea;
        ${semanticSurfacePlainToken}: #fff;
        ${semanticTextPrimaryStrongToken}: #1f5fc3;
        ${semanticWarningTintToken}: #fff3cd;
      }
    `, 'src/styles/tokens/semantic.css')

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('implementation-shaped semantic token definition(s)')
    expect(result.stderr).toContain(semanticGrayScaleToken)
    expect(result.stderr).toContain(semanticEditorSurfaceToken)
    expect(result.stderr).toContain(semanticAccentPurpleToken)
    expect(result.stderr).toContain(semanticGradientStartToken)
    expect(result.stderr).toContain(semanticSurfacePlainToken)
    expect(result.stderr).toContain(semanticTextPrimaryStrongToken)
    expect(result.stderr).toContain(semanticWarningTintToken)
  })

  it('rejects vague private component token names in global component tokens', () => {
    const result = runUiArchitectureTokenFixture(`
      :root {
        ${vagueComponentToken}: #fff;
      }
    `, 'src/styles/tokens/component.css')

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('vague component token definition(s)')
    expect(result.stderr).toContain(vagueComponentToken)
  })

  it('allows role-named component tokens in global component tokens', () => {
    const result = runUiArchitectureTokenFixture(`
      :root {
        ${roleComponentToken}: #fff;
      }
    `, 'src/styles/tokens/component.css')

    expect(result.status).toBe(0)
    expect(result.stdout).toContain('UI architecture check passed')
  })

  it('rejects root tokens that depend on body-only compatibility tokens', () => {
    const result = runUiArchitectureTokenFixture(`
      :root {
        --color-edit-shell-end: var(--body-only-token);
      }

      body {
        --body-only-token: #1a1a2e;
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('cannot reference body-scoped token --body-only-token')
  })

  it('rejects root tokens that depend on undefined tokens', () => {
    const result = runUiArchitectureTokenFixture(`
      :root {
        --color-edit-shell-end: var(--missing-token);
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('references undefined token --missing-token')
  })

  it('rejects body compatibility aliases in token files', () => {
    const result = runUiArchitectureTokenFixture(`
      :root {
        ${paletteSurfaceEditorToken}: #1a1a2e;
        --color-edit-shell-end: var(${paletteSurfaceEditorToken});
      }

      body {
        ${legacyBgColor}: var(--color-edit-shell-end);
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain(`body defines ${legacyBgColor}; body compatibility aliases are no longer allowed`)
  })

  it('rejects cyclic root token dependencies', () => {
    const result = runUiArchitectureTokenFixture(`
      :root {
        --color-a: var(--color-b);
        --color-b: var(--color-a);
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('token dependency cycle detected')
  })
})

describe('UI architecture CSS variable ownership lint', () => {
  it('rejects business CSS that shadows global semantic tokens', () => {
    const result = runUiArchitectureSourceFixture('InsightView.styles.css', `
      .insight-page {
        ${legacyTextPrimary}: var(--color-text-default);
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain(`reserved global CSS variable ${legacyTextPrimary}`)
  })

  it('allows owner-namespaced business CSS variables', () => {
    const result = runUiArchitectureSourceFixture('InsightView.styles.css', `
      .insight-page {
        --insight-text-primary: var(--color-text-default);
      }

      .insight-page h2 {
        color: var(--insight-text-primary);
      }
    `)

    expect(result.status).toBe(0)
    expect(result.stdout).toContain('UI architecture check passed')
  })

  it('rejects undefined semantic CSS variables without fallbacks in component styles', () => {
    const result = runUiArchitectureSourceFixture('src/components/product/ProductFocusFixture.vue', `
      <template><div class="product-focus-fixture">Focus</div></template>
      <style scoped>
      .product-focus-fixture {
        outline: 2px solid var(--color-focus-ring);
      }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('undefined semantic CSS variable reference(s) --color-focus-ring')
  })

  it('rejects owner-scoped CSS variables that are never referenced', () => {
    const result = runUiArchitectureSourceFixture('src/components/translate/SettingsSidebar.vue', `
      <template><aside class="settings-sidebar"></aside></template>
      <style scoped>
      .settings-sidebar {
        ${unusedOwnerToken}: #fff;
      }

      .settings-sidebar {
        color: var(--color-text-default);
      }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('unused owner CSS variable definition(s)')
    expect(result.stderr).toContain(unusedOwnerToken)
  })

  it('allows owner-scoped CSS variables when the owner uses them', () => {
    const result = runUiArchitectureSourceFixture('src/components/translate/SettingsSidebar.vue', `
      <template><aside class="settings-sidebar"></aside></template>
      <style scoped>
      .settings-sidebar {
        ${unusedOwnerToken}: #fff;
      }

      .settings-sidebar {
        background: var(${unusedOwnerToken});
      }
      </style>
    `)

    expect(result.status).toBe(0)
    expect(result.stdout).toContain('UI architecture check passed')
  })

  it('ignores test-only token references when checking production owner token usage', () => {
    const token = '--translate-architecture-unused-owner-token'
    const componentPath = join(
      frontendRoot,
      'src/components/translate/__UiArchitectureUnusedOwnerTokenFixture.vue'
    )
    const testPath = join(frontendRoot, 'tests/unit/__uiArchitectureUnusedOwnerTokenFixture.spec.ts')

    writeFileSync(componentPath, `
      <template><aside class="ui-architecture-unused-owner-token-fixture"></aside></template>
      <style scoped>
      .ui-architecture-unused-owner-token-fixture {
        ${token}: #fff;
      }
      </style>
    `)
    writeFileSync(testPath, `
      export const testOnlyReference = 'var(${token})'
    `)

    try {
      const result = spawnSync(
        process.execPath,
        ['scripts/check-ui-architecture.mjs'],
        {
          cwd: frontendRoot,
          encoding: 'utf8',
        }
      )

      expect(result.status).toBe(1)
      expect(result.stderr).toContain('unused owner CSS variable definition(s)')
      expect(result.stderr).toContain(token)
    } finally {
      rmSync(componentPath, { force: true })
      rmSync(testPath, { force: true })
    }
  })

  it('rejects global tokens without a production consumer', () => {
    const unusedGlobalToken = '--color-surface-unused-architecture-fixture'
    const result = runUiArchitectureTokenUsageFixture(
      `:root { ${unusedGlobalToken}: #fff; }`,
      'src/components/product/ProductEmptyState.vue',
      '<template><section class="product-empty-state"></section></template>'
    )

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('unused global CSS variable definition(s)')
    expect(result.stderr).toContain(unusedGlobalToken)
  })

  it('rejects primitive defaults that shadow their public owner override variables', () => {
    const result = runUiArchitectureSourceFixture('src/components/product/ProductAvatar.vue', `
      <template><span class="product-avatar"></span></template>
      <style scoped>
      .product-avatar {
        --product-avatar-width: 56px;
        width: var(--product-avatar-width);
      }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('self-shadowing public primitive CSS variable definition(s)')
    expect(result.stderr).toContain('--product-avatar-width')
  })

  it('allows a primitive to provide variables for a descendant primitive', () => {
    const result = runUiArchitectureSourceFixture('src/components/product/ProductPageHeader.vue', `
      <template><header class="product-page-header"><slot /></header></template>
      <style scoped>
      .product-page-header {
        --product-header-action-color: var(--color-text-default);
      }
      </style>
    `)

    expect(result.status).toBe(0)
    expect(result.stdout).toContain('UI architecture check passed')
  })

  it('allows page owners to set public UI primitive variables', () => {
    const result = runUiArchitectureSourceFixture('src/views/CharacterStudioView.vue', `
      <template><main class="studio-page"></main></template>
      <style scoped>
      .studio-page {
        --studio-border-default: rgba(28, 55, 94, 0.08);
        --ui-button-ghost-border: 1px solid var(--studio-border-default);
      }
      </style>
    `)

    expect(result.status).toBe(0)
    expect(result.stdout).toContain('UI architecture check passed')
  })

  it('allows page owners to set public product primitive variables', () => {
    const result = runUiArchitectureSourceFixture('src/views/InsightView.vue', `
      <template><main class="insight-page"></main></template>
      <style scoped>
      .insight-page {
        --insight-view-sidebar-divider: var(--color-border-muted);
        --product-tabbed-workspace-border: var(--insight-view-sidebar-divider);
      }
      </style>
    `)

    expect(result.status).toBe(0)
    expect(result.stdout).toContain('UI architecture check passed')
  })

  it('rejects business CSS that reads legacy compatibility variables', () => {
    const result = runUiArchitectureSourceFixture('BookSearch.vue', `
      <template><div class="book-search"></div></template>
      <style scoped>
      .book-search {
        color: var(${legacyTextColor});
        background: var(${legacyCardBg});
        border-color: var(${legacyBorderColor});
      }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('legacy compatibility CSS variable reference(s)')
    expect(result.stderr).toContain(legacyTextColor)
    expect(result.stderr).toContain(legacyCardBg)
    expect(result.stderr).toContain(legacyBorderColor)
  })

  it('rejects business CSS that reads old global semantic aliases', () => {
    const result = runUiArchitectureSourceFixture('UiButton.vue', `
      <template><div class="ui-button">Test</div></template>
      <style scoped>
      .ui-button {
        color: var(${legacyTextPrimary});
        border-color: var(${legacyColorPrimary});
      }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('legacy global CSS variable reference(s)')
    expect(result.stderr).toContain(legacyTextPrimary)
    expect(result.stderr).toContain(legacyColorPrimary)
  })

  it('rejects business CSS that reads old short alias variables', () => {
    const result = runUiArchitectureSourceFixture('ReaderControls.vue', `
      <template><div class="reader-controls"></div></template>
      <script setup lang="ts">
      document.documentElement.style.setProperty('${legacyReaderBgColor}', '#fff')
      </script>
      <style scoped>
      .reader-controls {
        background: var(${legacyShortBgSecondary});
        border-color: var(${legacyShortBgActive});
        color: var(${legacyShortPrimary});
      }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('legacy short CSS variable reference(s)')
    expect(result.stderr).toContain(legacyShortBgSecondary)
    expect(result.stderr).toContain(legacyShortBgActive)
    expect(result.stderr).toContain(legacyShortPrimary)
    expect(result.stderr).toContain(legacyReaderBgColor)
  })

  it('rejects business CSS that reads generated owner tokens', () => {
    const result = runUiArchitectureSourceFixture('BookCard.vue', `
      <template><div class="book-card"></div></template>
      <style scoped>
      .book-card {
        background: var(${generatedOwnerToken});
      }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('generated owner token reference(s)')
    expect(result.stderr).toContain(generatedOwnerToken)
  })

  it('rejects business CSS that reads private palette tokens', () => {
    const result = runUiArchitectureSourceFixture('BookCard.vue', `
      <template><div class="book-card"></div></template>
      <style scoped>
      .book-card {
        background: var(${privatePaletteToken});
      }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('private palette token reference(s)')
    expect(result.stderr).toContain(privatePaletteToken)
  })
})

describe('UI architecture style ownership lint', () => {
  it('reports heavy owner review signals only in audit output', () => {
    const longTemplate = Array.from(
      { length: 901 },
      (_, index) => `<div class="large-panel__row">${index}</div>`
    ).join('\n')
    const result = runUiArchitectureSourceFixture('LargePanel.vue', `
      <template>
        <section class="large-panel">
          ${longTemplate}
        </section>
      </template>

      <style scoped>
      .large-panel { display: grid; }
      </style>
    `)

    expect(result.status).toBe(0)
    expect(result.stdout).toContain('UI architecture check passed')
  })

  it('allows high-interaction owners under their explicit shell threshold', () => {
    const longTemplate = Array.from(
      { length: 1000 },
      (_, index) => `<div class="edit-workspace__row">${index}</div>`
    ).join('\n')
    const result = runUiArchitectureSourceFixture('src/components/edit/EditWorkspace.vue', `
      <template>
        <section class="edit-workspace">
          ${longTemplate}
        </section>
      </template>

      <style scoped>
      .edit-workspace { display: grid; }
      </style>
    `)

    expect(result.status).toBe(0)
    expect(result.stdout).toContain('UI architecture check passed')
  })

  it('rejects ordinary script CSS imports in UI components', () => {
    const result = runUiArchitectureSourceFixture('Panel.vue', `
      <script setup lang="ts">
      import './Panel.styles.css'
      </script>

      <template><section class="panel"></section></template>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('ordinary script CSS imports are not allowed')
    expect(result.stderr).toContain('Panel.styles.css')
  })

  it('rejects business CSS that overrides UI primitive internals', () => {
    const result = runUiArchitectureSourceFixture('InsightPanel.css', `
      .insight-panel .ui-button--primary {
        min-width: 120px;
      }

      .insight-panel .ui-button--danger {
        color: var(--color-status-danger-text);
      }

      .insight-panel .ui-input {
        width: 100%;
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('UI primitive selector(s)')
    expect(result.stderr).toContain('.ui-button--primary')
    expect(result.stderr).toContain('.ui-button--danger')
    expect(result.stderr).toContain('.ui-input')
  })

  it('rejects scoped Vue styles that override UI primitive internals', () => {
    const result = runUiArchitectureSourceFixture('InsightPanel.vue', `
      <template><section class="insight-panel"></section></template>
      <style scoped>
      .insight-panel .ui-button--primary {
        min-width: 120px;
      }

      .insight-panel .ui-input {
        width: 100%;
      }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('UI primitive selector(s)')
    expect(result.stderr).toContain('.ui-button--primary')
    expect(result.stderr).toContain('.ui-input')
  })

  it('rejects business CSS that targets current UiField internals', () => {
    const result = runUiArchitectureSourceFixture('OpenAIExtraBodyEditor.vue', `
      <template><UiField class="openai-extra-body-editor" /></template>
      <style scoped>
      .openai-extra-body-editor ${primitiveFieldVariantSelector} {
        margin-bottom: 0;
      }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('UI primitive selector(s)')
    expect(result.stderr).toContain('.ui-field')
    expect(result.stderr).toContain(primitiveFieldVariantSelector)
  })

  it('rejects business CSS that chains owner classes to current UiField roots', () => {
    const result = runUiArchitectureSourceFixture('OpenAIExtraBodyEditor.vue', `
      <template><UiField class="openai-extra-body-editor" /></template>
      <style scoped>
      .openai-extra-body-editor.ui-field {
        margin-bottom: 0;
      }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('UI primitive selector(s)')
    expect(result.stderr).toContain('.ui-field')
  })

  it('rejects business CSS that styles primitives through relational element selectors', () => {
    const result = runUiArchitectureSourceFixture('SettingsPanel.vue', `
      <template><section class="settings-panel"></section></template>
      <style scoped>
      .settings-panel :where(button) {
        min-height: 38px;
      }

      .settings-panel__toggle:has(input:checked) {
        color: var(--color-text-brand);
      }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('UI primitive relational selector(s)')
    expect(result.stderr).toContain(':where(button)')
    expect(result.stderr).toContain(':has(input:checked)')
  })

  it('rejects business CSS that styles checkbox primitives through raw input selectors', () => {
    const result = runUiArchitectureSourceFixture('SettingsPanel.vue', `
      <template><section class="settings-panel"></section></template>
      <style scoped>
      .settings-panel__checkbox input[type='checkbox'] {
        width: auto;
      }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('UI primitive input selector(s)')
    expect(result.stderr).toContain("input[type='checkbox']")
  })

  it('rejects generic UiInput usage for boolean controls in business Vue components', () => {
    const result = runUiArchitectureSourceFixture('SettingsPanel.vue', `
      <script setup lang="ts">
      import UiInput from '@/components/ui/UiInput.vue'
      </script>

      <template>
        <section class="settings-panel">
          <UiInput type="checkbox" />
          <UiInput
            type="radio"
          />
        </section>
      </template>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('generic UiInput boolean control(s)')
    expect(result.stderr).toContain('type="checkbox"')
    expect(result.stderr).toContain('type="radio"')
  })

  it('rejects generic UiInput usage for numeric controls in business Vue components', () => {
    const result = runUiArchitectureSourceFixture('TaskWorkbench.vue', `
      <script setup lang="ts">
      import UiInput from '@/components/ui/UiInput.vue'
      </script>

      <template>
        <section class="task-workbench">
          <UiInput :value="String(interval)" type="number" min="0" />
        </section>
      </template>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('generic UiInput numeric control(s)')
    expect(result.stderr).toContain('type="number"')
  })

  it('allows the shared color input primitive to own native color controls', () => {
    const result = runUiArchitectureSourceFixture('src/components/ui/UiColorInput.vue', `
      <template>
        <${'input'} class="ui-color-input" type="color" />
      </template>

      <style scoped>
      .ui-color-input {
        border: var(--ui-colorpicker-border, 1px solid var(--color-border-input));
      }
      </style>
    `)

    expect(result.status).toBe(0)
    expect(result.stdout).toContain('UI architecture check passed')
  })

  it('rejects domain owner token references from UI primitives', () => {
    const result = runUiArchitectureSourceFixture('src/components/ui/UiButton.vue', `
      <template><button class="ui-button">Button</button></template>
      <style scoped>
      .ui-button--ghost {
        border: 1px solid var(${studioOwnerToken});
      }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('business owner token reference(s)')
    expect(result.stderr).toContain(studioOwnerToken)
  })

  it('rejects domain owner token references from product primitives', () => {
    const result = runUiArchitectureSourceFixture('src/components/product/ProductTabbedWorkspace.vue', `
      <template><section class="product-tabbed-workspace">Tabs</section></template>
      <style scoped>
      .product-tabbed-workspace {
        --product-tabbed-workspace-tab-background-active: var(${insightActionPrimaryToken});
        background: var(--product-tabbed-workspace-tab-background-active);
      }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('business owner token reference(s)')
    expect(result.stderr).toContain(insightActionPrimaryToken)
  })

  it('allows explicit global style imports for Teleport or slot reach-through owners', () => {
    const result = runUiArchitectureSourceFixture('Panel.vue', `
      <script setup lang="ts">
      import './Panel.global.styles.css'
      </script>

      <template><section class="panel"></section></template>
    `)

    expect(result.status).toBe(0)
    expect(result.stdout).toContain('UI architecture check passed')
  })

  it('rejects global form primitive skins that style primitive internals from a helper CSS file', () => {
    const result = runUiArchitectureSourceFixture('src/components/ui/form-primitives.css', `
      .ui-settings-field .ui-input,
      .ui-settings-field .ui-select {
        padding: 10px 12px;
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('global form skin selector(s)')
    expect(result.stderr).toContain('.ui-settings-field .ui-input')
    expect(result.stderr).toContain('.ui-settings-field .ui-select')
  })

  it('rejects business provide/inject as a component split transport', () => {
    const result = runUiArchitectureSourceFixture('src/components/insight/ContinuationPanel.vue', `
      <script setup lang="ts">
      import { provide, inject } from 'vue'
      provide(ContinuationStateKey, state)
      const continuationState = inject(ContinuationStateKey)
      </script>
      <template><section class="continuation-panel"></section></template>
      <style scoped>
      .continuation-panel { display: block; }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('business provide/inject')
    expect(result.stderr).toContain('provide')
    expect(result.stderr).toContain('inject')
  })

  it('rejects raw visual values inside BaseModal customStyle maps', () => {
    const result = runUiArchitectureSourceFixture('src/components/settings/PluginAgentModal.vue', `
      <template>
        <BaseModal
          custom-class="plugin-agent-modal"
          :custom-style="{
            width: '95vw',
            '--plugin-agent-modal-surface-base': '#eee',
            '--plugin-agent-modal-shadow-default': 'rgba(15, 23, 42, .05)'
          }"
        />
      </template>
      <style scoped>
      .plugin-agent-modal-owner { display: block; }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('raw visual customStyle value(s)')
    expect(result.stderr).toContain('#eee')
    expect(result.stderr).toContain('rgba(15, 23, 42, .05)')
  })

  it('rejects business BaseModal customStyle maps even when values are layout-only', () => {
    const result = runUiArchitectureSourceFixture('src/components/settings/PluginAgentModal.vue', `
      <template>
        <BaseModal
          custom-class="plugin-agent-modal"
          :custom-style="{
            width: '95vw',
            maxHeight: '90vh'
          }"
        />
      </template>
      <style scoped>
      .plugin-agent-modal-owner { display: block; }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('BaseModal customStyle is not allowed in business UI')
  })

  it('rejects business BaseModal visual props', () => {
    const result = runUiArchitectureSourceFixture('src/components/translate/WebImportModal.vue', `
      <template>
        <BaseModal
          custom-class="web-import-modal"
          border="1px solid var(--color-border-default)"
          box-shadow="0 24px 64px var(--shadow-medium)"
          footer-background="var(--color-surface-muted)"
        />
      </template>
      <style scoped>
      .web-import-modal-owner { display: block; }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('BaseModal visual prop(s)')
    expect(result.stderr).toContain('border')
    expect(result.stderr).toContain('box-shadow')
    expect(result.stderr).toContain('footer-background')
  })

  it('rejects scoped BaseModal internals styling in business components', () => {
    const result = runUiArchitectureSourceFixture('Panel.vue', `
      <template>
        <BaseModal custom-class="panel-modal" />
      </template>

      <style scoped>
      .panel-modal .ui-modal__body {
        padding: 0;
      }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('scoped styles must not target BaseModal .ui-modal__* internals')
  })

  it('rejects local dark-mode media queries in business styles', () => {
    const result = runUiArchitectureSourceFixture('src/components/translate/WebImportDisclaimer.vue', `
      <style scoped>
      .disclaimer-panel {
        color: var(--color-text-default);
      }

      @media (prefers-color-scheme: dark) {
        .disclaimer-panel {
          color: var(--color-text-inverse);
        }
      }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('local prefers-color-scheme dark-mode overrides are not allowed')
  })

  it('rejects tests that locate business UI through primitive internal classes', () => {
    const result = runUiArchitectureSourceFixture('Panel.test.ts', `
      const generateButton = wrapper.find('button${primitiveButtonInternalSelector}')
      await generateButton.trigger('click')
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('primitive internal class selector(s)')
    expect(result.stderr).toContain(primitiveButtonInternalSelector)
  })

  it('rejects tests that locate business UI through current UiField internals', () => {
    const result = runUiArchitectureSourceFixture('Panel.test.ts', `
      const field = wrapper.find('${primitiveFieldVariantSelector}')
      const action = wrapper.find('${primitiveFieldLabelActionSelector} button')
      expect(field.exists()).toBe(true)
      expect(action.exists()).toBe(true)
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('primitive internal class selector(s)')
    expect(result.stderr).toContain(primitiveFieldVariantSelector)
    expect(result.stderr).toContain(primitiveFieldLabelActionSelector)
  })

  it('rejects visual specs that locate business UI through primitive internal classes', () => {
    const result = runUiArchitectureSourceFixture('tests/visual/panel.spec.ts', `
      await page.route('**/*', async route => route.continue())
      const exportActions = page.getByRole('group', { name: '概览导出操作' })
      const generateButton = exportActions.locator('${primitiveButtonInternalSelector}')
      await expect(generateButton).toHaveCSS('border-radius', '8px')
      await page.unroute('**/*')
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('primitive internal class selector(s)')
    expect(result.stderr).toContain(primitiveButtonInternalSelector)
  })

  it('rejects tests that locate business modals through BaseModal internals', () => {
    const result = runUiArchitectureSourceFixture('Panel.test.ts', `
      const modalBody = document.body.querySelector('${primitiveModalBodySelector}')
      expect(modalBody).toBeTruthy()
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('primitive internal class selector(s)')
    expect(result.stderr).toContain(primitiveModalBodySelector)
  })
})

describe('UI architecture icon ownership lint', () => {
  it('rejects direct lucide imports outside the UI icon registry', () => {
    const result = runUiArchitectureSourceFixture('src/components/settings/LocalIconButton.vue', `
      <script setup lang="ts">
      import { Search } from '@lucide/vue'
      </script>
      <template>
        <Search />
      </template>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('direct @lucide/vue imports are only allowed in the UI icon registry')
  })

  it('rejects product-level string icon fallback props', () => {
    const result = runUiArchitectureSourceFixture('src/components/product/ProductTabbedWorkspace.vue', `
      <script setup lang="ts">
      export type ProductWorkspaceTab = {
        id: string
        label: string
        iconName?: UiIconName
        icon?: string
      }
      </script>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('product icon props must use typed iconName values instead of string icon fallbacks')
  })

  it('rejects translation progress pool icons typed as raw strings', () => {
    const result = runUiArchitectureSourceFixture('src/composables/translation/parallel/types.ts', `
      export interface PoolStatus {
        name: string
        icon: string
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('translation progress pool icons must use UiIconName, not raw string')
  })

  it('allows the UI icon registry to own lucide imports', () => {
    const result = runUiArchitectureSourceFixture('src/components/ui/iconRegistry.ts', `
      import { Search } from '@lucide/vue'
      export const icons = { search: Search }
    `)

    expect(result.status).toBe(0)
    expect(result.stdout).toContain('UI architecture check passed')
  })
})

describe('UI architecture layout shell lint', () => {
  it('allows select primitives to own Teleport dropdown positioning', () => {
    const comboboxResult = runUiArchitectureSourceFixture('src/components/ui/UiCombobox.vue', `
      <template><div class="ui-combobox-dropdown"></div></template>
      <style scoped>
      .ui-combobox-dropdown {
        position: fixed;
        inset: auto;
      }
      </style>
    `)
    const selectResult = runUiArchitectureSourceFixture('src/components/ui/UiSelect.vue', `
      <template><div class="ui-select-dropdown"></div></template>
      <style scoped>
      .ui-select-dropdown {
        position: fixed;
        inset: auto;
      }
      </style>
    `)

    expect(comboboxResult.status).toBe(0)
    expect(comboboxResult.stdout).toContain('UI architecture check passed')
    expect(selectResult.status).toBe(0)
    expect(selectResult.stdout).toContain('UI architecture check passed')
  })

  it('rejects page-owned viewport height algorithms', () => {
    const result = runUiArchitectureSourceFixture('TranslateView.vue', `
      <template><main class="translate-page"></main></template>
      <style scoped>
      .translate-page {
        min-height: 100vh;
      }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('layout bypass detected')
  })

  it('rejects page-owned fixed overlay algorithms', () => {
    const result = runUiArchitectureSourceFixture('InsightView.vue', `
      <template><main class="insight-page"></main></template>
      <style scoped>
      .insight-page .loading-overlay {
        position: fixed;
        inset: 0;
      }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('layout bypass detected')
  })
})

describe('UI architecture old implementation mindset lint', () => {
  it('rejects UI comments that describe the current implementation as copied from the old UI', () => {
    const result = runUiArchitectureSourceFixture('BookCard.vue', `
      <template><article class="book-card"></article></template>
      <script setup lang="ts">
      // ${migrationMindsetKeepExisting}视觉，${migrationMindsetReplica}交互。
      </script>
      <style scoped>
      /* ${migrationMindsetStyleSource} */
      .book-card { display: block; }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('old implementation mindset comment')
    expect(result.stderr).toContain(migrationMindsetKeepExisting)
    expect(result.stderr).toContain(migrationMindsetReplica)
    expect(result.stderr).toContain(migrationMindsetStyleSource)
  })

  it('rejects UI comments that keep old source-file or old-version wording', () => {
    const result = runUiArchitectureSourceFixture('ReaderControls.vue', `
      <template><section class="reader-controls"></section></template>
      <script setup lang="ts">
      // ${migrationMindsetOldVersion}
      // ${migrationMindsetMigratedFrom}
      // ${migrationMindsetOriginalReference}
      // ${migrationMindsetOriginalJsReference}
      // ${migrationMindsetOldFileName}
      </script>
      <style scoped>
      /* ${migrationMindsetCompleteStyles} */
      .reader-controls { display: block; }
      </style>
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('old implementation mindset comment')
    expect(result.stderr).toContain(migrationMindsetCompleteStyles)
    expect(result.stderr).toContain(migrationMindsetOldVersion)
    expect(result.stderr).toContain(migrationMindsetMigratedFrom)
    expect(result.stderr).toContain(migrationMindsetOriginalCore)
    expect(result.stderr).toContain(migrationMindsetOriginalJsReference)
    expect(result.stderr).toContain(migrationMindsetOldFileNameCore)
  })

  it('rejects implementation-history labels in frontend contract comments', () => {
    const result = runUiArchitectureSourceFixture('src/types/bubble.ts', `
      // ${migrationMindsetSimplifiedDesign}
      export type Direction = 'vertical' | 'horizontal'
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('old implementation mindset comment')
    expect(result.stderr).toContain(migrationMindsetSimplifiedDesign)
  })

  it('keeps audit output free of accepted debt noise', () => {
    const result = runUiArchitectureAudit()

    expect(result.status).toBe(0)
    expect(result.stderr).not.toContain('accepted large SFC owners')
    expect(result.stderr).not.toContain('accepted large CSS owners')
    expect(result.stderr).not.toContain('permanent shell/layout owners')
    expect(result.stderr).not.toContain('pending layout')
    expect(result.stderr).toContain('heavy owner review signals')
    expect(result.stderr).toContain('owner token density signals')
  })
})

describe('UI architecture source hygiene lint', () => {
  it('rejects stale requirement and property narration inside frontend tests', () => {
    const result = runUiArchitectureSourceFixture('tests/property/example.property.ts', `
      /**
       * ${staleTestFeatureLabel}, ${staleTestPropertyLabel}: scaffold-era behavior contract
       * ${staleTestRequirementLabel} 1.1, 1.2
       */
      import { describe, it } from 'vitest'

      describe('example property', () => {
        it('keeps the current product contract', () => {})
      })
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('stale test requirement/property narration')
    expect(result.stderr).toContain(staleTestFeatureLabel)
    expect(result.stderr).toContain(staleTestRequirementLabel)
  })

  it('rejects implementation-history wording inside production composables', () => {
    const result = runUiArchitectureSourceFixture('src/composables/useExtractedState.ts', `
      // ${composableHistoryExtracted}的逻辑。
      // 【${composableHistorySimplified}】保留当前值。
      export function useExtractedState() {
        return {}
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('composable implementation-history wording')
    expect(result.stderr).toContain(composableHistoryExtracted)
    expect(result.stderr).toContain(composableHistorySimplified)
  })

  it('rejects explicit any in production composables', () => {
    const result = runUiArchitectureSourceFixture('src/composables/useUnsafePayload.ts', `
      export function useUnsafePayload(payload: unknown) {
        return payload ${composableExplicitAny}
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('explicit any in production composable')
    expect(result.stderr).toContain(composableExplicitAny)
  })

  it('rejects explicit any in production types', () => {
    const result = runUiArchitectureSourceFixture('src/types/unsafePayload.ts', `
      export interface UnsafePayload {
        payload: ${'any'}
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('explicit any in production type')
    expect(result.stderr).toContain(': any')
  })

  it('rejects explicit any in production utils', () => {
    const result = runUiArchitectureSourceFixture('src/utils/unsafeParser.ts', `
      export function parsePayload(payload: unknown) {
        return payload ${composableExplicitAny}
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('explicit any in production util')
    expect(result.stderr).toContain(composableExplicitAny)
  })

  it('rejects unresolved relative exports in frontend source', () => {
    const result = runUiArchitectureSourceFixture('src/types/index.ts', `
      ${relativeExportStatement} '${missingRelativeExport}'
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('unresolved relative export')
    expect(result.stderr).toContain(missingRelativeExport)
  })

  it('requires current page-domain type modules in the public type barrel', () => {
    const result = runUiArchitectureSourceFixture('src/types/index.ts', `
      export interface LocalOnly {}
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('types barrel missing export(s)')
    expect(result.stderr).toContain(characterStudioTypeExport)
    expect(result.stderr).toContain(webImportTypeExport)
  })

  it.each(['build_output.txt', 'build_error.txt', 'vite-dev.log'])(
    'rejects checked-in local build and dev log file %s',
    artifactName => {
      const result = runUiArchitectureSourceFixture(artifactName, `
        vite build output
      `)

      expect(result.status).toBe(1)
      expect(result.stderr).toContain('local build/dev log files are not allowed')
    }
  )
})

describe('UI architecture frontend schema compatibility lint', () => {
  it('rejects legacy provider ids and manifest alias fields in frontend source', () => {
    const result = runUiArchitectureSourceFixture('aiProviders.ts', `
      export const customProvider = '${legacyProviderCustomOpenAi}'
      export const customVisionProvider = '${legacyProviderCustomOpenAiVision}'
      export const manifest = { ${legacyIdsField}: ['${legacyProviderCustomOpenAi}'] }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('legacy frontend schema/provider reference(s)')
    expect(result.stderr).toContain(legacyProviderCustomOpenAi)
    expect(result.stderr).toContain(legacyProviderCustomOpenAiVision)
    expect(result.stderr).toContain(legacyIdsField)
  })

  it('rejects old settings migration and mirror helpers in frontend source', () => {
    const result = runUiArchitectureSourceFixture('settings.ts', `
      const ${legacyStorageKey} = 'saber-translator-settings'
      function ${oldSchemaMergeHelper}() {}
      function ${oldStripMirrorHelper}() {}
      function ${oldSyncMirrorHelper}() {}
      function ${oldCoerceRetryHelper}() {}
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('legacy frontend schema/provider reference(s)')
    expect(result.stderr).toContain(legacyStorageKey)
    expect(result.stderr).toContain(oldSchemaMergeHelper)
    expect(result.stderr).toContain(oldStripMirrorHelper)
    expect(result.stderr).toContain(oldSyncMirrorHelper)
    expect(result.stderr).toContain(oldCoerceRetryHelper)
  })

  it('rejects old OpenAI mirror fields and OCR threshold fields in frontend source', () => {
    const result = runUiArchitectureSourceFixture('openaiOptions.ts', `
      export const options = {
        ${oldIsJsonModeField}: true,
        ${oldForceJsonField}: true,
        ${oldMaxRetriesField}: 3,
        ${threshold48pxField}: 0.7,
        ${thresholdMangaOcrField}: 0.8,
        ${thresholdPaddleOcrField}: 0.9,
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('legacy frontend schema/provider reference(s)')
    expect(result.stderr).toContain(oldIsJsonModeField)
    expect(result.stderr).toContain(oldForceJsonField)
    expect(result.stderr).toContain(oldMaxRetriesField)
    expect(result.stderr).toContain(threshold48pxField)
    expect(result.stderr).toContain(thresholdMangaOcrField)
    expect(result.stderr).toContain(thresholdPaddleOcrField)
  })

  it('rejects optional current schema version fields in frontend source', () => {
    const result = runUiArchitectureSourceFixture('src/types/webImport.ts', `
      export interface WebImportSettingsPayload {
        ${webImportSchemaVersionField}?: number
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('current schema version fields must be required')
    expect(result.stderr).toContain(`${webImportSchemaVersionField}?:`)
  })

  it('rejects partial WebImport settings/provider payload types in frontend source', () => {
    const result = runUiArchitectureSourceFixture('src/api/webImport.ts', `
      import type { WebImportProviderConfigs, WebImportSettings, WebImportSettingsPayload } from '@/types/webImport'
      export interface WebImportSettingsResponse {
        settings?: ${partialWebImportSettings}
        providerConfigs?: ${partialWebImportProviderConfigs}
      }
      export function save(payload: ${partialWebImportSettingsPayload}) {
        return payload
      }
    `)

    expect(result.status).toBe(1)
    expect(result.stderr).toContain('WebImport settings/provider payloads must enter the frontend as unknown')
    expect(result.stderr).toContain(partialWebImportSettings)
    expect(result.stderr).toContain(partialWebImportProviderConfigs)
    expect(result.stderr).toContain(partialWebImportSettingsPayload)
  })
})
