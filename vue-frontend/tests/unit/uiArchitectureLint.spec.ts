import { mkdtempSync, writeFileSync } from 'node:fs'
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
const migrationMindsetKeepExisting = '保持' + '既有'
const migrationMindsetReplica = '复刻' + '原版'
const migrationMindsetStyleSource = 'Source: Panel' + '.styles.css'
const migrationMindsetCompleteStyles = '完整样式 - 从 reader' + '.css 迁移'
const migrationMindsetOldVersion = '旧版 handleBubbleMouseDown'
const migrationMindsetMigratedFrom = '迁移自 main' + '.js'
const migrationMindsetOriginalReference = '对应' + '原版 edit_mode.js'
const migrationMindsetOriginalCore = '原版 edit_mode.js'
const migrationMindsetOldFileName = '当前行为 bookshelf' + '.js'
const migrationMindsetOldFileNameCore = 'bookshelf' + '.js'

function runUiArchitectureTokenFixture(tokensCss: string) {
  const fixtureDir = mkdtempSync(join(tmpdir(), 'ui-architecture-tokens-'))
  const fixturePath = join(fixtureDir, 'tokens.css')
  writeFileSync(fixturePath, tokensCss)

  return spawnSync(
    process.execPath,
    [
      'scripts/check-ui-architecture.mjs',
      '--tokens-fixture',
      fixturePath,
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
  writeFileSync(fixturePath, content)

  return spawnSync(
    process.execPath,
    [
      'scripts/check-ui-architecture.mjs',
      '--source-fixture',
      fixturePath,
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
        --palette-surface-editor: #1a1a2e;
        --color-edit-shell-end: var(--palette-surface-editor);
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
})

describe('UI architecture style ownership lint', () => {
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
    expect(result.stderr).toContain(migrationMindsetOldFileNameCore)
  })

  it('keeps audit output free of accepted debt noise', () => {
    const result = runUiArchitectureAudit()

    expect(result.status).toBe(0)
    expect(result.stderr).not.toContain('accepted large SFC owners')
    expect(result.stderr).not.toContain('accepted large CSS owners')
    expect(result.stderr).not.toContain('permanent shell/layout owners')
    expect(result.stderr).not.toContain('pending layout')
  })
})
