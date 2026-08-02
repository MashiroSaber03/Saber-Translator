import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

function source(path: string): string {
  return readFileSync(resolve(process.cwd(), path), 'utf8')
}

describe('frontend state owner architecture', () => {
  it('keeps settings schema and theme lifecycles outside the Pinia facade', () => {
    const settingsStore = source('src/stores/settings/index.ts')

    expect(settingsStore).toContain("from './schema'")
    expect(settingsStore).not.toContain("from './normalizeSettings'")
    expect(settingsStore).toContain("from './useThemePreference'")
    expect(settingsStore).not.toContain('function parseCurrentSettings')
    expect(settingsStore).not.toContain('function ensureNumericTypes')
    expect(settingsStore).not.toContain('function resolveSystemTheme')
  })

  it('keeps Plugin Agent display animation outside the workflow facade', () => {
    const pluginAgent = source('src/components/settings/usePluginAgentModal.ts')

    expect(pluginAgent).toContain("from './usePluginAgentDisplayAnimation'")
    expect(pluginAgent).not.toContain('assistantMessageDisplayTimers')
    expect(pluginAgent).not.toContain('assistantDisplayTimers')
  })

  it('keeps Character Studio chat workflow in its own store owner', () => {
    const characterStudio = source('src/stores/characterStudioStore.ts')

    expect(characterStudio).toContain("from './characterStudio/useCharacterStudioChat'")
    expect(characterStudio).not.toContain('async function sendChatMessage')
    expect(characterStudio).not.toContain('async function regenerateChatMessage')
  })

  it('keeps the translation facade on backend jobs without a browser pipeline', () => {
    const translation = source('src/composables/useTranslationPipeline.ts')

    expect(translation).toContain("from '@/api/v2/translation'")
    expect(translation).toContain('createChapterTranslationJob')
    expect(translation).toContain('createChapterRemoveTextJob')
    expect(translation).not.toContain('usePipeline')
    expect(translation).not.toContain('sessionStore')
    expect(translation).not.toContain('base64')
  })
})
