import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

function source(path: string): string {
  return readFileSync(resolve(process.cwd(), path), 'utf8')
}

describe('frontend state owner architecture', () => {
  it('keeps settings schema, normalization, and theme lifecycles outside the Pinia facade', () => {
    const settingsStore = source('src/stores/settings/index.ts')

    expect(settingsStore).toContain("from './schema'")
    expect(settingsStore).toContain("from './normalizeSettings'")
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

  it('keeps pipeline runtime free of a session-store cycle', () => {
    const sessionStore = source('src/stores/sessionStore.ts')
    const runtime = source('src/composables/translation/core/runtime.ts')

    expect(sessionStore).toContain("from '@/composables/translation/core/runtime'")
    expect(sessionStore).not.toContain("await import('@/composables/translation/core/runtime')")
    expect(runtime).not.toContain("from '@/stores/sessionStore'")
  })
})
