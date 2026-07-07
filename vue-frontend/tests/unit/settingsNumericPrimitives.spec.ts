import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const fixedSettingsOwners = [
  'src/components/settings/TranslationSettings.vue',
  'src/components/settings/HqTranslationSettings.vue',
  'src/components/settings/OcrSettings.vue',
  'src/components/settings/DetectionSettings.vue',
  'src/components/settings/ProofreadingSettings.vue',
  'src/components/settings/TextStyleDefaultsSettings.vue',
]

describe('settings numeric primitives', () => {
  it('uses UiNumberField for fixed settings number controls', () => {
    for (const filePath of fixedSettingsOwners) {
      const source = readFileSync(resolve(process.cwd(), filePath), 'utf8')

      expect(source, filePath).toContain('UiNumberField')
      expect(source, filePath).not.toMatch(/<UiInput\b(?=[^>]*\btype="number")|<UiInput\b(?=[^>]*\bv-model\.number)/)
    }
  })
})
