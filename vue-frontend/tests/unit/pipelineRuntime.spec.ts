import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it } from 'vitest'
import { DEFAULT_AUTO_GLOSSARY_PROMPT } from '@/constants'
import {
  createPipelineRuntime,
  createTaskContext,
} from '@/composables/translation/core/runtime'
import { createDefaultSettings } from '@/stores/settings/defaults'
import type { ImageData } from '@/types/image'

describe('pipeline runtime', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('uses the current complete default book constraint schema', () => {
    const runtime = createPipelineRuntime('standard', {
      settingsSnapshot: createDefaultSettings(),
      bookId: null,
      chapterId: null,
    })

    expect(runtime.bookTranslationConstraints.glossary).toEqual({
      enabled: false,
      autoExtractEnabled: false,
      autoExtractPrompt: DEFAULT_AUTO_GLOSSARY_PROMPT,
      entries: [],
    })
    expect(runtime.bookTranslationConstraints.non_translate).toEqual({
      enabled: false,
      entries: [],
    })
  })

  it('keeps default book constraints on the shared helper source of truth', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/composables/translation/core/runtime.ts'),
      'utf8',
    )

    expect(source).toContain("createEmptyBookTranslationConstraints")
    expect(source).not.toContain('const defaultConstraints: BookTranslationConstraints = {')
  })

  it('keeps runtime cloning on the shared JSON-safe helper', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/composables/translation/core/runtime.ts'),
      'utf8',
    )

    expect(source).toContain("from '@/utils/deepClone'")
    expect(source).not.toMatch(/function\s+cloneDeep\b/)
    expect(source).not.toContain('JSON.parse(JSON.stringify(value))')
  })

  it('isolates runtime settings snapshots from later source mutations', () => {
    const settings = createDefaultSettings()
    settings.textStyle.fontSize = 18

    const runtime = createPipelineRuntime('standard', {
      settingsSnapshot: settings,
      bookId: null,
      chapterId: null,
    })
    settings.textStyle.fontSize = 42

    expect(runtime.settingsSnapshot.textStyle.fontSize).toBe(18)
  })

  it('isolates task source images from later source mutations', () => {
    const sourceImage: ImageData = {
      id: 'img-1',
      fileName: 'page-1.png',
      originalDataURL: 'data:image/png;base64,orig',
      translatedDataURL: null,
      cleanImageData: null,
      bubbleStates: null,
      translationStatus: 'pending',
      translationFailed: false,
      fontSize: 16,
      autoFontSize: false,
      fontFamily: 'fonts/STSONG.TTF',
      layoutDirection: 'auto',
      textColor: '#000000',
      fillColor: '#ffffff',
      inpaintMethod: 'solid',
      strokeEnabled: false,
      strokeColor: '#000000',
      strokeWidth: 1,
      hasUnsavedChanges: false,
    }

    const context = createTaskContext(0, sourceImage, 'standard')
    sourceImage.fileName = 'mutated.png'

    expect(context.sourceImage.fileName).toBe('page-1.png')
  })
})
