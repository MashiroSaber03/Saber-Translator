import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

describe('store source contracts', () => {
  it('keeps overlay pointer preview state out of the durable bubble store', () => {
    const storeSource = readFileSync(resolve(process.cwd(), 'src/stores/bubbleStore.ts'), 'utf8')
    const overlaySource = readFileSync(
      resolve(process.cwd(), 'src/components/edit/BubbleOverlay.vue'),
      'utf8'
    )

    for (const field of [
      'isDragging',
      'draggingIndex',
      'dragOffsetX',
      'dragOffsetY',
      'dragInitialX',
      'dragInitialY',
      'isResizing',
      'resizingIndex',
      'resizeCurrentCoords',
      'isRotating',
      'rotatingIndex',
      'rotateCurrentAngle',
    ]) {
      expect(storeSource, field).not.toContain(field)
    }

    expect(overlaySource).not.toContain("from '@/stores/bubbleStore'")
    expect(overlaySource).not.toContain("from 'pinia'")
  })

  it('serializes insight OpenAI-compatible config only at the API boundary', () => {
    const apiSource = readFileSync(resolve(process.cwd(), 'src/api/insight.ts'), 'utf8')
    const storeSource = readFileSync(resolve(process.cwd(), 'src/stores/insightStore.ts'), 'utf8')

    expect(apiSource).toContain('serializeOpenAICompatibleOptionsForApi(draft.openaiOptions)')
    expect(storeSource).not.toContain('serializeOpenAICompatibleOptionsForApi')
    expect(storeSource).not.toContain('openai_options')
  })

  it('keeps one current Insight settings snapshot between the API and store', () => {
    const storeSource = readFileSync(resolve(process.cwd(), 'src/stores/insightStore.ts'), 'utf8')

    expect(storeSource).toContain('function getConfigForApi(): InsightSettingsSnapshot')
    expect(storeSource).toContain('providerDrafts: deepClone(providerConfigs.value)')
    expect(storeSource).toContain('function setConfigFromApi(snapshot: InsightSettingsSnapshot)')
    expect(storeSource).not.toContain('insightConfigApiPayload')
    expect(storeSource).not.toContain('insightConfigApiHydration')
    expect(storeSource).not.toContain('insightProviderSettingsHydration')
    expect(storeSource).not.toContain('provider_settings')
    expect(storeSource).not.toContain('chat_llm')
    expect(storeSource).not.toContain('image_gen')
  })

  it('keeps insight provider default normalization outside the central store', () => {
    const storeSource = readFileSync(resolve(process.cwd(), 'src/stores/insightStore.ts'), 'utf8')
    const defaultsSource = readFileSync(
      resolve(process.cwd(), 'src/stores/insight/insightConfigDefaults.ts'),
      'utf8'
    )

    expect(storeSource).toContain("from './insight/insightConfigDefaults'")
    expect(storeSource).toContain('normalizeInsightRerankerConfig')
    expect(storeSource).toContain('normalizeInsightImageGenConfig')
    expect(storeSource).not.toContain('function normalizeRerankerConfig')
    expect(storeSource).not.toContain('function normalizeImageGenConfig')
    expect(defaultsSource).toContain('export function normalizeInsightRerankerConfig')
    expect(defaultsSource).toContain('export function normalizeInsightImageGenConfig')
  })

  it('keeps Insight business configuration out of browser storage', () => {
    const storeSource = readFileSync(resolve(process.cwd(), 'src/stores/insightStore.ts'), 'utf8')

    expect(storeSource).not.toContain('insightConfigStorage')
    expect(storeSource).not.toContain('localStorage')
    expect(storeSource).not.toContain('buildInsightConfigStoragePayload')
    expect(storeSource).not.toContain('parseInsightConfigStorage')
  })

  it('hydrates Insight provider drafts directly from the current snapshot', () => {
    const storeSource = readFileSync(resolve(process.cwd(), 'src/stores/insightStore.ts'), 'utf8')

    expect(storeSource).toContain('providerConfigs.value = deepClone(snapshot.providerDrafts)')
    expect(storeSource).not.toContain('applyInsightProviderSettingsFromApi')
  })

  it('hydrates active Insight config directly from the current snapshot', () => {
    const storeSource = readFileSync(resolve(process.cwd(), 'src/stores/insightStore.ts'), 'utf8')

    expect(storeSource).toContain('config.value = deepClone(snapshot.config)')
    expect(storeSource).not.toContain('applyActiveInsightConfigFromApi')
  })

  it('keeps insight provider cache manager on shared current option types', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/stores/insight/useInsightConfigManager.ts'),
      'utf8'
    )

    expect(source).toContain('StoreVlmConfig')
    expect(source).toContain('InsightProviderDrafts')
    expect(source).toContain('InsightVlmProviderDraft')
    expect(source).not.toContain('export type VlmProviderDraft')
    expect(source).toContain("import { cloneOpenAiOptions } from '@/utils/openaiOptions'")
    expect(source).not.toContain('Partial<VlmProviderDraft>')
    expect(source).not.toContain('Partial<LlmProviderDraft>')
    expect(source).not.toContain('JSON.parse(JSON.stringify')

  })

  it('keeps the insight QA composable API limited to QA state ownership', () => {
    const qaSource = readFileSync(
      resolve(process.cwd(), 'src/stores/insight/useInsightQA.ts'),
      'utf8'
    )
    const storeSource = readFileSync(resolve(process.cwd(), 'src/stores/insightStore.ts'), 'utf8')

    expect(qaSource).toContain('export function useInsightQA()')
    expect(storeSource).toContain('const qaComposable = useInsightQA()')

    for (const staleContract of [
      'UseInsightQAOptions',
      '_options',
      "import type { Ref } from 'vue'",
    ]) {
      expect(qaSource).not.toContain(staleContract)
    }

    expect(storeSource).not.toContain('useInsightQA({')
    expect(storeSource).not.toContain('UseInsightQAOptions')
  })

  it('keeps settings provider cache OpenAI option types sourced from shared settings types', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/stores/settings/types.ts'), 'utf8')

    expect(source).toContain("import type { OpenAICompatibleOptions } from '@/types/settings'")
    expect(source.match(/openaiOptions\?: OpenAICompatibleOptions/g) ?? []).toHaveLength(4)
    expect(source).not.toMatch(/openaiOptions\?:\s*\{\s*request\?:/s)
  })

  it('keeps web import store on the shared clone helper', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/stores/webImportStore.ts'), 'utf8')

    expect(source).toContain("import { deepClone } from '@/utils/deepClone'")
    expect(source).toContain("} from './webImportSettingsPayload'")
    expect(source).not.toContain('function parseCurrentWebImportSettings')
    expect(source).not.toContain('function parseCurrentProviderConfigs')
    expect(source).not.toContain('function parseCurrentWebImportPayload')
    expect(source).not.toContain('function parseCurrentLocalPayload')
    expect(source).not.toContain('function cloneValue')
    expect(source).not.toContain('JSON.parse(JSON.stringify(value))')

  })
})
