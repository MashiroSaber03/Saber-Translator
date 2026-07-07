import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const storeFiles = [
  'src/stores/imageStore.ts',
  'src/stores/bubbleStore.ts',
  'src/stores/bookshelfStore.ts',
]

const bookshelfPropertyFiles = [
  'tests/property/bookshelfStore.property.ts',
  'tests/property/bookCrud.property.ts',
  'tests/property/chapterReorder.property.ts',
  'tests/property/tagBatch.property.ts',
]

describe('store source contracts', () => {
  it('keeps core stores free of scaffold-style section narration', () => {
    for (const file of storeFiles) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      expect(source, file).not.toContain('// ============================================================')
      expect(source, file).not.toContain('状态定义')
      expect(source, file).not.toContain('Store 定义')
      expect(source, file).not.toContain('计算属性')
    }
  })

  it('keeps bubble store comments focused on behavior rather than API narration', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/stores/bubbleStore.ts'), 'utf8')

    expect(source).not.toContain('气泡状态管理 Store')
    expect(source).not.toContain('选择管理方法')
    expect(source).not.toContain('序列化方法')
    expect(source).not.toContain('返回 Store' + ' 接口')
    expect(source).not.toContain('@' + 'param')
    expect(source).not.toContain('@' + 'returns')
  })

  it('keeps bookshelf store comments focused on state behavior rather than API narration', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/stores/bookshelfStore.ts'), 'utf8')

    for (const staleNarration of [
      '书架状态管理' + ' Store',
      '类型' + '定义',
      '书籍排序' + '方式',
      '搜索和筛选' + '方法',
      'API 调用' + '方法',
      '返回 Store' + ' 接口',
      '书籍管理' + '（本地）',
      '@' + 'param',
      '@' + 'returns',
    ]) {
      expect(source).not.toContain(staleNarration)
    }
  })

  it('keeps bookshelf property tests focused on behavior contracts', () => {
    for (const file of bookshelfPropertyFiles) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')

      for (const staleNarration of [
        '书架状态管理属性测试',
        '书籍CRUD操作属性测试',
        '章节拖拽排序属性测试',
        '标签批量操作属性测试',
        '测试数据生成器',
        '属性测试',
        '生成有效',
        '生成书籍数据的 Arbitrary',
        '生成章节数据的 Arbitrary',
        '生成标签数据的 Arbitrary',
        '// ============================================================',
        '/' + '**',
        '验证：',
        '设置书籍列表',
      ]) {
        expect(source, file).not.toContain(staleNarration)
      }
    }
  })

  it('keeps bubble store property tests focused on behavior contracts', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/property/bubbleStore.property.ts'), 'utf8')

    for (const staleNarration of [
      '气泡状态管理属性测试',
      '测试数据生成器',
      '属性测试',
      '生成有效',
      '生成气泡状态数组',
      '确保 x2 > x1',
      '序列化',
      '反序列化',
      '验证数量一致',
      '验证每个气泡',
      '执行多选操作',
      '删除气泡后索引调整测试',
      '// ============================================================',
      '/' + '**',
      '验证 ',
    ]) {
      expect(source).not.toContain(staleNarration)
    }
  })

  it('keeps overlay pointer preview state out of the durable bubble store', () => {
    const storeSource = readFileSync(resolve(process.cwd(), 'src/stores/bubbleStore.ts'), 'utf8')
    const overlaySource = readFileSync(resolve(process.cwd(), 'src/components/edit/BubbleOverlay.vue'), 'utf8')

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

  it('serializes insight OpenAI-compatible config through the shared store helper', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/stores/insight/insightConfigApiPayload.ts'), 'utf8')

    expect(source).not.toContain('openai_options: { request: { force_json_output: config.value')
    expect(source.match(/openai_options: serializeInsightOpenAiOptions/g) ?? []).toHaveLength(4)
  })

  it('keeps insight store comments focused on current contracts', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/stores/insightStore.ts'), 'utf8')

    for (const staleNarration of [
      '使用拆分的 composables',
      '导入拆分的 composables',
      '从统一类型导入',
      '公开 Store 相关类型',
      'Store 内部使用的类型别名',
      '// ============================================================',
      '核心状态',
      'Composables 初始化',
      '配置管理',
      '计算属性',
      '状态管理方法',
      '问答管理',
      '笔记管理',
      '使用 configManager',
    ]) {
      expect(source).not.toContain(staleNarration)
    }
  })

  it('keeps insight progress property tests focused on behavior contracts', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/property/insightProgress.property.ts'), 'utf8')

    for (const staleNarration of [
      '漫画分析进度状态属性测试',
      '测试数据生成器',
      '属性测试',
      '生成有效',
      '// ============================================================',
      '/' + '**',
      '验证',
      'return true',
      '创建新的 Pinia 实例',
    ]) {
      expect(source).not.toContain(staleNarration)
    }
  })

  it('keeps insight QA property tests focused on behavior contracts', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/property/insightQA.property.ts'), 'utf8')

    for (const staleNarration of [
      '漫画分析问答流式响应属性测试',
      '测试数据生成器',
      '属性测试',
      '生成有效',
      '// ============================================================',
      '/' + '**',
      '验证',
      'return true',
      '每次测试创建新的 Pinia 实例',
      '模拟流式响应',
    ]) {
      expect(source).not.toContain(staleNarration)
    }
  })

  it('keeps insight API config serialization in named helpers', () => {
    const storeSource = readFileSync(resolve(process.cwd(), 'src/stores/insightStore.ts'), 'utf8')
    const helperSource = readFileSync(resolve(process.cwd(), 'src/stores/insight/insightConfigApiPayload.ts'), 'utf8')

    for (const inlineMapper of [
      'vlm: { provider: config.value.vlm.provider',
      'chat_llm: { use_same_as_vlm: config.value.llm.useSameAsVlm',
      'analysis: { batch: { pages_per_batch: config.value.batch.pagesPerBatch',
      'rerankerProvider: mapProvider(providerConfigs.value.reranker, c => ({ api_key:',
      'imageGenProvider: mapProvider(providerConfigs.value.imageGen, c => ({ api_key:',
    ]) {
      expect(storeSource).not.toContain(inlineMapper)
    }

    expect(storeSource).toContain("from './insight/insightConfigApiPayload'")
    expect(storeSource).toContain('buildInsightConfigApiPayload')
    expect(storeSource).not.toContain('function serializeActiveVlmConfigForApi')
    expect(storeSource).not.toContain('function mapProviderConfig')

    for (const serializerName of [
      'serializeActiveVlmConfigForApi',
      'serializeActiveLlmConfigForApi',
      'serializeBatchConfigForApi',
      'serializeRerankerProviderConfigForApi',
      'serializeImageGenProviderConfigForApi',
    ]) {
      expect(helperSource).toContain(`function ${serializerName}`)
    }
  })

  it('keeps insight provider default normalization outside the central store', () => {
    const storeSource = readFileSync(resolve(process.cwd(), 'src/stores/insightStore.ts'), 'utf8')
    const defaultsSource = readFileSync(resolve(process.cwd(), 'src/stores/insight/insightConfigDefaults.ts'), 'utf8')

    expect(storeSource).toContain("from './insight/insightConfigDefaults'")
    expect(storeSource).toContain('normalizeInsightRerankerConfig')
    expect(storeSource).toContain('normalizeInsightImageGenConfig')
    expect(storeSource).not.toContain('function normalizeRerankerConfig')
    expect(storeSource).not.toContain('function normalizeImageGenConfig')
    expect(defaultsSource).toContain('export function normalizeInsightRerankerConfig')
    expect(defaultsSource).toContain('export function normalizeInsightImageGenConfig')
  })

  it('keeps insight local storage schema parsing outside the central store', () => {
    const storeSource = readFileSync(resolve(process.cwd(), 'src/stores/insightStore.ts'), 'utf8')
    const storageSource = readFileSync(resolve(process.cwd(), 'src/stores/insight/insightConfigStorage.ts'), 'utf8')

    expect(storeSource).toContain("from './insight/insightConfigStorage'")
    expect(storeSource).toContain('buildInsightConfigStoragePayload')
    expect(storeSource).toContain('parseInsightConfigStorage')
    expect(storeSource).not.toContain('function isStoreVlmConfig')
    expect(storeSource).not.toContain('function isOpenAiOptions')
    expect(storeSource).not.toContain('function parseInsightConfigStorage')
    expect(storageSource).toContain('export function buildInsightConfigStoragePayload')
    expect(storageSource).toContain('export function parseInsightConfigStorage')
  })

  it('keeps insight provider settings hydration outside the central store', () => {
    const storeSource = readFileSync(resolve(process.cwd(), 'src/stores/insightStore.ts'), 'utf8')
    const hydrationSource = readFileSync(
      resolve(process.cwd(), 'src/stores/insight/insightProviderSettingsHydration.ts'),
      'utf8'
    )

    expect(storeSource).toContain("from './insight/insightProviderSettingsHydration'")
    expect(storeSource).toContain('applyInsightProviderSettingsFromApi')
    expect(storeSource).not.toContain('if (ps.vlmProvider) for')
    expect(storeSource).not.toContain('if (ps.imageGenProvider) for')
    expect(hydrationSource).toContain('export function applyInsightProviderSettingsFromApi')
  })

  it('keeps active insight API config hydration outside the central store', () => {
    const storeSource = readFileSync(resolve(process.cwd(), 'src/stores/insightStore.ts'), 'utf8')
    const hydrationSource = readFileSync(
      resolve(process.cwd(), 'src/stores/insight/insightConfigApiHydration.ts'),
      'utf8'
    )

    expect(storeSource).toContain("from './insight/insightConfigApiHydration'")
    expect(storeSource).toContain('applyActiveInsightConfigFromApi')
    expect(storeSource).not.toContain('const vlm = apiConfig.vlm')
    expect(storeSource).not.toContain('if (vlm) config.value.vlm =')
    expect(storeSource).not.toContain('if (batch) { const cl = batch.custom_layers')
    expect(hydrationSource).toContain('export function applyActiveInsightConfigFromApi')
  })

  it('keeps insight provider cache manager on shared current option types without scaffold narration', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/stores/insight/useInsightConfigManager.ts'),
      'utf8'
    )

    expect(source).toContain("import type { OpenAICompatibleOptions } from '@/types/settings'")
    expect(source).toContain("import { cloneOpenAiOptions } from '@/utils/openaiOptions'")
    expect(source.match(/openaiOptions: OpenAICompatibleOptions/g) ?? []).toHaveLength(2)
    expect(source).not.toContain('JSON.parse(JSON.stringify')

    for (const staleNarration of [
      'Insight 配置管理 Composable',
      '统一管理 VLM/LLM/Embedding/Reranker/ImageGen 五种服务商配置的保存/恢复',
      '服务商配置字段映射',
      'VLM 配置字段',
      'LLM 配置字段',
      '服务商配置缓存结构',
      '创建配置管理器',
      '保存配置缓存到 localStorage',
      '从 localStorage 加载配置缓存',
      '创建通用的服务商配置管理器',
      '保存当前服务商配置到缓存',
      '从缓存恢复服务商配置',
      'VLM 配置管理器',
      'LLM 配置管理器',
      'Embedding 配置管理器',
      'Reranker 配置管理器',
      'ImageGen 配置管理器',
    ]) {
      expect(source).not.toContain(staleNarration)
    }
  })

  it('keeps the insight store helper barrel free of mechanical narration', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/stores/insight/index.ts'), 'utf8')

    expect(source).not.toContain('Insight Composables 索引文件')
    expect(source).not.toContain('/' + '**')
    expect(source).toContain('useInsightNotes')
    expect(source).toContain('useInsightQA')
    expect(source).toContain('useInsightConfigManager')
  })

  it('keeps the insight QA composable API limited to QA state ownership', () => {
    const qaSource = readFileSync(resolve(process.cwd(), 'src/stores/insight/useInsightQA.ts'), 'utf8')
    const barrelSource = readFileSync(resolve(process.cwd(), 'src/stores/insight/index.ts'), 'utf8')
    const storeSource = readFileSync(resolve(process.cwd(), 'src/stores/insightStore.ts'), 'utf8')

    expect(qaSource).toContain('export function useInsightQA()')
    expect(storeSource).toContain('const qaComposable = useInsightQA()')
    expect(barrelSource).toContain('QAMessage')
    expect(barrelSource).toContain("'" + './useInsightQA' + "'")

    for (const staleContract of [
      'UseInsightQAOptions',
      '_options',
      "import type { Ref } from 'vue'",
      '问答管理 Composable',
      '管理 Insight 问答历史',
      '/' + '** 问答消息 */',
    ]) {
      expect(qaSource).not.toContain(staleContract)
      expect(barrelSource).not.toContain(staleContract)
    }

    expect(storeSource).not.toContain('useInsightQA({')
    expect(storeSource).not.toContain('UseInsightQAOptions')
  })

  it('keeps settings provider cache OpenAI option types sourced from shared settings types', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/stores/settings/types.ts'), 'utf8')

    expect(source).toContain("import type { OpenAICompatibleOptions } from '@/types/settings'")
    expect(source).toContain('export type ProviderOpenAICompatibleOptions')
    expect(source.match(/openaiOptions\?: ProviderOpenAICompatibleOptions/g) ?? []).toHaveLength(4)
    expect(source).not.toMatch(/openaiOptions\?:\s*\{\s*request\?:/s)
  })

  it('keeps web import store on the shared clone helper without scaffold narration', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/stores/webImportStore.ts'), 'utf8')

    expect(source).toContain("import { deepClone } from '@/utils/deepClone'")
    expect(source).toContain("} from './webImportSettingsPayload'")
    expect(source).not.toContain('function parseCurrentWebImportSettings')
    expect(source).not.toContain('function parseCurrentProviderConfigs')
    expect(source).not.toContain('function parseCurrentWebImportPayload')
    expect(source).not.toContain('function parseCurrentLocalPayload')
    expect(source).not.toContain('function cloneValue')
    expect(source).not.toContain('JSON.parse(JSON.stringify(value))')

    for (const staleNarration of [
      '网页导入状态管理 Store',
      '管理网页导入设置、设置草稿和运行时状态',
      '// ============================================================',
      '已提交设置',
      '草稿设置',
      '运行时状态',
      '计算属性',
      'localStorage 持久化',
      '设置草稿操作',
      '运行时状态操作',
    ]) {
      expect(source).not.toContain(staleNarration)
    }
  })

  it('restores session image text-style fields without unknown ImageData casts', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/stores/sessionStore.ts'), 'utf8')

    expect(source).not.toContain('normalizeImageTextStyleFields(img as unknown as Partial<ImageData>)')
  })
})
