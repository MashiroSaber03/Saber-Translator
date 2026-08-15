import { execFileSync } from 'node:child_process'
import { resolve } from 'node:path'
import { chromium, expect, test, type Browser, type Page, type Route } from '@playwright/test'

import type { components } from '../../src/api/generated/v2'

type V2Schema<Name extends keyof components['schemas']> = components['schemas'][Name]

const jsonHeaders = {
  'content-type': 'application/json; charset=utf-8',
}

function parseCssColorToRgb(value: string): [number, number, number] {
  const rgbMatch = value.match(/^rgb\((\d+),\s*(\d+),\s*(\d+)\)$/)
  if (rgbMatch) {
    return [Number(rgbMatch[1]), Number(rgbMatch[2]), Number(rgbMatch[3])]
  }

  const rgbaMatch = value.match(/^rgba\((\d+),\s*(\d+),\s*(\d+),\s*[\d.]+\)$/)
  if (rgbaMatch) {
    return [Number(rgbaMatch[1]), Number(rgbaMatch[2]), Number(rgbaMatch[3])]
  }

  const srgbMatch = value.match(/^color\(srgb\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\)$/)
  if (srgbMatch) {
    return [
      Math.round(Number(srgbMatch[1]) * 255),
      Math.round(Number(srgbMatch[2]) * 255),
      Math.round(Number(srgbMatch[3]) * 255),
    ]
  }

  throw new Error(`Unsupported CSS color format: ${value}`)
}

function expectCssColorNear(value: string, expected: [number, number, number], tolerance = 1) {
  const actual = parseCssColorToRgb(value)
  actual.forEach((channel, index) => {
    expect(Math.abs(channel - expected[index]!)).toBeLessThanOrEqual(tolerance)
  })
}

async function enableDarkTheme(page: Page) {
  await page.addInitScript(() => {
    window.localStorage.setItem('theme', 'dark')
  })
}

async function expectDarkThemeSurface(page: Page, selector: string) {
  await expect(page.locator(selector)).toBeVisible()
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
  await expect(page.locator('body')).toHaveAttribute('data-theme', 'dark')

  await expect
    .poll(async () => {
      const colors = await page.evaluate(() => {
        const bodyStyle = window.getComputedStyle(document.body)
        return {
          background: bodyStyle.backgroundColor,
          text: bodyStyle.color,
        }
      })
      const background = parseCssColorToRgb(colors.background)
      const text = parseCssColorToRgb(colors.text)
      return {
        ...colors,
        isDarkStable: Math.max(...background) < 80 && Math.min(...text) > 180,
      }
    })
    .toMatchObject({ isDarkStable: true })

  const bodyColors = await page.evaluate(() => {
    const bodyStyle = window.getComputedStyle(document.body)
    return {
      background: bodyStyle.backgroundColor,
      text: bodyStyle.color,
    }
  })
  const background = parseCssColorToRgb(bodyColors.background)
  const text = parseCssColorToRgb(bodyColors.text)

  expect(Math.max(...background)).toBeLessThan(80)
  expect(Math.min(...text)).toBeGreaterThan(180)
}

const demoPageSvg = `
<svg xmlns="http://www.w3.org/2000/svg" width="900" height="1280" viewBox="0 0 900 1280">
  <rect width="900" height="1280" fill="#f7f4ed"/>
  <rect x="80" y="80" width="740" height="1120" rx="18" fill="#ffffff" stroke="#24324a" stroke-width="8"/>
  <rect x="140" y="150" width="620" height="360" rx="12" fill="#e8edf5" stroke="#24324a" stroke-width="5"/>
  <circle cx="330" cy="330" r="88" fill="#8da0bd"/>
  <circle cx="570" cy="330" r="88" fill="#c9a56c"/>
  <rect x="150" y="580" width="260" height="130" rx="60" fill="#ffffff" stroke="#24324a" stroke-width="5"/>
  <rect x="490" y="580" width="260" height="130" rx="60" fill="#ffffff" stroke="#24324a" stroke-width="5"/>
  <rect x="150" y="790" width="600" height="38" rx="19" fill="#2f3f5c"/>
  <rect x="150" y="870" width="520" height="38" rx="19" fill="#2f3f5c"/>
  <rect x="150" y="950" width="580" height="38" rx="19" fill="#2f3f5c"/>
</svg>`

const demoBubbleState = {
  originalText: '原文',
  translatedText: '测试译文',
  textboxText: '',
  coords: [150, 580, 410, 710],
  polygon: [],
  fontSize: 28,
  textDirection: 'horizontal',
  autoTextDirection: 'horizontal',
  textColor: '#000000',
  fillColor: '#ffffff',
  rotationAngle: 0,
  position: { x: 0, y: 0 },
  strokeEnabled: true,
  strokeColor: '#ffffff',
  strokeWidth: 3,
  lineSpacing: 1.2,
  textAlign: 'center',
  inpaintMethod: 'solid',
  autoFgColor: null,
  autoBgColor: null,
  colorConfidence: 0,
  textlines: [],
  ocrResult: null,
} satisfies V2Schema<'BubblePayload'>

const demoStudioDocument = {
  id: 'demo-doc',
  bookId: 'demo-book',
  origin: {
    type: 'analysis',
    source_character: '绫濑澪',
  },
  status: {
    is_favorite: true,
    frozen_sections: ['identity'],
    last_diagnostics: null,
    last_validated_at: '2026-01-01T00:00:00Z',
  },
  meta: {
    title: '绫濑澪',
    tags: ['主角', '视觉回归'],
  },
  identity: {
    name: '绫濑澪',
    aliases: ['澪', '学生会长'],
    description: '冷静、敏锐，擅长在混乱场面里快速整理线索。',
    personality: '外表克制，内心有很强的责任感；说话直接但会照顾同伴的情绪。',
    scenario: '故事进入校园祭前夜，角色需要在紧迫时间内找出异常事件的源头。',
  },
  coreMessages: {
    first_message: '你也注意到走廊尽头的灯光不对劲了吗？先别靠近，我来确认路线。',
    message_example: '<START>\n{{char}}: 我会记录所有细节，哪怕只是一次停顿。',
    alternate_greetings: [
      '现在不是犹豫的时候。你负责观察，我负责判断。',
      '把你刚才看到的顺序再说一遍，我会从里面找出矛盾。',
    ],
    system_prompt: '保持角色冷静、敏锐、具备推理倾向。',
    post_history_instructions: '回复时优先推进调查，并保留角色的克制语气。',
    creator_notes: '视觉回归 fixture，用于覆盖角色工坊编辑态。',
    character_version: '2.0.0',
  },
  lorebook: {
    name: '绫濑澪世界书',
    entries: [
      {
        id: 'entry-campus',
        comment: '校园祭异常',
        keys: ['校园祭', '旧校舍'],
        secondary_keys: ['灯光', '广播'],
        content: '校园祭前夜旧校舍出现异常灯光，广播会重复播放三年前的节目片段。',
        enabled: true,
        constant: false,
        selective: true,
        priority: 120,
        position: 'before_char',
        depth: 4,
        probability: 100,
        prevent_recursion: true,
        use_regex: false,
        match_persona_description: true,
        match_character_description: true,
        match_character_personality: true,
        match_character_depth_prompt: true,
        match_scenario: true,
        children: [
          {
            id: 'entry-clubroom',
            comment: '资料室',
            keys: ['资料室'],
            secondary_keys: [],
            content: '资料室保存着三年前舞台事故的排练记录。',
            enabled: true,
            constant: false,
            selective: true,
            priority: 90,
            position: 'before_char',
            depth: 3,
            probability: 100,
            prevent_recursion: true,
            use_regex: false,
            match_persona_description: true,
            match_character_description: true,
            match_character_personality: true,
            match_character_depth_prompt: true,
            match_scenario: true,
            children: [],
          },
        ],
      },
    ],
  },
  regexScripts: [
    {
      id: 'regex-state',
      scriptName: '隐藏状态块',
      findRegex: '<state>[\\s\\S]*?</state>',
      replaceString: '',
      placement: [2],
      markdownOnly: false,
      promptOnly: false,
      runOnEdit: true,
      disabled: false,
    },
  ],
  stateTasks: [
    {
      id: 'task-trust',
      name: '初始化信任值',
      triggerTiming: 'initialization',
      interval: 0,
      commands: '<<taskjs>>\nawait STscript("/setvar key=trust_score 20");\n<</taskjs>>',
      disabled: false,
    },
  ],
  exportArtifacts: {
    last_review: {
      summary: '角色设定完整，已有可用的世界书与运行时资源。',
      issues: ['备用问候仍可继续丰富场景差异。'],
      suggestions: ['可以补充与学生会成员的关系条目。'],
    },
  },
  revision: 1,
  avatarUrl: null,
  createdAt: '2026-01-01T00:00:00Z',
  updatedAt: '2026-01-01T00:00:00Z',
}

const demoStudioChatSession = {
  session_id: 'demo-session',
  doc_id: 'demo-doc',
  index_revision: 1,
  title: '视觉回归会话',
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
  archived_at: null,
  greeting_source: { type: 'first_message' },
  summary_blocks: [
    {
      summary: '用户和角色正在排查旧校舍灯光异常。',
    },
  ],
  messages: [
    {
      message_id: 'msg-user-1',
      role: 'user',
      content: '我们现在应该先去哪里？',
      attachments: [],
      runtime_log: [],
      variables_snapshot: { trust_score: 20 },
      generation_meta: {},
      created_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-01T00:00:00Z',
    },
    {
      message_id: 'msg-assistant-1',
      role: 'assistant',
      content: '先去资料室。异常广播的时间点和三年前的排练记录可能有关。',
      attachments: [],
      runtime_log: [],
      variables_snapshot: { trust_score: 21 },
      generation_meta: { model: 'visual-fixture' },
      created_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-01T00:00:00Z',
    },
  ],
  variables: { trust_score: 21 },
  revision: 1,
  generation: 1,
}

const fixtureTimestamp = '2026-01-01T00:00:00Z'
const demoV2Chapter = {
  id: 'demo-chapter',
  bookId: 'demo-book',
  ordinal: 1,
  title: 'Chapter 1',
  pageCount: 2,
  pageOrderRevision: 1,
} satisfies V2Schema<'Chapter'>
const demoV2Book = {
  id: 'demo-book',
  title: 'Demo Manga',
  chapterOrderRevision: 1,
  coverAssetUrl: '/api/v2/assets/demo-cover',
  chapterCount: 1,
  pageCount: 2,
  tags: [],
  createdAt: fixtureTimestamp,
  updatedAt: fixtureTimestamp,
  chapters: [demoV2Chapter],
} satisfies V2Schema<'BookDetail'>
function createDemoV2Pages(pageCount: number) {
  return Array.from({ length: pageCount }, (_, index) => {
    const ordinal = index + 1
    return {
      id: `demo-page-${ordinal}`,
      chapterId: 'demo-chapter',
      ordinal,
      logicalSourcePath: `${String(ordinal).padStart(3, '0')}.svg`,
      sourceRevision: 1,
      documentRevision: 1,
      renderedRevision: 1,
      renderStatus: 'ready',
      detectionState: 'ready',
      sourceUrl: `/api/v2/assets/demo-source-${ordinal}`,
      thumbnailSourceUrl: `/api/v2/assets/demo-source-thumb-${ordinal}`,
      cleanUrl: `/api/v2/assets/demo-clean-${ordinal}`,
      translatedUrl: `/api/v2/assets/demo-rendered-${ordinal}`,
      width: 900,
      height: 1280,
    }
  })
}

const demoV2Pages = createDemoV2Pages(2)

const fixtureTextStyle = {
  fontSize: 26,
  autoFontSize: true,
  fontFamily: 'fonts/思源黑体SourceHanSansK-Bold.TTF',
  layoutDirection: 'auto',
  textColor: '#000000',
  fillColor: '#FFFFFF',
  inpaintMethod: 'solid',
  useAutoTextColor: false,
  strokeEnabled: true,
  strokeColor: '#FFFFFF',
  strokeWidth: 3,
  lineSpacing: 1,
  textAlign: 'start',
}

function fixtureOpenAiOptions(useStream: boolean, rpmLimit = 0) {
  return {
    request: { forceJsonOutput: false },
    execution: {
      useStream,
      rpmLimit,
      transportRetries: 1,
      businessRetries: 3,
    },
  }
}

function createFixtureSettings() {
  return {
    settingsSchemaVersion: 5,
    textStyle: { ...fixtureTextStyle },
    ocrEngine: 'manga_ocr',
    sourceLanguage: 'japanese',
    textDetector: 'default',
    minTextBlockAreaPercent: 0.05,
    enableAuxYoloDetection: false,
    auxYoloConfThreshold: 0.4,
    auxYoloOverlapThreshold: 0.1,
    enableSaberYoloRefine: true,
    saberYoloRefineOverlapThreshold: 50,
    baiduOcr: {
      apiKey: '',
      secretKey: '',
      version: 'standard',
      sourceLanguage: 'JAP',
    },
    paddleOcrVl: { sourceLanguage: 'japanese' },
    aiVisionOcr: {
      provider: 'gemini',
      apiKey: '',
      modelName: '',
      prompt: '识别图像中的文字。',
      promptMode: 'normal',
      customBaseUrl: '',
      openaiOptions: fixtureOpenAiOptions(false),
      minImageSize: 32,
    },
    hybridOcr: {
      enabled: false,
      secondaryEngine: '48px_ocr',
      confidenceThreshold: 0.2,
    },
    translation: {
      provider: 'siliconflow',
      apiKey: '',
      modelName: '',
      customBaseUrl: '',
      openaiOptions: fixtureOpenAiOptions(true, 7),
      translationMode: 'batch',
      batchNormalPrompt: '翻译漫画文本。',
      batchJsonPrompt: '以 JSON 翻译漫画文本。',
      singleNormalPrompt: '翻译单个文本框。',
      singleJsonPrompt: '以 JSON 翻译单个文本框。',
    },
    targetLanguage: 'zh',
    translatePrompt: '翻译漫画文本。',
    useTextboxPrompt: false,
    textboxPrompt: '',
    hqTranslation: {
      provider: 'siliconflow',
      apiKey: '',
      modelName: '',
      customBaseUrl: '',
      openaiOptions: fixtureOpenAiOptions(true, 7),
      batchSize: 3,
      prompt: '高质量翻译漫画文本。',
    },
    pluginAgent: {
      provider: 'siliconflow',
      apiKey: '',
      modelName: '',
      customBaseUrl: '',
      openaiOptions: fixtureOpenAiOptions(true),
    },
    proofreading: {
      enabled: false,
      rounds: [],
    },
    boxExpand: {
      ratio: 0,
      top: 0,
      bottom: 0,
      left: 0,
      right: 0,
    },
    preciseMask: {
      dilateSize: 10,
      boxExpandRatio: 20,
    },
    showDetectionDebug: false,
    parallel: {
      enabled: false,
      deepLearningLockSize: 1,
    },
    removeTextWithOcr: false,
    enableVerboseLogs: false,
    lamaDisableResize: false,
  }
}

function fixtureSettingsDocument() {
  const settings = createFixtureSettings()
  return {
    settings: [
      {
        domain: 'translation',
        payload: settings,
        revision: 1,
        schemaVersion: 5,
      },
      {
        domain: 'text_style_defaults',
        payload: settings.textStyle,
        revision: 1,
        schemaVersion: 1,
      },
      {
        domain: 'workflow_preferences',
        payload: {
          rememberWorkflowModeEnabled: true,
          lastWorkflowMode: 'translate-current',
        },
        revision: 1,
        schemaVersion: 1,
      },
      {
        domain: 'web_import',
        payload: {
          firecrawl: {},
          agent: {
            provider: 'openai',
            customBaseUrl: '',
            modelName: 'gpt-4o-mini',
            useStream: false,
            forceJsonOutput: true,
            maxRetries: 3,
            timeout: 120,
          },
          extraction: {
            prompt: '提取网页中的漫画图片。',
            maxIterations: 10,
          },
          download: {
            concurrency: 3,
            timeout: 30,
            retries: 3,
            delay: 100,
            useReferer: true,
          },
          imagePreprocess: {
            enabled: false,
            autoRotate: true,
            compression: {
              enabled: false,
              quality: 85,
              maxWidth: 0,
              maxHeight: 0,
            },
            formatConvert: {
              enabled: false,
              targetFormat: 'original',
            },
          },
          advanced: {
            bypassProxy: false,
          },
          ui: {
            showAgentLogs: true,
            autoImport: false,
          },
        },
        revision: 1,
        schemaVersion: 1,
      },
    ],
    bookSettings: [],
    providerSettings: [],
    credentials: [],
  }
}

function fixtureTranslationBootstrap(loaded: boolean, pages = demoV2Pages) {
  const bookId = loaded ? 'demo-book' : 'quick-workspace'
  const chapterId = loaded ? 'demo-chapter' : 'quick-chapter'
  return {
    activeJobs: [],
    activeWebImportDraft: null,
    book: {
      id: bookId,
      title: loaded ? 'Demo Manga' : '快速翻译',
      kind: loaded ? 'library' : 'quick_workspace',
    },
    chapter: {
      id: chapterId,
      title: loaded ? 'Chapter 1' : '快速翻译',
      pageOrderRevision: 1,
      settingsMemory: {},
      settingsMemorySchemaVersion: 1,
      settingsMemoryRevision: 1,
    },
    constraints: {
      payload: { glossary: {}, nonTranslate: {} },
      schemaVersion: 1,
      revision: 1,
    },
    navigation: {
      lastVisitedPageId: null,
      revision: 1,
    },
    pages: {
      items: loaded ? pages : [],
      nextCursor: null,
      pageOrderRevision: 1,
    },
    settings: fixtureSettingsDocument(),
    fonts: [
      {
        id: 'font-source-han',
        displayName: '思源黑体',
        kind: 'builtin',
        builtinKey: 'source-han-sans-k-bold',
        assetUrl: null,
      },
    ],
    prompts: [],
  }
}

const demoV2StudioDocument = {
  ...demoStudioDocument,
  status: {
    ...demoStudioDocument.status,
    last_diagnostics: null,
  },
  meta: {
    title: demoStudioDocument.meta.title,
    tags: demoStudioDocument.meta.tags,
  },
  title: demoStudioDocument.meta.title,
  revision: 3,
  avatarAssetId: null,
  avatarUrl: null,
  createdAt: fixtureTimestamp,
  updatedAt: fixtureTimestamp,
} satisfies V2Schema<'StudioDocument'>

const demoV2StudioSession = {
  sessionId: 'demo-session',
  documentId: 'demo-doc',
  indexRevision: 2,
  revision: 4,
  generation: 1,
  archived: false,
  archivedAt: null,
  title: demoStudioChatSession.title,
  createdAt: fixtureTimestamp,
  updatedAt: fixtureTimestamp,
  greetingSource: { type: 'first_message' },
  summaryBlocks: demoStudioChatSession.summary_blocks.map(block => ({
    summary: block.summary,
  })),
  summaryThroughMessageId: null,
  summaryGeneration: 0,
  runtimeState: {},
  messages: demoStudioChatSession.messages.map((message, index) => ({
    messageId: message.message_id,
    ordinal: index + 1,
    role: message.role,
    content: message.content,
    attachments: [],
    runtimeLog: message.runtime_log,
    variablesSnapshot: message.variables_snapshot,
    generationMeta: message.generation_meta,
    createdAt: message.created_at,
    updatedAt: message.updated_at,
  })),
  variables: demoStudioChatSession.variables,
} satisfies V2Schema<'StudioChatSession'>

const demoContinuationProject = {
  bookId: 'demo-book',
  characters: [
    {
      aliases: ['D'],
      characterId: 'demo-character',
      enabled: true,
      name: 'Demo',
      payload: { description: 'Visual fixture character' },
      projectId: 'demo-project',
      revision: 1,
    },
  ],
  config: {
    direction: '',
    pageCount: 12,
    styleReferencePages: 3,
  },
  pages: [],
  projectId: 'demo-project',
  referenceAssets: [],
  revision: 1,
  script: null,
  sourceRunId: 'demo-run',
} satisfies V2Schema<'ContinuationProject'>

interface VisualFixtureOptions {
  bookshelfHasBook?: boolean
  insightHasBook?: boolean
  studioDocuments?: boolean
  editBubbles?: boolean
  pages?: typeof demoV2Pages
}

async function mockApi(route: Route, options: VisualFixtureOptions = {}) {
  const requestUrl = new URL(route.request().url())
  const path = requestUrl.pathname
  const method = route.request().method()
  const loadedTranslation =
    requestUrl.searchParams.get('bookId') === 'demo-book' &&
    requestUrl.searchParams.get('chapterId') === 'demo-chapter'
  const fixturePages = options.pages ?? demoV2Pages
  const fixtureChapter = {
    ...demoV2Chapter,
    pageCount: fixturePages.length,
  }
  const fixtureBook = {
    ...demoV2Book,
    chapters: [fixtureChapter],
    pageCount: fixturePages.length,
  }
  const fulfillJson = async (body: unknown, status = 200) => {
    await route.fulfill({
      status,
      headers: jsonHeaders,
      body: JSON.stringify(body),
    })
  }

  if (path === '/api/v2/jobs/events') {
    await route.fulfill({
      headers: {
        'cache-control': 'no-cache',
        'content-type': 'text/event-stream; charset=utf-8',
      },
      body: '',
    })
    return
  }

  if (path === '/api/v2/jobs') {
    await fulfillJson({ items: [], queueRevision: 1 })
    return
  }

  if (path === '/api/v2/system/server-info') {
    await fulfillJson({
      host: '0.0.0.0',
      hostname: 'visual-fixture',
      lanUrl: 'http://192.168.1.100:5000',
      port: 5000,
    })
    return
  }

  if (path === '/api/v2/translation/bootstrap') {
    await fulfillJson(fixtureTranslationBootstrap(loadedTranslation, fixturePages))
    return
  }

  if (path.startsWith('/api/v2/assets/')) {
    await route.fulfill({
      headers: {
        'cache-control': 'public, max-age=31536000, immutable',
        'content-type': 'image/svg+xml',
      },
      body: demoPageSvg,
    })
    return
  }

  if (/^\/api\/v2\/pages\/demo-page-\d+\/document$/.test(path)) {
    const pageId = path.split('/').at(-2) || 'demo-page-1'
    const bubble =
      options.editBubbles && pageId === 'demo-page-1'
        ? [
            {
              bubbleId: 'demo-bubble-1',
              ordinal: 1,
              fontId: 'font-source-han',
              payload: demoBubbleState,
            },
          ]
        : []
    const document = {
      pageId,
      chapterId: 'demo-chapter',
      documentRevision: 1,
      defaultFontId: 'font-source-han',
      pageStyleDefaults: fixtureTextStyle,
      pageStyleSchemaVersion: 1,
      renderStatus: 'not_rendered',
      bubbles: bubble,
    }
    if (route.request().method() === 'PATCH') {
      const command = route.request().postDataJSON() as {
        mutations: Array<{
          bubbleId?: string
          clientMutationId: string
          op: 'create' | 'delete' | 'patch' | 'reset'
        }>
      }
      await fulfillJson({
        document: {
          ...document,
          documentRevision: 2,
        },
        mutationResults: command.mutations.map(mutation => ({
          bubbleId: mutation.bubbleId ?? `created-${mutation.clientMutationId}`,
          clientMutationId: mutation.clientMutationId,
          op: mutation.op,
        })),
      })
    } else {
      await fulfillJson(document)
    }
    return
  }

  if (/^\/api\/v2\/pages\/demo-page-\d+$/.test(path)) {
    const pageNumber = Number(path.split('-').at(-1) || 1)
    await fulfillJson(fixturePages[pageNumber - 1] ?? fixturePages[0])
    return
  }

  if (path === '/api/v2/books') {
    await fulfillJson({
      items: options.bookshelfHasBook ? [fixtureBook] : [],
    })
    return
  }

  if (path === '/api/v2/books/demo-book') {
    await fulfillJson(fixtureBook)
    return
  }

  if (path === '/api/v2/books/demo-book/translation-constraints') {
    await fulfillJson({
      bookId: 'demo-book',
      revision: 1,
      payload: { glossary: {}, nonTranslate: {} },
    })
    return
  }

  if (path === '/api/v2/chapters/demo-chapter/pages') {
    await fulfillJson({
      items: fixturePages,
      nextCursor: null,
      pageOrderRevision: 1,
    })
    return
  }

  if (path === '/api/v2/tags') {
    await fulfillJson({ items: [] })
    return
  }

  if (path === '/api/v2/settings') {
    await fulfillJson(fixtureSettingsDocument())
    return
  }

  if (path === '/api/v2/settings/workflow-preferences' && method === 'PATCH') {
    const body = route.request().postDataJSON() as {
      payload?: Record<string, unknown>
    }
    await fulfillJson({
      domain: 'workflow_preferences',
      payload: body.payload ?? {},
      revision: 2,
      schemaVersion: 1,
    })
    return
  }

  if (path === '/api/v2/fonts') {
    await fulfillJson({
      items: [
        {
          id: 'font-source-han',
          displayName: '思源黑体',
          kind: 'builtin',
          builtinKey: 'source-han-sans-k-bold',
          assetUrl: null,
        },
      ],
    })
    return
  }

  if (path === '/api/v2/prompts') {
    await fulfillJson({ items: [] })
    return
  }

  if (path === '/api/v2/plugins') {
    await fulfillJson({
      items: [
        {
          pluginId: 'demo-plugin',
          displayName: 'Demo Plugin',
          author: 'Visual Fixture',
          description: 'Visual fixture plugin',
          state: 'enabled',
          defaultEnabled: true,
          runtimeEnabled: true,
          config: { replacement: '导师' },
          configRevision: 1,
          errorMessage: null,
          pluginVersionId: 'demo-plugin-version',
          packageVersion: '1.0.0',
          currentRevision: 1,
          manifest: {
            schema_version: 3,
            plugin_id: 'demo-plugin',
            display_name: 'Demo Plugin',
            package_version: '1.0.0',
            entrypoint: 'plugin.py:Plugin',
            hooks: ['after_translate'],
            supported_steps: ['translate'],
            supported_modes: ['standard'],
            priority: 100,
            failure_policy: 'continue',
            author: 'Visual Fixture',
            description: 'Visual fixture plugin',
            default_enabled: true,
            config_schema: {
              replacement: { type: 'text', default: '导师' },
            },
          },
          configSchema: {
            replacement: { type: 'text', default: '导师' },
          },
        },
      ],
    })
    return
  }

  if (path === '/api/v2/web-import/support-checks') {
    await fulfillJson({
      sourceUrl: 'https://example.com/chapter-1',
      galleryDlAvailable: true,
      galleryDlSupported: true,
      recommendedEngine: 'gallery-dl',
    })
    return
  }

  if (path === '/api/v2/web-import/drafts' && method === 'POST') {
    await fulfillJson(
      {
        batchId: 'demo-web-import-batch',
        draftId: 'demo-web-import-draft',
        jobIds: ['demo-web-import-job'],
        status: 'queued',
      },
      202
    )
    return
  }

  if (path === '/api/v2/web-import/drafts/demo-web-import-draft') {
    await fulfillJson({
      actualEngine: 'ai-agent',
      candidateCount: 2,
      chapterId: 'quick-chapter',
      expiresAt: '2026-01-02T00:00:00Z',
      failedCount: 0,
      id: 'demo-web-import-draft',
      jobs: [{ id: 'demo-web-import-job', kind: 'web_import_extract', status: 'completed' }],
      requestedEngine: 'auto',
      revision: 1,
      selectedCount: 2,
      sourceUrl: 'https://example.com/chapter-1',
      status: 'ready',
    })
    return
  }

  if (path === '/api/v2/web-import/drafts/demo-web-import-draft/pages') {
    await fulfillJson({
      items: [1, 2].map(ordinal => ({
        checksum: `checksum-${ordinal}`,
        error: null,
        id: `demo-draft-page-${ordinal}`,
        ordinal,
        selected: true,
        sourceMediaUrl: `/api/v2/assets/demo-web-source-${ordinal}`,
        sourceUrl: `https://example.com/page-${ordinal}.jpg`,
        thumbnailUrl: `/api/v2/assets/demo-web-thumb-${ordinal}`,
      })),
      nextCursor: null,
    })
    return
  }

  if (path === '/api/v2/insight/bootstrap') {
    await fulfillJson({
      activeJobs: [],
      books:
        options.insightHasBook === false
          ? []
          : [
              {
                activeRun: {
                  publishedAt: fixtureTimestamp,
                  runId: 'demo-run',
                  status: 'completed',
                },
                analyzedPageCount: fixturePages.length,
                bookId: 'demo-book',
                coverUrl: '/api/v2/assets/demo-cover',
                pageCount: fixturePages.length,
                title: 'Demo Manga',
              },
            ],
      qa: { available: true, reason: '' },
    })
    return
  }

  if (path === '/api/v2/insight/books/demo-book/chapters') {
    await fulfillJson({
      items: [
        {
          analysisCounts: {
            ready: fixturePages.length,
            stale: 0,
            failed: 0,
            running: 0,
          },
          chapterId: 'demo-chapter',
          ordinal: 1,
          pageCount: fixturePages.length,
          title: 'Chapter 1',
        },
      ],
    })
    return
  }

  if (path === '/api/v2/insight/books/demo-book/pages') {
    await fulfillJson({
      items: fixturePages.map((page, index) => ({
        activeAnalysisId: `demo-analysis-${index + 1}`,
        analysisState: 'ready',
        chapterId: page.chapterId,
        displayPageNumber: index + 1,
        pageId: page.id,
        sourceAssetId: `demo-source-${index + 1}`,
        thumbnailUrl: page.thumbnailSourceUrl,
      })),
      nextCursor: null,
    })
    return
  }

  if (/^\/api\/v2\/insight\/pages\/demo-page-\d+$/.test(path)) {
    const pageNumber = Number(path.split('-').at(-1) || 1)
    await fulfillJson({
      analysis: {
        page_num: pageNumber,
        summary: `第 ${pageNumber} 页的视觉回归分析`,
        dialogues: [],
        characters: [],
      },
      analysisState: 'ready',
      bookId: 'demo-book',
      chapterId: 'demo-chapter',
      chapterTitle: 'Chapter 1',
      displayPageNumber: pageNumber,
      generatedAt: fixtureTimestamp,
      pageId: `demo-page-${pageNumber}`,
      preview: false,
      runId: 'demo-run',
      sourceAssetId: `demo-source-${pageNumber}`,
      sourceUrl: `/api/v2/assets/demo-source-${pageNumber}`,
      staleReasons: [],
    })
    return
  }

  if (path === '/api/v2/insight/notes') {
    await fulfillJson({ items: [], nextCursor: null })
    return
  }

  if (path === '/api/v2/insight/artifacts/overviews') {
    await fulfillJson({ items: ['no_spoiler', 'story_summary', 'character_guide'] })
    return
  }

  if (path.startsWith('/api/v2/insight/artifacts/overviews/')) {
    const template = decodeURIComponent(path.split('/').at(-1) || 'story_summary')
    await fulfillJson({
      artifactId: `demo-overview-${template}`,
      bookId: 'demo-book',
      kind: 'overview',
      payload: {
        title: `视觉回归概览：${template}`,
        content:
          template === 'story_summary'
            ? '这是用于视觉回归的故事概览。'
            : template === 'no_spoiler'
              ? '这是用于视觉回归的无剧透概览。'
              : `已生成的 ${template} 概览。`,
      },
      revision: 1,
      runId: 'demo-run',
      status: 'ready',
      template,
    })
    return
  }

  if (path === '/api/v2/insight/books/demo-book/recent-page-analyses') {
    await fulfillJson({
      items: [
        {
          pageId: 'demo-page-2',
          displayPageNumber: 2,
          summary: '主角在雨夜发现了新的线索。',
          generatedAt: '2026-08-08T08:00:00Z',
        },
        {
          pageId: 'demo-page-1',
          displayPageNumber: 1,
          summary: '故事从安静的街道展开。',
          generatedAt: '2026-08-08T07:00:00Z',
        },
      ],
    })
    return
  }

  if (path === '/api/v2/insight/timeline') {
    await fulfillJson({
      mode: 'enhanced',
      content: {},
      events: [],
      characters: [],
    })
    return
  }

  if (path === '/api/v2/insight/qa/status') {
    await fulfillJson({
      available: true,
      reason: null,
      generation: 1,
      coverage: {
        pages: fixturePages.length,
        events: 0,
      },
    })
    return
  }

  if (path === '/api/v2/insight/books/demo-book/qa') {
    await route.fulfill({
      headers: { 'content-type': 'text/event-stream; charset=utf-8' },
      body: [
        'event: status\n',
        `data: ${JSON.stringify({ requestId: 'visual-qa-request', status: 'retrieving' })}\n\n`,
        'event: context\n',
        `data: ${JSON.stringify({
          mode: 'exact',
          citations: [
            {
              pageId: 'demo-page-1',
              pageNumber: 1,
              excerpt: '视觉回归证据',
              score: 0.9,
            },
          ],
        })}\n\n`,
        'event: chunk\n',
        `data: ${JSON.stringify({ text: '这是一个用于视觉回归的回答，包含稳定的排版和引用。' })}\n\n`,
        'event: done\n',
        'data: {}\n\n',
      ].join(''),
    })
    return
  }

  if (path === '/api/v2/insight/books/demo-book/continuation') {
    await fulfillJson({
      activeRunId: 'demo-run',
      bookId: 'demo-book',
      missing: [],
      project: demoContinuationProject,
      ready: true,
    })
    return
  }

  if (path === '/api/v2/insight/continuation/projects/demo-project/forms') {
    await fulfillJson({
      items: [
        {
          adoptedAssetId: null,
          characterId: 'demo-character',
          formId: 'demo-form',
          imageVersions: [],
          name: '默认',
          payload: {
            description: '默认形态',
            enabled: true,
          },
          referenceAssetId: null,
          referenceAssetUrl: null,
          referenceThumbnailUrl: null,
          revision: 1,
        },
      ],
      nextCursor: null,
    })
    return
  }

  if (path === '/api/v2/studio/books/demo-book/index') {
    await fulfillJson({
      bookId: 'demo-book',
      candidateStatus: { available: true, reason: null },
      documents: options.studioDocuments
        ? [
            {
              avatarAssetId: null,
              documentId: 'demo-doc',
              hasAvatar: false,
              isFavorite: true,
              kind: 'analysis',
              revision: 3,
              sourceCharacter: '绫濑澪',
              tags: ['主角', '视觉回归'],
              title: '绫濑澪',
              updatedAt: fixtureTimestamp,
            },
          ]
        : [],
    })
    return
  }

  if (path === '/api/v2/studio/books/demo-book/candidates') {
    await fulfillJson({ available: true, reason: null, items: [] })
    return
  }

  if (path === '/api/v2/studio/documents/demo-doc') {
    await fulfillJson(demoV2StudioDocument)
    return
  }

  if (path === '/api/v2/studio/documents/demo-doc/chat') {
    await fulfillJson({
      activeSession: demoV2StudioSession,
      availableGreetings: [
        {
          greetingId: 'first_message',
          label: '主问候',
          content: demoStudioDocument.coreMessages.first_message,
          source: { type: 'first_message' },
        },
      ],
      documentId: 'demo-doc',
      indexRevision: 2,
      sessions: [
        {
          sessionId: 'demo-session',
          title: '视觉回归会话',
          revision: 4,
          generation: 1,
          archived: false,
          archivedAt: null,
          messageCount: demoV2StudioSession.messages.length,
          lastMessageExcerpt: demoV2StudioSession.messages.at(-1)?.content || '',
          updatedAt: fixtureTimestamp,
        },
      ],
    })
    return
  }

  if (path.startsWith('/api/v2/')) {
    await fulfillJson(
      {
        error: {
          code: 'visual_fixture_missing',
          message: `Missing visual fixture for ${method} ${path}`,
        },
      },
      501
    )
    return
  }

  await fulfillJson(
    {
      error: {
        code: 'visual_fixture_missing',
        message: `Unexpected non-v2 API request: ${method} ${path}`,
      },
    },
    501
  )
}

async function prepareVisualPage(page: Page, options: VisualFixtureOptions = {}) {
  await page.route('**/*', async route => {
    const requestUrl = new URL(route.request().url())
    if (requestUrl.pathname.startsWith('/api/')) {
      await mockApi(route, options)
      return
    }
    await route.continue()
  })
  await page.addInitScript(() => {
    window.localStorage.clear()
    window.localStorage.setItem('webImportDisclaimerAccepted', 'true')
    window.localStorage.setItem('saber_translator_dismiss_setup_reminder', 'true')
  })
  page.on('console', message => {
    if (message.type() === 'error') {
      console.log(`[browser:${message.type()}] ${message.text()}`)
    }
  })
  page.on('pageerror', error => {
    console.log(`[browser:pageerror] ${error.message}`)
  })
  page.on('response', response => {
    if (response.status() >= 400) {
      console.log(
        `[browser:http-${response.status()}] ${response.request().method()} ${response.url()}`
      )
    }
  })
}

test.beforeEach(async ({ page }) => {
  await prepareVisualPage(page)
})

interface BrowserMemorySnapshot {
  privateBytes: number
  workingSetBytes: number
}

const processMemoryScript = `
import json
import psutil
import sys

private_bytes = 0
working_set_bytes = 0
for raw_pid in sys.argv[1].split(","):
    if not raw_pid:
        continue
    try:
        info = psutil.Process(int(raw_pid)).memory_info()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue
    private_bytes += int(getattr(info, "private", getattr(info, "rss", 0)))
    working_set_bytes += int(info.rss)

print(json.dumps({
    "privateBytes": private_bytes,
    "workingSetBytes": working_set_bytes,
}))
`

async function sampleBrowserMemory(browser: Browser): Promise<BrowserMemorySnapshot> {
  const session = await browser.newBrowserCDPSession()
  try {
    const result = (await session.send('SystemInfo.getProcessInfo')) as {
      processInfo: Array<{ id: number }>
    }
    const processIds = result.processInfo.map(process => process.id)
    const pythonExecutable = resolve(process.cwd(), '..', 'venv', 'Scripts', 'python.exe')
    return JSON.parse(
      execFileSync(pythonExecutable, ['-c', processMemoryScript, processIds.join(',')], {
        encoding: 'utf8',
        timeout: 10_000,
      })
    ) as BrowserMemorySnapshot
  } finally {
    await session.detach()
  }
}

test('100/500/1000-page reader keeps requests, DOM, heap, and process memory bounded', async () => {
  test.setTimeout(120_000)
  const measurements: Array<{
    domNodes: number
    imageRequests: number
    jsHeapMb: number
    maxRenderedImages: number
    pageCount: number
    privateDeltaMb: number
    thumbnailRequests: number
    workingSetDeltaMb: number
  }> = []

  for (const pageCount of [100, 500, 1000]) {
    const browser = await chromium.launch({
      args: ['--enable-precise-memory-info', '--js-flags=--expose-gc'],
      headless: true,
    })
    try {
      const context = await browser.newContext({
        viewport: { width: 1440, height: 1000 },
      })
      const page = await context.newPage()
      await page.goto('about:blank')
      const baseline = await sampleBrowserMemory(browser)
      const sourceRequests = new Set<string>()
      const thumbnailRequests = new Set<string>()

      page.on('request', request => {
        const path = new URL(request.url()).pathname
        if (/^\/api\/v2\/assets\/demo-source-\d+$/.test(path)) {
          sourceRequests.add(path)
        }
        if (/^\/api\/v2\/assets\/demo-(?:source|rendered)-thumb-\d+$/.test(path)) {
          thumbnailRequests.add(path)
        }
      })
      await prepareVisualPage(page, {
        pages: createDemoV2Pages(pageCount),
      })

      const pageSession = await context.newCDPSession(page)
      await pageSession.send('Performance.enable')
      await page.goto('http://127.0.0.1:5173/reader?book=demo-book&chapter=demo-chapter')
      await expect(page.locator('.reader-canvas__stream')).toBeVisible({ timeout: 15_000 })

      const stream = page.locator('.virtual-page-stream')
      const renderedCounts = [await page.locator('.virtual-page-stream__image').count()]
      for (const ratio of [0.5, 1]) {
        await stream.evaluate((element, nextRatio) => {
          element.scrollTop = (element.scrollHeight - element.clientHeight) * nextRatio
          element.dispatchEvent(new Event('scroll'))
        }, ratio)
        await page.waitForTimeout(250)
        renderedCounts.push(await page.locator('.virtual-page-stream__image').count())
      }

      await page.evaluate(() => {
        const collectGarbage = (
          globalThis as typeof globalThis & {
            gc?: () => void
          }
        ).gc
        collectGarbage?.()
      })
      const performance = (await pageSession.send('Performance.getMetrics')) as {
        metrics: Array<{ name: string; value: number }>
      }
      const dom = (await pageSession.send('Memory.getDOMCounters')) as {
        nodes: number
      }
      const current = await sampleBrowserMemory(browser)
      const heapBytes =
        performance.metrics.find(metric => metric.name === 'JSHeapUsedSize')?.value ?? 0
      const toMb = (bytes: number) => bytes / 1024 / 1024

      measurements.push({
        domNodes: dom.nodes,
        imageRequests: sourceRequests.size,
        jsHeapMb: toMb(heapBytes),
        maxRenderedImages: Math.max(...renderedCounts),
        pageCount,
        privateDeltaMb: Math.max(0, toMb(current.privateBytes - baseline.privateBytes)),
        thumbnailRequests: thumbnailRequests.size,
        workingSetDeltaMb: Math.max(0, toMb(current.workingSetBytes - baseline.workingSetBytes)),
      })
    } finally {
      await browser.close()
    }
  }

  for (const measurement of measurements) {
    expect(measurement.maxRenderedImages).toBeLessThanOrEqual(8)
    expect(measurement.imageRequests).toBeLessThanOrEqual(24)
    expect(measurement.thumbnailRequests).toBe(0)
    expect(measurement.domNodes).toBeLessThan(5000)
    expect(measurement.jsHeapMb).toBeLessThan(100)
    expect(measurement.privateDeltaMb).toBeLessThan(300)
    expect(measurement.workingSetDeltaMb).toBeLessThan(300)
  }

  const fiveHundred = measurements.find(item => item.pageCount === 500)!
  const oneThousand = measurements.find(item => item.pageCount === 1000)!
  expect(oneThousand.jsHeapMb - fiveHundred.jsHeapMb).toBeLessThan(15)
  expect(oneThousand.privateDeltaMb - fiveHundred.privateDeltaMb).toBeLessThan(80)
  expect(oneThousand.workingSetDeltaMb - fiveHundred.workingSetDeltaMb).toBeLessThan(80)

  await test.info().attach('reader-memory-trend.json', {
    body: Buffer.from(JSON.stringify(measurements, null, 2)),
    contentType: 'application/json',
  })
})

test('100/500/1000-page thumbnail surfaces keep DOM and process memory bounded', async () => {
  test.setTimeout(process.env.PW_MEMORY_SURFACE ? 90_000 : 360_000)
  type Surface = {
    name: string
    setup: (page: Page) => Promise<{
      horizontal?: boolean
      itemSelector: string
      viewportSelector: string
    }>
  }
  type Measurement = {
    domNodes: number
    imageRequests: number
    jsHeapMb: number
    maxRenderedItems: number
    pageCount: number
    privateDeltaMb: number
    surface: string
    thumbnailRequests: number
    viewportHeight: number
    viewportScrollHeight: number
    workingSetDeltaMb: number
  }

  const surfaces: Surface[] = [
    {
      name: 'translate-sidebar',
      setup: async page => {
        await page.goto('http://127.0.0.1:5173/translate?book=demo-book&chapter=demo-chapter')
        await expect(page.getByTestId('translation-result-display')).toBeVisible()
        return {
          itemSelector: '[data-product-thumbnail-id]',
          viewportSelector: '.thumbnail-sidebar .virtual-thumbnail-list',
        }
      },
    },
    {
      name: 'translate-edit-sidebar',
      setup: async page => {
        await page.goto('http://127.0.0.1:5173/translate?book=demo-book&chapter=demo-chapter')
        await expect(page.getByTestId('translation-result-display')).toBeVisible()
        await page.getByRole('button', { name: '切换编辑模式' }).click()
        await expect(page.locator('.edit-workspace')).toBeVisible()
        await page
          .getByRole('button', {
            name: '显示或隐藏缩略图',
          })
          .first()
          .click()
        return {
          horizontal: true,
          itemSelector: '.edit-thumbnails-panel__item',
          viewportSelector: '.edit-thumbnails-panel__viewport',
        }
      },
    },
    {
      name: 'page-selection',
      setup: async page => {
        await page.goto('http://127.0.0.1:5173/translate?book=demo-book&chapter=demo-chapter')
        await expect(page.getByTestId('translation-result-display')).toBeVisible()
        await page.locator('#workflowModeSelect').click()
        await page
          .getByRole('option', {
            name: '翻译所有图片',
          })
          .click()
        await page.locator('.page-selection-section .product-collapsible-section__header').click()
        const selectionSwitch = page.getByRole('switch', {
          name: '启用指定翻译页码',
        })
        await expect(selectionSwitch).toBeEnabled()
        await selectionSwitch.click()
        await page.getByRole('button', { name: '选择页码' }).click()
        await expect(page.locator('.page-selection-modal')).toBeVisible()
        return {
          itemSelector: '[data-product-thumbnail-id]',
          viewportSelector: '.page-selection-modal .virtual-thumbnail-grid',
        }
      },
    },
    {
      name: 'insight-pages-tree',
      setup: async page => {
        await page.goto('http://127.0.0.1:5173/insight?book=demo-book')
        await expect(page.locator('.pages-tree-panel')).toBeVisible()
        return {
          itemSelector: '[data-product-thumbnail-id]',
          viewportSelector: '.pages-tree-panel__pages-grid.virtual-thumbnail-grid',
        }
      },
    },
  ]
  const measurements: Measurement[] = []
  const selectedSurfaces = process.env.PW_MEMORY_SURFACE
    ? surfaces.filter(surface => surface.name === process.env.PW_MEMORY_SURFACE)
    : surfaces

  for (const surface of selectedSurfaces) {
    for (const pageCount of [100, 500, 1000]) {
      const browser = await chromium.launch({
        args: ['--enable-precise-memory-info', '--js-flags=--expose-gc'],
        headless: true,
      })
      try {
        const context = await browser.newContext({
          viewport: { width: 1440, height: 1000 },
        })
        const page = await context.newPage()
        await page.goto('about:blank')
        const baseline = await sampleBrowserMemory(browser)
        const imageRequests = new Set<string>()
        const thumbnailRequests = new Set<string>()
        page.on('request', request => {
          const path = new URL(request.url()).pathname
          if (/^\/api\/v2\/assets\/demo-(?:source|clean|rendered)-\d+$/.test(path)) {
            imageRequests.add(path)
          }
          if (/^\/api\/v2\/assets\/demo-(?:source|rendered)-thumb-\d+$/.test(path)) {
            thumbnailRequests.add(path)
          }
        })
        await prepareVisualPage(page, {
          pages: createDemoV2Pages(pageCount),
        })
        const surfaceState = await surface.setup(page)
        const viewport = page.locator(surfaceState.viewportSelector)
        await expect(viewport).toBeAttached()
        const viewportGeometry = await viewport.evaluate(element => ({
          height: element.clientHeight,
          scrollHeight: element.scrollHeight,
        }))

        const renderedCounts = [await viewport.locator(surfaceState.itemSelector).count()]
        for (const ratio of [0.5, 1]) {
          await viewport.evaluate(
            (element, state) => {
              if (state.horizontal) {
                element.scrollLeft = (element.scrollWidth - element.clientWidth) * state.ratio
              } else {
                element.scrollTop = (element.scrollHeight - element.clientHeight) * state.ratio
              }
              element.dispatchEvent(new Event('scroll'))
            },
            {
              horizontal: Boolean(surfaceState.horizontal),
              ratio,
            }
          )
          await page.waitForTimeout(250)
          renderedCounts.push(await viewport.locator(surfaceState.itemSelector).count())
        }
        await page.evaluate(() => {
          const collectGarbage = (
            globalThis as typeof globalThis & {
              gc?: () => void
            }
          ).gc
          collectGarbage?.()
        })
        const pageSession = await context.newCDPSession(page)
        await pageSession.send('Performance.enable')
        const performance = (await pageSession.send('Performance.getMetrics')) as {
          metrics: Array<{ name: string; value: number }>
        }
        const dom = (await pageSession.send('Memory.getDOMCounters')) as {
          nodes: number
        }
        await pageSession.detach()
        const current = await sampleBrowserMemory(browser)
        const heapBytes =
          performance.metrics.find(metric => metric.name === 'JSHeapUsedSize')?.value ?? 0
        const toMb = (bytes: number) => bytes / 1024 / 1024
        measurements.push({
          domNodes: dom.nodes,
          imageRequests: imageRequests.size,
          jsHeapMb: toMb(heapBytes),
          maxRenderedItems: Math.max(...renderedCounts),
          pageCount,
          privateDeltaMb: Math.max(0, toMb(current.privateBytes - baseline.privateBytes)),
          surface: surface.name,
          thumbnailRequests: thumbnailRequests.size,
          viewportHeight: viewportGeometry.height,
          viewportScrollHeight: viewportGeometry.scrollHeight,
          workingSetDeltaMb: Math.max(0, toMb(current.workingSetBytes - baseline.workingSetBytes)),
        })
      } finally {
        await browser.close()
      }
    }
  }

  for (const measurement of measurements) {
    expect(measurement.maxRenderedItems, JSON.stringify(measurement)).toBeLessThanOrEqual(64)
    expect(measurement.imageRequests).toBeLessThanOrEqual(12)
    expect(measurement.thumbnailRequests).toBeLessThanOrEqual(128)
    expect(measurement.domNodes).toBeLessThan(7000)
    expect(measurement.jsHeapMb).toBeLessThan(140)
    expect(measurement.privateDeltaMb).toBeLessThan(300)
    expect(measurement.workingSetDeltaMb).toBeLessThan(300)
  }
  for (const surface of selectedSurfaces) {
    const fiveHundred = measurements.find(
      item => item.surface === surface.name && item.pageCount === 500
    )!
    const oneThousand = measurements.find(
      item => item.surface === surface.name && item.pageCount === 1000
    )!
    expect(oneThousand.jsHeapMb - fiveHundred.jsHeapMb).toBeLessThan(25)
    expect(oneThousand.privateDeltaMb - fiveHundred.privateDeltaMb).toBeLessThan(100)
    expect(oneThousand.workingSetDeltaMb - fiveHundred.workingSetDeltaMb).toBeLessThan(100)
  }

  await test.info().attach('thumbnail-surfaces-memory-trend.json', {
    body: Buffer.from(JSON.stringify(measurements, null, 2)),
    contentType: 'application/json',
  })
})

test('dark theme reaches every primary route surface without local overrides', async ({ page }) => {
  await enableDarkTheme(page)

  const routes = [
    { path: '/', selector: '.bookshelf-page' },
    { path: '/translate', selector: '.translate-page' },
    { path: '/insight?book=demo-book', selector: '.insight-page' },
    { path: '/reader?book=demo-book&chapter=demo-chapter', selector: '.reader-page' },
    { path: '/insight/character-studio?book=demo-book', selector: '.studio-page' },
  ]

  for (const route of routes) {
    await page.goto(route.path)
    await expectDarkThemeSurface(page, route.selector)
  }
})

test('dark theme keeps selector primitives readable through one visual contract', async ({
  page,
}) => {
  await enableDarkTheme(page)
  await page.goto('/translate?book=demo-book&chapter=demo-chapter')
  await expect(page.locator('.text-style-section')).toBeVisible()
  const fontSelector = page
    .locator('.text-style-section__field')
    .filter({ hasText: '文本字体' })
    .getByRole('combobox')
  const layoutSelector = page
    .locator('.text-style-section__field')
    .filter({ hasText: '排版方向' })
    .getByRole('combobox')
  await expect(fontSelector).toBeVisible()
  await expect(layoutSelector).toBeVisible()

  const readSelectorStyle = async (locator: typeof fontSelector) =>
    locator.evaluate(element => {
      const style = window.getComputedStyle(element)
      return {
        background: style.backgroundColor,
        text: style.color,
        borderRadius: style.borderRadius,
      }
    })

  const selectorSamples = {
    combobox: await readSelectorStyle(fontSelector),
    select: await readSelectorStyle(layoutSelector),
  }

  for (const sample of [selectorSamples.combobox, selectorSamples.select]) {
    const background = parseCssColorToRgb(sample.background)
    const text = parseCssColorToRgb(sample.text)
    expect(Math.max(...background)).toBeLessThan(80)
    expect(Math.min(...text)).toBeGreaterThan(180)
  }

  expect(selectorSamples.select.text).toBe(selectorSamples.combobox.text)
  expect(selectorSamples.select.background).toBe(selectorSamples.combobox.background)
  expect(selectorSamples.select.borderRadius).toBe(selectorSamples.combobox.borderRadius)
})

test('selector dropdown layers keep an opaque surface above surrounding fields', async ({
  page,
}) => {
  const verifyDropdowns = async () => {
    await page.goto('/translate?book=demo-book&chapter=demo-chapter')
    await expect(page.locator('.text-style-section')).toBeVisible()

    const selectors = [
      {
        trigger: page
          .locator('.text-style-section__field')
          .filter({ hasText: '文本字体' })
          .getByRole('combobox'),
        dropdown: page.locator('.ui-combobox-dropdown'),
      },
      {
        trigger: page
          .locator('.text-style-section__field')
          .filter({ hasText: '排版方向' })
          .getByRole('combobox'),
        dropdown: page.locator('.ui-select-dropdown'),
      },
    ]

    for (const selector of selectors) {
      await selector.trigger.click()
      await expect(selector.dropdown).toBeVisible()
      const surface = await selector.dropdown.evaluate(element => {
        const style = window.getComputedStyle(element)
        return {
          background: style.backgroundColor,
          borderStyle: style.borderStyle,
          position: style.position,
        }
      })
      expect(surface.background).not.toBe('rgba(0, 0, 0, 0)')
      expect(surface.borderStyle).toBe('solid')
      expect(surface.position).toBe('fixed')
      await page.keyboard.press('Escape')
    }
  }

  await verifyDropdowns()
  await enableDarkTheme(page)
  await verifyDropdowns()
})

test('bookshelf empty state keeps its layout contract', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: '我的书架' })).toBeVisible()
  await expect(page.locator('.bookshelf-toolbar__title')).toHaveCSS('color', 'rgb(51, 51, 51)')
  await expect(page).toHaveScreenshot('bookshelf-empty.png', {
    fullPage: true,
    animations: 'disabled',
  })
})

test('book detail modal keeps nested chapter form styling contract', async ({ page }) => {
  await page.unroute('**/*')
  await prepareVisualPage(page, { bookshelfHasBook: true })
  await page.goto('/')
  await page.getByText('Demo Manga').click()
  await expect(page.locator('.book-detail-container')).toBeVisible()
  await page.getByRole('button', { name: /新建章节/ }).click()

  const chapterLabel = page.locator('label[for="chapterTitleInput"]')
  const chapterInput = page.locator('#chapterTitleInput')
  await expect(chapterInput).toBeVisible()
  await expect(chapterLabel).toHaveCSS('display', 'block')
  await expect(chapterInput).toHaveCSS('border-radius', '8px')
  const chapterInputLayout = await chapterInput.evaluate(element => {
    const field = element.closest('.chapter-form-content__field')
    const inputBox = element.getBoundingClientRect()
    const fieldBox = field?.getBoundingClientRect()
    return {
      inputWidth: inputBox.width,
      fieldWidth: fieldBox?.width ?? 0,
    }
  })
  expect(chapterInputLayout.inputWidth).toBeGreaterThan(300)
  expect(Math.abs(chapterInputLayout.inputWidth - chapterInputLayout.fieldWidth)).toBeLessThan(1)
})

test('bookshelf create edit and tag modals keep their form contracts', async ({ page }) => {
  await page.unroute('**/*')
  await prepareVisualPage(page, { bookshelfHasBook: true })
  await page.goto('/')

  await page.getByRole('button', { name: '新建书籍' }).click()
  const createBookModal = page
    .locator('[data-testid="base-dialog-container"]')
    .filter({ hasText: '新建书籍' })
  await expect(createBookModal).toBeVisible()
  await expect(createBookModal.getByLabel('书籍名称')).toHaveCSS('border-radius', '8px')
  await expect(createBookModal).toHaveScreenshot('bookshelf-create-book-modal.png', {
    animations: 'disabled',
  })
  await page.keyboard.press('Escape')

  await page.getByRole('button', { name: /管理标签/ }).click()
  const tagModal = page
    .locator('[data-testid="base-dialog-container"]')
    .filter({ hasText: '标签管理' })
  await expect(tagModal).toBeVisible()
  await expect(tagModal).toHaveScreenshot('bookshelf-tag-manage-modal.png', {
    animations: 'disabled',
  })
  await page.keyboard.press('Escape')

  await page.getByText('Demo Manga').click()
  await page.getByRole('button', { name: '编辑书籍' }).click()
  const editBookModal = page
    .locator('[data-testid="base-dialog-container"]')
    .filter({ hasText: '编辑书籍' })
  await expect(editBookModal).toBeVisible()
  await expect(editBookModal.getByLabel('书籍名称')).toHaveCSS('border-radius', '8px')
  await expect(editBookModal).toHaveScreenshot('bookshelf-edit-book-modal.png', {
    animations: 'disabled',
  })
})

test('book detail delete confirmation keeps BaseModal styling contract', async ({ page }) => {
  await page.unroute('**/*')
  await prepareVisualPage(page, { bookshelfHasBook: true })
  await page.goto('/')
  await page.getByText('Demo Manga').click()
  await page.getByRole('button', { name: '删除书籍' }).click()

  const confirmModal = page.locator('.confirm-modal')
  await expect(confirmModal).toBeVisible()
  await expect(confirmModal.locator('[data-testid="base-dialog-body"]')).toHaveCSS(
    'padding-top',
    '20px'
  )
  await expect(confirmModal).toHaveScreenshot('book-detail-delete-confirm.png', {
    animations: 'disabled',
  })
})

test('translate workspace empty state keeps its layout contract', async ({ page }) => {
  await page.goto('/translate')
  await expect(page.locator('.translate-page')).toBeVisible()
  await expect(page).toHaveScreenshot('translate-empty.png', {
    fullPage: true,
    animations: 'disabled',
  })
})

test('translate loaded workspace keeps fixed sidebar sizing contract', async ({ page }) => {
  await page.goto('/translate?book=demo-book&chapter=demo-chapter')
  await expect(page.getByTestId('translation-result-display')).toBeVisible()
  await expect(
    page.locator('.thumbnail-sidebar').getByRole('button', { name: /选择图片 \d+:/ })
  ).toHaveCount(2)
  await expect(page.locator('.settings-sidebar')).toHaveCSS('width', '300px')
  await expect(page.locator('.settings-sidebar')).toHaveCSS('padding-left', '20px')
  await expect(page.locator('.settings-sidebar')).toHaveCSS('padding-right', '20px')
  await expect(page.locator('.thumbnail-sidebar')).toHaveCSS('width', '230px')
  await expect(page.locator('.thumbnail-sidebar')).toHaveCSS('padding-top', '20px')
  await expect(page.locator('.thumbnail-sidebar')).toHaveCSS('padding-left', '0px')
  await expect(page.locator('.thumbnail-sidebar')).toHaveCSS('padding-right', '0px')
  await expect(page.locator('.thumbnail-sidebar__title')).toHaveCSS('color', 'rgb(44, 62, 80)')
  await expect(page.locator('.translate-shell__main')).toHaveCSS('margin-left', '340px')
  await expect(page.locator('.translate-shell__main')).toHaveCSS('margin-right', '240px')
  await expect(page).toHaveScreenshot('translate-loaded.png', {
    fullPage: true,
    animations: 'disabled',
  })
})

test('translate header keeps enlarged controls inside its surface', async ({ page }) => {
  await page.setViewportSize({ width: 1180, height: 900 })
  await page.goto('/translate?book=demo-book&chapter=demo-chapter')
  await expect(page.getByTestId('translation-result-display')).toBeVisible()

  await page.evaluate(() => {
    document.documentElement.style.fontSize = '200%'
  })

  const surface = page.locator('.product-page-header__content')
  await expect(surface).toBeVisible()
  const surfaceBounds = await surface.boundingBox()
  expect(surfaceBounds).not.toBeNull()

  const visibleRegions = surface.locator(
    '.product-page-header__brand, .product-page-header__nav, .product-page-header__actions'
  )
  const regionCount = await visibleRegions.count()
  for (let index = 0; index < regionCount; index += 1) {
    const bounds = await visibleRegions.nth(index).boundingBox()
    expect(bounds).not.toBeNull()
    expect(bounds!.x).toBeGreaterThanOrEqual(surfaceBounds!.x - 1)
    expect(bounds!.y).toBeGreaterThanOrEqual(surfaceBounds!.y - 1)
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(
      surfaceBounds!.x + surfaceBounds!.width + 1
    )
    expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(
      surfaceBounds!.y + surfaceBounds!.height + 1
    )
  }
})

test('translate edit workspace keeps dark editor shell contract', async ({ page }) => {
  await page.goto('/translate?book=demo-book&chapter=demo-chapter')
  await expect(page.getByTestId('translation-result-display')).toBeVisible()
  await page.getByRole('button', { name: '切换编辑模式' }).click()

  const workspace = page.locator('.edit-workspace')
  const toolbar = page.locator('.edit-toolbar')
  await expect(workspace).toBeVisible()
  await expect(toolbar).toBeVisible()
  const workspaceBackground = await workspace.evaluate(
    element => getComputedStyle(element).backgroundColor
  )
  expectCssColorNear(workspaceBackground, [26, 26, 46], 2)
  await expect(toolbar).toHaveCSS('background-image', /linear-gradient/)
  await expect(page).toHaveScreenshot('translate-edit-workspace.png', {
    fullPage: true,
    animations: 'disabled',
  })
})

test('translate edit workspace selected bubble keeps editor panel contract', async ({ page }) => {
  await page.unroute('**/*')
  await prepareVisualPage(page, { editBubbles: true })
  await page.goto('/translate?book=demo-book&chapter=demo-chapter')
  await expect(page.getByTestId('translation-result-display')).toBeVisible()
  await page.getByRole('button', { name: '切换编辑模式' }).click()

  const workspace = page.locator('.edit-workspace')
  const toolbar = page.locator('.edit-toolbar')
  const bubble = page.locator('.bubble-overlay__highlight-box').first()
  await expect(workspace).toBeVisible()
  await expect(toolbar).toBeVisible()
  await expect(bubble).toBeVisible()
  await bubble.click()
  await expect(bubble).toHaveClass(/bubble-overlay__highlight-box--selected/)
  expect(await page.locator('.bubble-overlay__resize-handle').count()).toBeGreaterThanOrEqual(8)
  await expect(page.locator('.bubble-editor__textarea--translated')).toBeVisible()
  await expect(page.locator('.bubble-editor')).toBeVisible()
  const workspaceBackground = await workspace.evaluate(
    element => getComputedStyle(element).backgroundColor
  )
  expectCssColorNear(workspaceBackground, [26, 26, 46], 2)
  await expect(toolbar).toHaveCSS('background-image', /linear-gradient/)
  await expect(page).toHaveScreenshot('translate-edit-workspace-selected-bubble.png', {
    fullPage: true,
    animations: 'disabled',
  })
})

test('translation settings modal keeps its form styling contract', async ({ page }) => {
  await page.goto('/translate')
  await expect(page.locator('.translate-page')).toBeVisible()
  const settingsModal = page.locator('.settings-modal-wrapper')
  if (!(await settingsModal.isVisible())) {
    const guideConfigureButton = page.getByRole('button', { name: '立即配置' })
    if (await guideConfigureButton.isVisible()) {
      await guideConfigureButton.click()
    } else {
      await page.getByRole('button', { name: '设置', exact: true }).click()
    }
  }
  await expect(settingsModal).toBeVisible()
  await page.getByRole('tab', { name: '翻译服务' }).click()
  await expect(page.locator('#settingsApiKey')).toBeVisible()
  await expect(page.locator('#settingsApiKey')).toHaveCSS('border-radius', '6px')
  await expect(settingsModal).toHaveScreenshot('translation-settings-modal.png', {
    animations: 'disabled',
  })
})

test('translation settings modal keeps every tab layout contract', async ({ page }) => {
  await page.goto('/translate')
  await page.getByRole('button', { name: '设置', exact: true }).click()
  const settingsModal = page.locator('.settings-modal-wrapper')
  await expect(settingsModal).toBeVisible()

  const settingsTabs = [
    { label: 'OCR识别', snapshot: 'settings-tab-ocr.png' },
    { label: '翻译服务', snapshot: 'settings-tab-translation.png' },
    { label: '检测设置', snapshot: 'settings-tab-detection.png' },
    { label: '高质量翻译', snapshot: 'settings-tab-hq.png' },
    { label: 'AI校对', snapshot: 'settings-tab-proofreading.png' },
    { label: '提示词管理', snapshot: 'settings-tab-prompt-library.png' },
    { label: '插件管理', snapshot: 'settings-tab-plugins.png' },
    { label: '文本默认值', snapshot: 'settings-tab-text-defaults.png' },
    { label: '更多', snapshot: 'settings-tab-more.png' },
  ]

  for (const tab of settingsTabs) {
    await settingsModal.getByRole('tab', { name: tab.label }).click()
    await expect(settingsModal.locator('[role="tab"][aria-selected="true"]')).toHaveText(tab.label)
    await settingsModal.evaluate(element => {
      element.querySelector<HTMLElement>('[data-testid="base-dialog-body"]')?.scrollTo(0, 0)
    })
    await expect(settingsModal).toHaveScreenshot(tab.snapshot, {
      animations: 'disabled',
      maxDiffPixelRatio: 0.021,
    })
  }
})

test('web import expanded settings keep modal layout contract', async ({ page }) => {
  await page.goto('/translate')
  await page.getByRole('button', { name: '从网页导入漫画图片' }).click()
  const webImportModal = page.locator('.web-import-modal')
  await expect(webImportModal).toBeVisible()
  await webImportModal.getByRole('button', { name: '网页导入设置' }).click()
  await webImportModal.getByRole('tab', { name: /高级设置/ }).click()
  await expect(webImportModal.locator('.product-collapsible-section__body')).toBeVisible()
  await expect(webImportModal.locator('[role="tab"][aria-selected="true"]')).toHaveText('高级设置')
  await webImportModal.evaluate(element => {
    element.scrollTop = 0
    element.querySelector<HTMLElement>('[data-testid="base-dialog-body"]')?.scrollTo(0, 0)
  })
  await expect(webImportModal).toHaveScreenshot('web-import-expanded-settings.png', {
    animations: 'disabled',
  })
})

test('web import result and logs keep modal layout contract', async ({ page }) => {
  await page.goto('/translate')
  await page.getByRole('button', { name: '从网页导入漫画图片' }).click()
  const webImportModal = page.locator('.web-import-modal')
  await expect(webImportModal).toBeVisible()
  await webImportModal.locator('#webImportSourceUrl').fill('https://example.com/chapter-1')
  await webImportModal.getByRole('button', { name: /开始提取/ }).click()

  await expect(webImportModal.locator('.product-log-panel')).toBeVisible()
  await expect(webImportModal.locator('.web-import-results-grid__section')).toBeVisible()
  await expect(webImportModal.locator('.product-selectable-image-grid')).toBeVisible()
  await expect(webImportModal.getByLabel('网页导入结果元信息')).toContainText('共 2 张')
  await expect(webImportModal).toHaveScreenshot('web-import-result-logs.png', {
    animations: 'disabled',
  })
})

test('plugin agent modal keeps three-column settings layout contract', async ({ page }) => {
  await page.goto('/translate')
  await page.getByRole('button', { name: '设置', exact: true }).click()
  const settingsModal = page.locator('.settings-modal-wrapper')
  await expect(settingsModal).toBeVisible()
  await settingsModal.getByRole('tab', { name: '插件管理' }).click()
  await settingsModal.getByRole('button', { name: '自动生成插件' }).click()

  const pluginAgentModal = page.locator('.plugin-agent-modal')
  await expect(pluginAgentModal).toBeVisible()
  await expect(pluginAgentModal.locator('.plugin-agent-column')).toHaveCount(3)
  await expect(pluginAgentModal).toHaveScreenshot('plugin-agent-modal.png', {
    animations: 'disabled',
  })
})

test('insight empty state keeps its layout contract', async ({ page }) => {
  await page.goto('/insight')
  await expect(page.locator('.insight-page')).toBeVisible()
  await expect(page).toHaveScreenshot('insight-empty.png', {
    fullPage: true,
    animations: 'disabled',
  })
})

test('insight selected-book sidebars keep their gutter contract', async ({ page }) => {
  await page.goto('/insight?book=demo-book')
  await expect(page.locator('.analysis-progress-panel')).toBeVisible()
  await expect(page.locator('.page-detail-panel')).toBeVisible()
  await expect(page.locator('.notes-panel')).toBeVisible()
  await expect(page.locator('.analysis-progress-panel')).toHaveCSS('padding-left', '16px')
  await expect(page.locator('.page-detail-panel')).toHaveCSS('padding-left', '18px')
  await expect(page.locator('.notes-panel')).toHaveCSS('padding-left', '18px')
  await expect(page.locator('.product-tabbed-workspace__tab').nth(1)).toHaveCSS(
    'color',
    'rgb(102, 102, 102)'
  )
  await expect(page.locator('.overview-panel__card--stats .overview-panel__card-title')).toHaveCSS(
    'color',
    'rgb(51, 51, 51)'
  )
  await expect(page.locator('.page-detail-panel .product-section-header__title')).toHaveCSS(
    'color',
    'rgb(51, 51, 51)'
  )
  await expect(page.getByText('这是用于视觉回归的无剧透概览。')).toBeVisible()
  await expect(page.locator('.overview-panel__recent-page-card')).toHaveCount(2)
  await expect(page.locator('.overview-panel')).toHaveScreenshot('insight-overview-populated.png', {
    animations: 'disabled',
  })
  await expect(page).toHaveScreenshot('insight-selected-sidebars.png', {
    fullPage: true,
    animations: 'disabled',
    maxDiffPixelRatio: 0.021,
  })
})

test('insight overview action buttons keep their component styling', async ({ page }) => {
  await page.goto('/insight?book=demo-book')
  await expect(page.locator('.overview-panel__card--stats')).toBeVisible()
  const exportActions = page.getByRole('group', { name: '概览导出操作' })
  const currentExportButton = exportActions.getByRole('button', { name: '导出当前' })
  const allExportButton = exportActions.getByRole('button', { name: '导出全部' })
  await expect(exportActions).toBeVisible()
  await expect(currentExportButton).toBeEnabled()
  await expect(currentExportButton).toHaveCSS('border-radius', '8px')
  await expect(currentExportButton).toHaveCSS('padding-left', '12px')
  await expect(currentExportButton).toHaveCSS('border-top-width', '0px')
  await expect(allExportButton).toHaveCSS('border-radius', '8px')
  await expect(allExportButton).toHaveCSS('padding-left', '12px')
  await expect(allExportButton).toHaveCSS('color', 'rgb(255, 255, 255)')
})

test('insight notes add modal keeps form spacing contract', async ({ page }) => {
  await page.goto('/insight?book=demo-book')
  await expect(page.locator('.notes-panel')).toBeVisible()
  await page.getByRole('button', { name: '添加笔记' }).click()

  const noteModal = page.locator('.notes-panel-modal').filter({ hasText: '添加笔记' })
  await expect(noteModal).toBeVisible()
  await expect(noteModal.getByLabel('标题')).toHaveCSS('border-radius', '6px')
  await expect(noteModal).toHaveScreenshot('insight-notes-add-modal.png', {
    animations: 'disabled',
  })
})

test('insight QA save-note modal keeps nested form styling contract', async ({ page }) => {
  await page.goto('/insight?book=demo-book')
  await page.getByRole('tab', { name: /智能问答/ }).click()
  await page.getByLabel('输入你的问题...').fill('这个章节讲了什么？')
  await page.getByRole('button', { name: '发送' }).click()

  await expect(page.getByText('这是一个用于视觉回归的回答')).toBeVisible()
  await page.getByRole('button', { name: '保存为笔记' }).click()
  const qaNoteModal = page.locator('.qa-note-modal')
  await expect(qaNoteModal).toBeVisible()
  await expect(page.locator('#qaNoteTitle')).toHaveCSS('border-radius', '6px')
  await expect(qaNoteModal).toHaveScreenshot('insight-qa-save-note-modal.png', {
    animations: 'disabled',
  })
})

test('insight continuation add-character dialog keeps field layout contract', async ({ page }) => {
  await page.goto('/insight?book=demo-book')
  await page.getByRole('tab', { name: /续写/ }).click()
  await expect(page.getByText('续写设置')).toBeVisible()
  await page.getByRole('button', { name: /新增角色/ }).click()

  const addCharacterDialog = page
    .locator('.continuation-dialog-modal')
    .filter({ hasText: '新增角色' })
  await expect(addCharacterDialog).toBeVisible()
  await expect(addCharacterDialog.locator('.continuation-dialog-form')).toHaveCSS('gap', '16px')
  await expect(addCharacterDialog.locator('.continuation-dialog-field').first()).toHaveCSS(
    'margin-bottom',
    '0px'
  )
  await expect(addCharacterDialog).toHaveScreenshot(
    'insight-continuation-add-character-dialog.png',
    {
      animations: 'disabled',
    }
  )
})

test('reader loaded state keeps its layout contract', async ({ page }) => {
  await page.goto('/reader?book=demo-book&chapter=demo-chapter')
  await expect(page.locator('.reader-page')).toBeVisible()
  await expect(page.getByRole('button', { name: /打开任务中心/ })).toHaveCount(0)
  await expect(page.locator('.reader-canvas__stream .virtual-page-stream__image')).toHaveCount(2)
  await expect(page.locator('.reader-header__book-title')).toHaveCSS('color', 'rgb(255, 255, 255)')
  await expect(page.locator('.reader-header__mode-button.product-header-action--active')).toHaveCSS(
    'color',
    'rgb(102, 126, 234)'
  )
  await expect(page).toHaveScreenshot('reader-loaded.png', {
    fullPage: true,
    animations: 'disabled',
  })
})

test('reader settings panel keeps its control layout contract', async ({ page }) => {
  await page.goto('/reader?book=demo-book&chapter=demo-chapter')
  await expect(page.locator('.reader-page')).toBeVisible()
  await page.getByRole('button', { name: '阅读设置' }).click()
  const settingsPanel = page.locator('.reader-controls__settings-panel')
  await expect(settingsPanel).toBeVisible()
  await expect(settingsPanel.locator('.reader-controls__setting-field')).toHaveCount(3)
  await expect(settingsPanel).toHaveScreenshot('reader-settings-panel.png', {
    animations: 'disabled',
  })

  const stream = page.locator('.reader-canvas__stream')
  const initialWidth = (await stream.boundingBox())?.width ?? 0
  await page.getByRole('slider', { name: '图片宽度' }).fill('70')
  await page.getByRole('slider', { name: '图片间距' }).fill('24')
  await page.getByRole('button', { name: '白色' }).click()

  await expect
    .poll(async () => (await stream.boundingBox())?.width ?? 0)
    .toBeLessThan(initialWidth - 100)
  await expect(page.locator('.virtual-page-stream__page').first()).toHaveCSS(
    'margin-bottom',
    '24px'
  )
  await expect(page.locator('.reader-canvas')).toHaveCSS('background-color', 'rgb(255, 255, 255)')
})

test('mobile translate loaded workspace keeps responsive layout contract', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/translate?book=demo-book&chapter=demo-chapter')
  await expect(page.locator('.translate-page')).toBeVisible()
  await expect(page.getByTestId('translation-result-display')).toBeVisible()
  await expect(page).toHaveScreenshot('mobile-translate-loaded.png', {
    fullPage: true,
    animations: 'disabled',
    maxDiffPixelRatio: 0.05,
  })
})

test('mobile insight selected book keeps responsive layout contract', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/insight?book=demo-book')
  await expect(page.locator('.insight-page')).toBeVisible()
  await expect(page.getByLabel('打开导航')).toBeVisible()
  await page.getByLabel('打开导航').click()
  await expect(page.locator('.analysis-progress-panel')).toBeVisible()
  await expect(page).toHaveScreenshot('mobile-insight-selected.png', {
    fullPage: true,
    animations: 'disabled',
  })
})

test('narrow insight continuation wizard keeps controls inside the scroll owner', async ({
  page,
}) => {
  await page.setViewportSize({ width: 820, height: 844 })
  await page.goto('/insight?book=demo-book')
  await page.getByRole('tab', { name: /续写/ }).click()
  await expect(page.getByText('续写设置')).toBeVisible()

  const layout = await page.locator('.continuation-panel').evaluate(element => {
    const panel = element as HTMLElement
    const scrollOwner = panel.closest('.product-workspace-panel__scroll') as HTMLElement | null
    const stepRows = new Set(
      Array.from(panel.querySelectorAll<HTMLElement>('.product-wizard-steps__step')).map(step =>
        Math.round(step.getBoundingClientRect().top)
      )
    )

    return {
      panelClientWidth: panel.clientWidth,
      panelScrollWidth: panel.scrollWidth,
      scrollOwnerClientWidth: scrollOwner?.clientWidth ?? 0,
      scrollOwnerScrollWidth: scrollOwner?.scrollWidth ?? 0,
      stepCount: panel.querySelectorAll('.product-wizard-steps__step').length,
      stepRowCount: stepRows.size,
    }
  })

  expect(layout.panelScrollWidth).toBeLessThanOrEqual(layout.panelClientWidth + 1)
  expect(layout.scrollOwnerScrollWidth).toBeLessThanOrEqual(layout.scrollOwnerClientWidth + 1)
  expect(layout.stepCount).toBe(4)
  expect(layout.stepRowCount).toBe(1)
})

test('character studio empty workspace keeps its layout contract', async ({ page }) => {
  await page.goto('/insight/character-studio?book=demo-book')
  await expect(page.locator('.studio-page')).toBeVisible()
  await expect(page).toHaveScreenshot('character-studio-empty.png', {
    fullPage: true,
    animations: 'disabled',
  })
})

test('character studio editor and preview keep split workspace contract', async ({ page }) => {
  await page.unroute('**/*')
  await prepareVisualPage(page, { bookshelfHasBook: true, studioDocuments: true })
  await page.goto('/insight/character-studio?book=demo-book')

  await expect(page.locator('.studio-page__workspace-shell')).toBeVisible()
  await expect(page.locator('.studio-editor')).toBeVisible()
  await expect(page.locator('.character-studio-preview')).toBeVisible()
  await expect(page.locator('.studio-hero-section__kicker')).toHaveText('当前角色')
  const heroKickerColor = await page
    .locator('.studio-hero-section__kicker')
    .evaluate(element => getComputedStyle(element).color)
  expectCssColorNear(heroKickerColor, [111, 132, 162])
  await expect(page.getByText('绫濑澪').first()).toBeVisible()
  await expect(page.locator('.studio-hero-section')).toHaveCSS('border-radius', '28px')
  await expect(
    page.locator('.character-studio-preview .studio-preview-workspace-panel').first()
  ).toHaveCSS('border-radius', '24px')
  await expect(page).toHaveScreenshot('character-studio-editor-preview.png', {
    fullPage: true,
    animations: 'disabled',
  })
})

test('character studio panes keep independent scroll containers', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 620 })
  await page.unroute('**/*')
  await prepareVisualPage(page, { bookshelfHasBook: true, studioDocuments: true })
  await page.goto('/insight/character-studio?book=demo-book')

  const editorScroll = page.getByTestId('editor-scroll')
  const chatScroll = page.getByTestId('chat-scroll')
  await expect(editorScroll).toBeVisible()
  await expect(chatScroll).toBeVisible()
  await expect(editorScroll).toHaveCSS('overflow-y', 'auto')
  await expect(chatScroll).toHaveCSS('overflow-y', 'auto')

  await page
    .locator('.studio-editor__shell')
    .getByRole('tab', { name: '角色设定', exact: true })
    .click()
  const editorCanScroll = await editorScroll.evaluate(
    element => element.scrollHeight > element.clientHeight
  )
  expect(editorCanScroll).toBe(true)
  await editorScroll.evaluate(element => {
    element.scrollTop = 160
  })
  await expect.poll(() => editorScroll.evaluate(element => element.scrollTop)).toBeGreaterThan(0)
})
