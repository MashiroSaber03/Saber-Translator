import { expect, test, type Page, type Route } from '@playwright/test'

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

  await expect.poll(async () => {
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
  }).toMatchObject({ isDarkStable: true })

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

const demoPageImage = 'data:image/svg+xml;utf8,' + encodeURIComponent(demoPageSvg)
const demoPageImageBase64 = Buffer.from(demoPageSvg).toString('base64')
let demoRenderImageBase64: string | null = null

async function getDemoRenderImageBase64(page: Page): Promise<string> {
  if (demoRenderImageBase64) {
    return demoRenderImageBase64
  }

  demoRenderImageBase64 = await page.evaluate(async (svgText) => {
    const image = new Image()
    image.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svgText)
    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve()
      image.onerror = () => reject(new Error('Failed to rasterize demo render fixture'))
    })

    const canvas = document.createElement('canvas')
    canvas.width = 900
    canvas.height = 1280
    const context = canvas.getContext('2d')
    if (!context) throw new Error('Canvas 2D context unavailable')
    context.drawImage(image, 0, 0)
    return canvas.toDataURL('image/png').replace(/^data:image\/png;base64,/, '')
  }, demoPageSvg)

  return demoRenderImageBase64
}

const demoBook = {
  id: 'demo-book',
  title: 'Demo Manga',
  description: 'Visual regression fixture',
  cover: demoPageImage,
  total_pages: 20,
  tags: [],
  chapters: [
    {
      id: 'demo-chapter',
      title: 'Chapter 1',
      page_count: 20,
      image_count: 2,
      session_path: 'bookshelf/demo-book/chapters/demo-chapter/session',
      created_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-01T00:00:00Z',
    },
  ],
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
}

const demoBubbleState = {
  originalText: '原文',
  translatedText: '测试译文',
  textboxText: '',
  coords: [150, 580, 410, 710],
  polygon: [],
  fontSize: 28,
  fontFamily: 'fonts/STXIHEI.TTF',
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
  textlines: [],
  ocrResult: null,
}

const demoStudioSummary = {
  id: 'demo-doc',
  title: '绫濑澪',
  origin: 'analysis',
  source_character: '绫濑澪',
  updated_at: '2026-01-01T00:00:00Z',
  tags: ['主角', '视觉回归'],
  is_favorite: true,
  has_avatar: false,
  sample_pages: [1, 2],
}

const demoStudioDocument = {
  id: 'demo-doc',
  bookId: 'demo-book',
  origin: {
    type: 'analysis',
    source_character: '绫濑澪',
    source_pages: [1, 2],
  },
  status: {
    is_favorite: true,
    frozen_sections: ['identity'],
    last_validated_at: '2026-01-01T00:00:00Z',
  },
  meta: {
    title: '绫濑澪',
    tags: ['主角', '视觉回归'],
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
  },
  avatar: {
    mode: 'none',
    asset_path: null,
    source_page: null,
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
  chatPreset: {
    opening_mode: 'first_message',
  },
  grounding: {
    timeline_mode: 'enhanced',
    sample_pages: [1, 2],
    relationships: [{ target: '主角团', relation: '协作调查' }],
    key_moments: [{ page: 1, summary: '发现旧校舍异常灯光' }],
  },
  exportArtifacts: {
    last_review: {
      summary: '角色设定完整，已有可用的世界书与运行时资源。',
      issues: ['备用问候仍可继续丰富场景差异。'],
      suggestions: ['可以补充与学生会成员的关系条目。'],
    },
  },
}

const demoStudioChatSession = {
  session_id: 'demo-session',
  doc_id: 'demo-doc',
  title: '视觉回归会话',
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
  archived_at: null,
  greeting_source: { type: 'first_message' },
  summary_blocks: [
    {
      summary_id: 'summary-1',
      content: '用户和角色正在排查旧校舍灯光异常。',
      created_at: '2026-01-01T00:00:00Z',
      covered_message_ids: ['msg-user-1', 'msg-assistant-1'],
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
  last_prompt_preview: '角色: 绫濑澪\n场景: 校园祭前夜\n目标: 调查旧校舍异常',
}

interface VisualFixtureOptions {
  books?: typeof demoBook[]
  studioDocuments?: boolean
  editBubbles?: boolean
}

async function mockApi(route: Route, options: VisualFixtureOptions = {}, renderImageBase64 = demoPageImageBase64) {
  const requestUrl = new URL(route.request().url())
  const path = requestUrl.pathname

  if (path === '/api/server-info') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({ success: true, lan_url: 'http://192.168.1.100:5000' }),
    })
    return
  }

  if (path === '/api/get_settings') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({ success: true, settings: {} }),
    })
    return
  }

  if (path === '/api/config/translate-workflow-preferences') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({
        success: true,
        preferences: {
          rememberWorkflowMode: true,
          defaultWorkflowMode: 'upload',
        },
      }),
    })
    return
  }

  if (path === '/api/config/text-style-defaults') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({
        success: true,
        defaults: {
          fontSize: 26,
          autoFontSize: true,
          fontFamily: 'fonts/思源黑体SourceHanSansK-Bold.TTF',
          layoutDirection: 'auto',
          textColor: '#000000',
          fillColor: '#ffffff',
          strokeEnabled: true,
          strokeColor: '#ffffff',
          strokeWidth: 3,
          inpaintMethod: 'solid',
          useAutoTextColor: false,
          lineSpacing: 1,
          textAlign: 'start',
        },
      }),
    })
    return
  }

  if (path === '/api/web-import/settings') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({ success: true, hasStoredSettings: false }),
    })
    return
  }

  if (path === '/api/web-import/check-support') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({ available: true, supported: true }),
    })
    return
  }

  if (path === '/api/web-import/extract') {
    const log = {
      timestamp: '2026-01-01T00:00:00Z',
      type: 'info',
      message: '已识别漫画页面并生成导入列表',
    }
    const result = {
      success: true,
      comicTitle: 'Demo Web Comic',
      chapterTitle: 'Chapter 1',
      totalPages: 2,
      sourceUrl: 'https://example.com/chapter-1',
      referer: 'https://example.com/chapter-1',
      engine: 'ai-agent',
      pages: [
        { pageNumber: 1, imageUrl: demoPageImage },
        { pageNumber: 2, imageUrl: demoPageImage },
      ],
    }
    await route.fulfill({
      headers: { 'content-type': 'text/event-stream; charset=utf-8' },
      body: [
        `event: log\ndata: ${JSON.stringify(log)}\n\n`,
        `event: result\ndata: ${JSON.stringify(result)}\n\n`,
      ].join(''),
    })
    return
  }

  if (path === '/api/bookshelf/books') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({ success: true, books: options.books ?? [] }),
    })
    return
  }

  if (path === '/api/bookshelf/books/demo-book') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({ success: true, book: demoBook }),
    })
    return
  }

  if (path === '/api/bookshelf/books/demo-book/chapters/demo-chapter/images') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({
        success: true,
        images: [
          { index: 0, original: demoPageImage, translated: demoPageImage, fileName: '001.svg' },
          { index: 1, original: demoPageImage, translated: demoPageImage, fileName: '002.svg' },
        ],
      }),
    })
    return
  }

  if (path === '/api/parallel/render') {
    const requestBody = route.request().postDataJSON() as { bubble_states?: unknown[] } | null
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({
        success: true,
        final_image: renderImageBase64,
        bubble_states: requestBody?.bubble_states ?? [],
      }),
    })
    return
  }

  if (path === '/api/sessions/load_by_path') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({
        success: true,
        session_data: {
          name: 'demo-chapter',
          version: '1.0',
          savedAt: '2026-01-01T00:00:00Z',
          imageCount: 2,
          ui_settings: {},
          currentImageIndex: 0,
          images: [
            {
              originalDataURL: demoPageImage,
              translatedDataURL: demoPageImage,
              cleanImageData: options.editBubbles ? demoPageImageBase64 : undefined,
              fileName: '001.svg',
              translationStatus: 'completed',
              bubbleStates: options.editBubbles ? [demoBubbleState] : undefined,
            },
            {
              originalDataURL: demoPageImage,
              translatedDataURL: demoPageImage,
              fileName: '002.svg',
              translationStatus: 'completed',
            },
          ],
        },
      }),
    })
    return
  }

  if (path === '/api/bookshelf/tags') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({ success: true, tags: [] }),
    })
    return
  }

  if (path === '/api/plugins') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({
        success: true,
        plugins: [
          {
            id: 'demo-plugin',
            name: 'demo-plugin',
            display_name: 'Demo Plugin',
            version: '1.0.0',
            description: 'Visual fixture plugin',
            enabled: true,
            supported_steps: ['translate'],
            supported_modes: ['manual'],
            has_config: true,
          },
        ],
      }),
    })
    return
  }

  if (path === '/api/plugins/default_states') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({ success: true, default_states: { 'demo-plugin': true } }),
    })
    return
  }

  if (path === '/api/plugins/agent/settings') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({
        success: true,
        overview: ['描述插件目标', '确认方案', '执行并验证'],
        overview_sections: [
          { title: '工作流程', items: ['描述需求', '锁定目标', '运行验证'] },
        ],
        prompt_examples: ['创建一个翻译前处理插件'],
        providers: [
          { value: 'siliconflow', label: 'SiliconFlow' },
          { value: 'openai', label: 'OpenAI' },
        ],
        plugins: [],
        session: null,
      }),
    })
    return
  }

  if (path === '/api/manga-insight/config') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({ success: true, config: {} }),
    })
    return
  }

  if (
    path.startsWith('/api/manga-insight/demo-book/thumbnail/') ||
    path.startsWith('/api/manga-insight/demo-book/page-image/')
  ) {
    await route.fulfill({
      headers: { 'content-type': 'image/svg+xml' },
      body: decodeURIComponent(demoPageImage.replace('data:image/svg+xml;utf8,', '')),
    })
    return
  }

  if (path === '/api/manga-insight/demo-book/analyze/status') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({
        success: true,
        status: 'idle',
        progress: { current: 0, total: 20, status: 'idle', message: '' },
        total_pages: 20,
        analyzed_pages: 0,
      }),
    })
    return
  }

  if (path === '/api/manga-insight/demo-book/chat') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({
        success: true,
        answer: '这是一个用于视觉回归的回答，包含稳定的排版和引用。',
        mode: 'precise',
        citations: [{ page: 1, score: 0.95 }],
      }),
    })
    return
  }

  if (path === '/api/manga-insight/demo-book/notes') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({ success: true, notes: [] }),
    })
    return
  }

  if (path === '/api/manga-insight/demo-book/continuation/prepare') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({
        success: true,
        ready: true,
        message: 'ready',
        story_summary_ready: true,
        timeline_ready: true,
        characters_added: 1,
        total_characters: 1,
        saved_data: {
          script: null,
          pages: [],
          config: { page_count: 12, style_reference_pages: 3, continuation_direction: '' },
          has_data: true,
        },
      }),
    })
    return
  }

  if (path === '/api/manga-insight/demo-book/continuation/characters') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({
        success: true,
        characters: [
          {
            name: 'Demo',
            aliases: ['D'],
            description: 'Visual fixture character',
            reference_image: '',
            enabled: true,
            forms: [
              {
                form_id: 'default',
                form_name: '默认',
                description: '默认形态',
                reference_image: '',
                enabled: true,
              },
            ],
          },
        ],
      }),
    })
    return
  }

  if (path === '/api/manga-insight/demo-book/character-studio/index') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({
        success: true,
        documents: options.studioDocuments ? [demoStudioSummary] : [],
        candidates: [],
        has_timeline: true,
      }),
    })
    return
  }

  if (path === '/api/manga-insight/demo-book/character-studio/documents/demo-doc') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({ success: true, document: demoStudioDocument }),
    })
    return
  }

  if (path === '/api/manga-insight/demo-book/character-studio/documents/demo-doc/chat') {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({
        success: true,
        doc_id: 'demo-doc',
        active_session: demoStudioChatSession,
        archived_sessions: [
          {
            session_id: 'archived-session',
            title: '已归档会话',
            message_count: 4,
            updated_at: '2026-01-01T00:00:00Z',
            archived_at: '2026-01-01T00:00:00Z',
            last_message_excerpt: '旧校舍的谜题仍未结束。',
          },
        ],
        available_greetings: [
          {
            greeting_id: 'first_message',
            label: '主问候',
            content: demoStudioDocument.coreMessages.first_message,
            source: { type: 'first_message' },
          },
        ],
        prompt_preview: demoStudioChatSession.last_prompt_preview,
      }),
    })
    return
  }

  if (path.endsWith('/prompts/defaults')) {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({ success: true, prompts: {} }),
    })
    return
  }

  if (path.endsWith('/prompts/library')) {
    await route.fulfill({
      headers: jsonHeaders,
      body: JSON.stringify({ success: true, library: [] }),
    })
    return
  }

  await route.fulfill({
    headers: jsonHeaders,
    body: JSON.stringify({ success: true }),
  })
}

async function prepareVisualPage(page: Page, options: VisualFixtureOptions = {}) {
  const renderImageBase64 = options.editBubbles ? await getDemoRenderImageBase64(page) : demoPageImageBase64

  await page.route('**/*', async route => {
    const requestUrl = new URL(route.request().url())
    if (requestUrl.pathname.startsWith('/api/')) {
      await mockApi(route, options, renderImageBase64)
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
}

test.beforeEach(async ({ page }) => {
  await prepareVisualPage(page)
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
  await prepareVisualPage(page, { books: [demoBook] })
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
  await prepareVisualPage(page, { books: [demoBook] })
  await page.goto('/')

  await page.getByRole('button', { name: '新建书籍' }).click()
  const createBookModal = page.locator('[data-testid="base-dialog-container"]').filter({ hasText: '新建书籍' })
  await expect(createBookModal).toBeVisible()
  await expect(createBookModal.getByLabel('书籍名称')).toHaveCSS('border-radius', '8px')
  await expect(createBookModal).toHaveScreenshot('bookshelf-create-book-modal.png', {
    animations: 'disabled',
  })
  await page.keyboard.press('Escape')

  await page.getByRole('button', { name: /管理标签/ }).click()
  const tagModal = page.locator('[data-testid="base-dialog-container"]').filter({ hasText: '标签管理' })
  await expect(tagModal).toBeVisible()
  await expect(tagModal).toHaveScreenshot('bookshelf-tag-manage-modal.png', {
    animations: 'disabled',
  })
  await page.keyboard.press('Escape')

  await page.getByText('Demo Manga').click()
  await page.getByRole('button', { name: '编辑书籍' }).click()
  const editBookModal = page.locator('[data-testid="base-dialog-container"]').filter({ hasText: '编辑书籍' })
  await expect(editBookModal).toBeVisible()
  await expect(editBookModal.getByLabel('书籍名称')).toHaveCSS('border-radius', '8px')
  await expect(editBookModal).toHaveScreenshot('bookshelf-edit-book-modal.png', {
    animations: 'disabled',
  })
})

test('book detail delete confirmation keeps BaseModal styling contract', async ({ page }) => {
  await page.unroute('**/*')
  await prepareVisualPage(page, { books: [demoBook] })
  await page.goto('/')
  await page.getByText('Demo Manga').click()
  await page.getByRole('button', { name: '删除书籍' }).click()

  const confirmModal = page.locator('.confirm-modal')
  await expect(confirmModal).toBeVisible()
  await expect(confirmModal.locator('[data-testid="base-dialog-body"]')).toHaveCSS('padding-top', '20px')
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
  await expect(page.locator('.thumbnail-sidebar').getByRole('button', { name: /选择图片 \d+:/ })).toHaveCount(2)
  await expect(page.locator('.settings-sidebar')).toHaveCSS('width', '300px')
  await expect(page.locator('.settings-sidebar')).toHaveCSS('padding-left', '20px')
  await expect(page.locator('.settings-sidebar')).toHaveCSS('padding-right', '20px')
  await expect(page.locator('.thumbnail-sidebar')).toHaveCSS('width', '230px')
  await expect(page.locator('.thumbnail-sidebar')).toHaveCSS('padding-left', '20px')
  await expect(page.locator('.thumbnail-sidebar')).toHaveCSS('padding-right', '20px')
  await expect(page.locator('.thumbnail-sidebar__title')).toHaveCSS('color', 'rgb(44, 62, 80)')
  await expect(page.locator('.translate-shell__main')).toHaveCSS('margin-left', '340px')
  await expect(page.locator('.translate-shell__main')).toHaveCSS('margin-right', '240px')
  await expect(page).toHaveScreenshot('translate-loaded.png', {
    fullPage: true,
    animations: 'disabled',
  })
})

test('translate edit workspace keeps dark editor shell contract', async ({ page }) => {
  await page.goto('/translate?book=demo-book&chapter=demo-chapter')
  await expect(page.getByTestId('translation-result-display')).toBeVisible()
  await page.getByRole('button', { name: '切换编辑模式' }).click()

  const workspace = page.locator('.edit-workspace')
  const toolbar = page.locator('.edit-toolbar')
  await expect(workspace).toBeVisible()
  await expect(toolbar).toBeVisible()
  const workspaceBackground = await workspace.evaluate(element => getComputedStyle(element).backgroundColor)
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
  const workspaceBackground = await workspace.evaluate(element => getComputedStyle(element).backgroundColor)
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
    await settingsModal.evaluate((element) => {
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
  await webImportModal.evaluate((element) => {
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
  await expect(page.locator('.product-tabbed-workspace__tab').nth(1)).toHaveCSS('color', 'rgb(102, 102, 102)')
  await expect(page.locator('.overview-panel__card--stats .overview-panel__card-title')).toHaveCSS('color', 'rgb(51, 51, 51)')
  await expect(page.locator('.page-detail-panel .product-section-header__title')).toHaveCSS('color', 'rgb(51, 51, 51)')
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
  await expect(currentExportButton).toHaveCSS('border-radius', '8px')
  await expect(currentExportButton).toHaveCSS('padding-left', '12px')
  await expect(currentExportButton).toHaveCSS('border-top-width', '1px')
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

  const addCharacterDialog = page.locator('.continuation-dialog-modal').filter({ hasText: '新增角色' })
  await expect(addCharacterDialog).toBeVisible()
  await expect(addCharacterDialog.locator('.continuation-dialog-form')).toHaveCSS('gap', '16px')
  await expect(addCharacterDialog.locator('.continuation-dialog-field').first()).toHaveCSS('margin-bottom', '0px')
  await expect(addCharacterDialog).toHaveScreenshot('insight-continuation-add-character-dialog.png', {
    animations: 'disabled',
  })
})

test('reader loaded state keeps its layout contract', async ({ page }) => {
  await page.goto('/reader?book=demo-book&chapter=demo-chapter')
  await expect(page.locator('.reader-page')).toBeVisible()
  await expect(page.locator('.reader-canvas__image')).toHaveCount(2)
  await expect(page.locator('.reader-header__book-title')).toHaveCSS('color', 'rgb(255, 255, 255)')
  await expect(page.locator('.reader-header__mode-button.product-header-action--active')).toHaveCSS('color', 'rgb(102, 126, 234)')
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

test('narrow insight continuation wizard keeps controls inside the scroll owner', async ({ page }) => {
  await page.setViewportSize({ width: 820, height: 844 })
  await page.goto('/insight?book=demo-book')
  await page.getByRole('tab', { name: /续写/ }).click()
  await expect(page.getByText('续写设置')).toBeVisible()

  const layout = await page.locator('.continuation-panel').evaluate(element => {
    const panel = element as HTMLElement
    const scrollOwner = panel.closest('.product-workspace-panel__scroll') as HTMLElement | null
    const stepRows = new Set(
      Array.from(panel.querySelectorAll<HTMLElement>('.product-wizard-steps__step')).map(step => Math.round(step.getBoundingClientRect().top)),
    )

    return {
      panelClientWidth: panel.clientWidth,
      panelScrollWidth: panel.scrollWidth,
      scrollOwnerClientWidth: scrollOwner?.clientWidth ?? 0,
      scrollOwnerScrollWidth: scrollOwner?.scrollWidth ?? 0,
      stepRowCount: stepRows.size,
    }
  })

  expect(layout.panelScrollWidth).toBeLessThanOrEqual(layout.panelClientWidth + 1)
  expect(layout.scrollOwnerScrollWidth).toBeLessThanOrEqual(layout.scrollOwnerClientWidth + 1)
  expect(layout.stepRowCount).toBeGreaterThan(1)
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
  await prepareVisualPage(page, { books: [demoBook], studioDocuments: true })
  await page.goto('/insight/character-studio?book=demo-book')

  await expect(page.locator('.studio-page__workspace-shell')).toBeVisible()
  await expect(page.locator('.studio-editor')).toBeVisible()
  await expect(page.locator('.character-studio-preview')).toBeVisible()
  await expect(page.locator('.studio-hero-section__kicker')).toHaveText('当前角色')
  const heroKickerColor = await page.locator('.studio-hero-section__kicker').evaluate(element => getComputedStyle(element).color)
  expectCssColorNear(heroKickerColor, [111, 132, 162])
  await expect(page.getByText('绫濑澪').first()).toBeVisible()
  await expect(page.locator('.studio-hero-section')).toHaveCSS('border-radius', '28px')
  await expect(page.locator('.character-studio-preview .studio-preview-workspace-panel').first()).toHaveCSS('border-radius', '24px')
  await expect(page).toHaveScreenshot('character-studio-editor-preview.png', {
    fullPage: true,
    animations: 'disabled',
  })
})

test('character studio panes keep independent scroll containers', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 620 })
  await page.unroute('**/*')
  await prepareVisualPage(page, { books: [demoBook], studioDocuments: true })
  await page.goto('/insight/character-studio?book=demo-book')

  const editorScroll = page.getByTestId('editor-scroll')
  const chatScroll = page.getByTestId('chat-scroll')
  await expect(editorScroll).toBeVisible()
  await expect(chatScroll).toBeVisible()
  await expect(editorScroll).toHaveCSS('overflow-y', 'auto')
  await expect(chatScroll).toHaveCSS('overflow-y', 'auto')

  await page.locator('.studio-editor__shell').getByRole('tab', { name: '角色设定', exact: true }).click()
  const editorCanScroll = await editorScroll.evaluate(element => element.scrollHeight > element.clientHeight)
  expect(editorCanScroll).toBe(true)
  await editorScroll.evaluate(element => { element.scrollTop = 160 })
  await expect.poll(() => editorScroll.evaluate(element => element.scrollTop)).toBeGreaterThan(0)
})
