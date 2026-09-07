import providers from './ai_provider_manifest.json'
import type { components } from '../../vue-frontend/src/api/generated/v2'
import './browserExtensionSettings.css'

type Schema = components['schemas']
export type Values = Record<string, unknown>
type Choice = readonly [string, string]
export type PluginSettingsApi = <T>(
  path: string,
  method?: string,
  body?: unknown,
) => Promise<T>

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  text = '',
  className = '',
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag)
  node.textContent = text
  node.className = className
  return node
}

function get(value: Values, path: string): unknown {
  return path
    .split('.')
    .reduce<unknown>((current, key) => (current as Values)?.[key], value)
}
function set(value: Values, path: string, next: unknown): void {
  const keys = path.split('.')
  const key = keys.pop()!
  let target = value
  for (const part of keys) target = target[part] as Values
  target[key] = next
}
export function field(
  parent: HTMLElement,
  label: string,
  data: Values,
  path: string,
  type:
    | 'text'
    | 'number'
    | 'checkbox'
    | 'password'
    | 'color'
    | 'textarea'
    | readonly Choice[] = 'text',
  change?: () => void,
): HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement {
  const wrapper = el('label', label)
  let input: HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement
  if (Array.isArray(type)) {
    input = el('select')
    for (const [value, title] of type) input.append(new Option(title, value))
    const current = String(get(data, path) ?? '')
    if (!type.some(([value]) => value === current) && current)
      input.append(new Option(current, current))
  } else if (type === 'textarea') input = el('textarea')
  else {
    input = el('input')
    input.type = type as string
    if (type === 'number') {
      input.step = 'any'
      input.min = '0'
    }
    if (type === 'password') input.autocomplete = 'new-password'
  }
  input.setAttribute('aria-label', label)
  input.value = String(get(data, path) ?? '')
  if (input instanceof HTMLInputElement && type === 'checkbox')
    input.checked = Boolean(get(data, path))
  input.addEventListener('input', () => {
    set(
      data,
      path,
      type === 'checkbox'
        ? (input as HTMLInputElement).checked
        : type === 'number'
          ? (input as HTMLInputElement).valueAsNumber
          : input.value,
    )
    change?.()
  })
  wrapper.append(input)
  parent.append(wrapper)
  return input
}
export function section(title: string, collapsed = false): HTMLElement {
  const node = el(collapsed ? 'details' : 'section')
  node.append(el(collapsed ? 'summary' : 'h2', title))
  return node
}

export function createPluginSettingsEditor(
  content: HTMLElement,
  api: PluginSettingsApi,
  message: (text: string, error?: boolean) => void,
  savingChanged: (saving: boolean) => void = () => {},
) {
  content.classList.add('browser-extension-settings-editor')
  let generation = 0
  let saveChanges: () => Promise<boolean> = async () => true
  async function persist(): Promise<boolean> {
    try {
      return await saveChanges()
    } catch (error) {
      message(error instanceof Error ? error.message : '保存失败', true)
      return false
    }
  }
  function button(
    text: string,
    action: () => Promise<unknown>,
    primary = false,
  ): HTMLButtonElement {
    const node = el('button', text, primary ? 'primary' : '')
    node.type = 'button'
    node.addEventListener('click', () => {
      node.disabled = true
      void action()
        .catch((error) =>
          message(error instanceof Error ? error.message : '操作失败', true),
        )
        .finally(() => {
          node.disabled = false
        })
    })
    return node
  }
  const providerSections = [
    ['translation', 'translation', '普通翻译', 'translation'],
    ['hqTranslation', 'hq', '高质量翻译', 'hqTranslation'],
    ['aiVisionOcr', 'ai_vision_ocr', '视觉 OCR', 'visionOcr'],
    ['browserDomAgent', 'browser_dom_agent', '网页识别 Agent', 'pluginAgent'],
  ] as const

  async function reload(): Promise<void> {
    const version = ++generation
    const [document, fonts] = await Promise.all([
      api<Schema['SettingsDocument']>(
        '/settings?domains=translation,text_style_defaults,hq,ai_vision_ocr,browser_dom_agent,ocr',
      ),
      api<Schema['FontList']>('/fonts'),
    ])
    if (version !== generation) return
    const translation = document.settings.find(
      (row) => row.domain === 'translation',
    )!
    const style = document.settings.find(
      (row) => row.domain === 'text_style_defaults',
    )!
    const original = structuredClone(document)
    const dirtyProviders = new Set<string>()
    const secrets = new Map<string, Values>()
    const providerDrafts = new Map<string, Values>()
    const form = el('form')
    form.append(
      el(
        'p',
        '插件配置单独保存，不影响翻译器默认设置。同一次网页翻译及后续图片使用开始时的配置；修改后重新开始翻译生效。',
        'hint',
      ),
    )
    const basic = section('语言与识别')
    field(basic, '目标语言', translation.payload, 'targetLanguage', [
      ['zh', '简体中文'],
      ['zh-TW', '繁体中文'],
      ['en', '英语'],
      ['ja', '日语'],
      ['ko', '韩语'],
      ['fr', '法语'],
      ['de', '德语'],
      ['es', '西班牙语'],
      ['ru', '俄语'],
    ])
    field(basic, 'OCR 引擎', translation.payload, 'ocrEngine', [
      ['manga_ocr', 'Manga OCR'],
      ['48px_ocr', '48px OCR'],
      ['paddleocr_vl', 'PaddleOCR-VL'],
      ['paddle_ocr', 'Paddle OCR'],
      ['baidu_ocr', '百度 OCR'],
      ['ai_vision', '视觉模型 OCR'],
    ])
    field(basic, '文本检测', translation.payload, 'textDetector', [
      ['default', 'Default'],
      ['ctd', 'CTD'],
      ['yolo', 'YSGYolo'],
    ])
    form.append(basic)
    for (const [key, domain, title, capability] of providerSections) {
      const box = section(title, true)
      const selected = translation.payload[key] as Values
      const initialProvider = selected.provider
      const detail = el('div')
      const drawProvider = () => {
        detail.replaceChildren()
        const provider = String(selected.provider)
        const identity = `${domain}:${provider}`
        const metadata = providers.find((row) => row.id === provider)
        const stored = document.providerSettings.find(
          (row) => row.domain === domain && row.provider === provider,
        )
        let draft = providerDrafts.get(identity)
        if (!draft) {
          const allowed = [
            'modelName',
            'customBaseUrl',
            'openaiOptions',
            ...(domain === 'translation' ? ['translationMode'] : []),
            ...(domain === 'hq' ? ['batchSize', 'prompt'] : []),
            ...(domain === 'ai_vision_ocr'
              ? ['prompt', 'promptMode', 'minImageSize']
              : []),
          ]
          draft = Object.fromEntries(
            allowed.map((name) => [name, structuredClone(selected[name])]),
          )
          if (provider !== initialProvider && !stored)
            Object.assign(draft, { modelName: '', customBaseUrl: '' })
          Object.assign(draft, structuredClone(stored?.payload ?? {}))
          providerDrafts.set(identity, draft)
        }
        const changed = () => {
          dirtyProviders.add(identity)
        }
        if (metadata?.requiresModel) {
          const model = field(
            detail,
            metadata.kind === 'adapter' ? '应用密钥' : '模型名称',
            draft,
            'modelName',
            metadata.kind === 'adapter' ? 'password' : 'text',
            changed,
          )
          model.setAttribute('list', identity)
          const options = el('datalist')
          options.id = identity
          detail.append(options)
          if (metadata.capabilities.includes('modelFetch'))
            detail.append(
              button('获取模型列表', async () => {
                const result = await api<Schema['ModelCatalogResponse']>(
                  '/model-catalog',
                  'POST',
                  {
                    domain,
                    provider,
                    baseUrl: draft!.customBaseUrl,
                    ...(secrets.has(identity)
                      ? { secret: secrets.get(identity) }
                      : {}),
                  },
                )
                options.replaceChildren(
                  ...result.models.map(
                    (value) => new Option(value.name, value.id),
                  ),
                )
                message(
                  `已获取 ${result.models.length} 个模型，可在模型输入框选择。`,
                )
              }),
            )
        }
        if (metadata?.kind !== 'adapter')
          field(
            detail,
            'API 地址（留空使用服务商默认地址）',
            draft,
            'customBaseUrl',
            'text',
            changed,
          )
        const credential = document.credentials.find(
          (row) => row.domain === domain && row.provider === provider,
        )
        if (metadata?.requiresApiKey) {
          const secret = secrets.get(identity) ?? {}
          const secretKey =
            domain === 'ai_vision_ocr' ? 'ai_vision_api_key' : 'api_key'
          const input = field(
            detail,
            metadata.kind === 'adapter'
              ? '应用 ID / App Key（留空保持）'
              : credential
                ? 'API Key · 已配置，留空保持'
                : 'API Key',
            secret,
            secretKey,
            'password',
            () => {
              if (String(secret[secretKey] ?? '').trim()) {
                secrets.set(identity, secret)
                changed()
              } else secrets.delete(identity)
            },
          )
          input.setAttribute(
            'placeholder',
            credential ? '已配置' : '输入 API Key',
          )
        }
        if (domain === 'translation')
          field(
            detail,
            '翻译方式',
            draft,
            'translationMode',
            [
              ['batch', '批量'],
              ['single', '逐条'],
            ],
            changed,
          )
        if (domain === 'hq')
          field(detail, '每批页数', draft, 'batchSize', 'number', changed)
        if ('prompt' in draft)
          field(detail, '提示词', draft, 'prompt', 'textarea', changed)
        if (metadata?.kind !== 'adapter') {
          field(
            detail,
            '流式请求',
            draft,
            'openaiOptions.execution.useStream',
            'checkbox',
            changed,
          )
          field(
            detail,
            '每分钟请求上限（0 不限制）',
            draft,
            'openaiOptions.execution.rpmLimit',
            'number',
            changed,
          )
          field(
            detail,
            '业务重试次数',
            draft,
            'openaiOptions.execution.businessRetries',
            'number',
            changed,
          )
        }
        detail.append(
          button('测试连接', async () => {
            const kind =
              domain === 'ai_vision_ocr'
                ? 'ai_vision_ocr'
                : domain === 'browser_dom_agent'
                  ? 'llm'
                  : provider === 'ollama' || provider === 'sakura'
                    ? provider
                    : provider === 'baidu_translate' ||
                        provider === 'youdao_translate'
                      ? provider
                      : 'ai_translate'
            const payload: Values = { domain, baseUrl: draft!.customBaseUrl }
            if (kind !== 'sakura') payload.model = draft!.modelName
            if (!['ollama', 'sakura'].includes(kind))
              payload.provider = provider
            if (secrets.has(identity)) payload.secret = secrets.get(identity)
            if (kind === 'baidu_translate' || kind === 'youdao_translate') {
              delete payload.model
              delete payload.provider
              delete payload.baseUrl
              const key =
                secrets.get(identity)?.api_key ??
                credential?.secret.api_key ??
                ''
              payload.secret =
                kind === 'baidu_translate'
                  ? { app_id: key, app_key: draft!.modelName }
                  : { app_key: key, app_secret: draft!.modelName }
            }
            const result = await api<Schema['ConnectionTestResponse']>(
              `/connection-tests/${kind}`,
              'POST',
              payload,
            )
            message(
              result.message ?? (result.success ? '连接成功' : '连接失败'),
              !result.success,
            )
          }),
        )
      }
      field(
        box,
        '服务商',
        selected,
        'provider',
        providers
          .filter((row) => row.capabilities.includes(capability))
          .map((row) => [row.id, row.label] as Choice),
        () => {
          drawProvider()
          dirtyProviders.add(`${domain}:${selected.provider}`)
        },
      )
      box.append(detail)
      drawProvider()
      form.append(box)
    }
    const baidu = section('百度 OCR 凭据与识别语言', true)
    const baiduStored = document.providerSettings.find(
      (row) => row.domain === 'ocr' && row.provider === 'baidu',
    )
    const baiduPayload = {
      ...(translation.payload.baiduOcr as Values),
      ...baiduStored?.payload,
    }
    providerDrafts.set('ocr:baidu', baiduPayload)
    const baiduChanged = () => {
      dirtyProviders.add('ocr:baidu')
    }
    field(
      baidu,
      '识别版本',
      baiduPayload,
      'version',
      [
        ['standard', '标准'],
        ['high_precision', '高精度'],
      ],
      baiduChanged,
    )
    field(
      baidu,
      '原文语言',
      baiduPayload,
      'sourceLanguage',
      [
        ['JAP', '日语'],
        ['ENG', '英语'],
        ['CHN_ENG', '中英'],
        ['KOR', '韩语'],
        ['auto_detect', '自动'],
      ],
      baiduChanged,
    )
    const baiduSecret: Values = {}
    for (const [key, title] of [
      ['baidu_api_key', 'API Key'],
      ['baidu_secret_key', 'Secret Key'],
    ]) {
      field(
        baidu,
        `${title}（已配置时留空保持）`,
        baiduSecret,
        key!,
        'password',
        () => {
          if (
            Object.values(baiduSecret).some((value) => String(value).trim())
          ) {
            secrets.set('ocr:baidu', baiduSecret)
            baiduChanged()
          } else secrets.delete('ocr:baidu')
        },
      )
    }
    form.append(baidu)
    const layout = section('字体与排版', true)
    field(
      layout,
      '字体',
      style.payload,
      'fontFamily',
      fonts.items.map((font) => [font.id, font.displayName] as Choice),
    )
    field(layout, '自动字号', style.payload, 'autoFontSize', 'checkbox')
    field(layout, '字号', style.payload, 'fontSize', 'number')
    field(layout, '排版方向', style.payload, 'layoutDirection', [
      ['auto', '自动'],
      ['vertical', '竖排'],
      ['horizontal', '横排'],
    ])
    field(layout, '描边', style.payload, 'strokeEnabled', 'checkbox')
    field(layout, '描边宽度', style.payload, 'strokeWidth', 'number')
    field(layout, '描边颜色', style.payload, 'strokeColor', 'color')
    field(layout, '文字颜色', style.payload, 'textColor', 'color')
    field(layout, '自动文字颜色', style.payload, 'useAutoTextColor', 'checkbox')
    field(layout, '修复方式', style.payload, 'inpaintMethod', [
      ['solid', '纯色填充'],
      ['lama_mpe', 'LaMA MPE'],
      ['litelama', 'LiteLaMA'],
    ])
    field(layout, '填充颜色', style.payload, 'fillColor', 'color')
    field(layout, '行间距', style.payload, 'lineSpacing', 'number')
    form.append(layout)
    const advanced = section('更多翻译设置', true)
    field(
      advanced,
      '并行翻译',
      translation.payload,
      'parallel.enabled',
      'checkbox',
    )
    field(
      advanced,
      '深度学习并发数',
      translation.payload,
      'parallel.deepLearningLockSize',
      'number',
    )
    field(
      advanced,
      '普通翻译提示词',
      translation.payload,
      'translation.batchNormalPrompt',
      'textarea',
    )
    field(
      advanced,
      'JSON 翻译提示词',
      translation.payload,
      'translation.batchJsonPrompt',
      'textarea',
    )
    field(
      advanced,
      '启用混合 OCR',
      translation.payload,
      'hybridOcr.enabled',
      'checkbox',
    )
    field(
      advanced,
      '最小文本块面积百分比',
      translation.payload,
      'minTextBlockAreaPercent',
      'number',
    )
    field(
      advanced,
      '启用辅助 YOLO 检测',
      translation.payload,
      'enableAuxYoloDetection',
      'checkbox',
    )
    field(
      advanced,
      '启用 SaberYOLO 精细检测',
      translation.payload,
      'enableSaberYoloRefine',
      'checkbox',
    )
    form.append(advanced)
    const save = el('div', '', 'save')
    saveChanges = async () => {
      if (!form.reportValidity()) return false
      const transaction: Schema['SettingsTransaction'] = {
        settings: [],
        providerSettings: [],
        credentialEdits: [],
      }
      for (const identity of dirtyProviders) {
        const [domain, provider] = identity.split(':') as [string, string]
        const stored = document.providerSettings.find(
          (row) => row.domain === domain && row.provider === provider,
        )
        const credential = document.credentials.find(
          (row) => row.domain === domain && row.provider === provider,
        )
        const editedSecret = secrets.get(identity)
        const secret = editedSecret
          ? {
              ...credential?.secret,
              ...Object.fromEntries(
                Object.entries(editedSecret).filter(([, value]) =>
                  String(value).trim(),
                ),
              ),
            }
          : undefined
        const credentialVersionId =
          stored?.credentialVersionId ?? credential?.credentialVersionId
        if (secret)
          transaction.credentialEdits!.push({
            domain,
            provider,
            secret,
            clientRef: identity,
            baseRevision: credential?.revision ?? 0,
            ...(credential ? { credentialId: credential.credentialId } : {}),
          })
        transaction.providerSettings!.push({
          domain,
          provider,
          payload: providerDrafts.get(identity)!,
          schemaVersion: 1,
          baseRevision: stored?.revision ?? 0,
          ...(secret
            ? { credentialEditRef: identity }
            : credentialVersionId
              ? { credentialVersionId }
              : {}),
        })
      }
      for (const entry of document.settings) {
        if (
          JSON.stringify(entry.payload) !==
          JSON.stringify(
            original.settings.find((row) => row.domain === entry.domain)
              ?.payload,
          )
        ) {
          transaction.settings!.push({
            domain: entry.domain,
            payload: entry.payload,
            baseRevision: entry.revision,
            schemaVersion: entry.schemaVersion,
          })
        }
      }
      if (
        !transaction.settings!.length &&
        !transaction.providerSettings!.length
      ) {
        message('设置没有变化。')
        return true
      }
      form.inert = true
      savingChanged(true)
      try {
        await api('/settings/transactions', 'PUT', transaction)
        await reload()
        message('插件配置已保存。重新开始翻译时使用新配置。')
        return true
      } finally {
        savingChanged(false)
        form.inert = false
      }
    }
    save.append(
      button('保存插件配置', persist, true),
      button('重新读取', reload),
    )
    form.append(save)
    form.addEventListener('submit', (event) => event.preventDefault())
    content.replaceChildren(form)
    message('')
  }

  return {
    load: reload,
    save: persist,
    dispose: () => {
      generation += 1
    },
  }
}
