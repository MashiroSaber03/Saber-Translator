import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const {
  getPluginsMock,
  getPluginDefaultStatesMock,
  refreshPluginsMock,
  enablePluginMock,
  disablePluginMock,
  deletePluginMock,
  exportPluginMock,
  importPluginMock,
  getPluginConfigSchemaMock,
  getPluginConfigMock,
  savePluginConfigMock,
  setPluginDefaultStateMock,
  toastSuccessMock,
  toastErrorMock,
  toastWarningMock,
  confirmProductActionMock,
} = vi.hoisted(() => ({
  getPluginsMock: vi.fn(),
  getPluginDefaultStatesMock: vi.fn(),
  refreshPluginsMock: vi.fn(),
  enablePluginMock: vi.fn(),
  disablePluginMock: vi.fn(),
  deletePluginMock: vi.fn(),
  exportPluginMock: vi.fn(),
  importPluginMock: vi.fn(),
  getPluginConfigSchemaMock: vi.fn(),
  getPluginConfigMock: vi.fn(),
  savePluginConfigMock: vi.fn(),
  setPluginDefaultStateMock: vi.fn(),
  toastSuccessMock: vi.fn(),
  toastErrorMock: vi.fn(),
  toastWarningMock: vi.fn(),
  confirmProductActionMock: vi.fn(),
}))

vi.mock('@/api/plugin', () => ({
  getPlugins: getPluginsMock,
  getPluginDefaultStates: getPluginDefaultStatesMock,
  refreshPlugins: refreshPluginsMock,
  enablePlugin: enablePluginMock,
  disablePlugin: disablePluginMock,
  deletePlugin: deletePluginMock,
  exportPlugin: exportPluginMock,
  importPlugin: importPluginMock,
  getPluginConfigSchema: getPluginConfigSchemaMock,
  getPluginConfig: getPluginConfigMock,
  savePluginConfig: savePluginConfigMock,
  setPluginDefaultState: setPluginDefaultStateMock,
}))

vi.mock('@/utils/toast', () => ({
  useToast: () => ({
    success: toastSuccessMock,
    error: toastErrorMock,
    warning: toastWarningMock,
  }),
}))

vi.mock('@/composables/useProductConfirm', () => ({
  confirmProductAction: confirmProductActionMock,
}))

vi.mock('@/components/settings/PluginAgentModal.vue', () => ({
  default: defineComponent({
    name: 'PluginAgentModal',
    props: {
      modelValue: {
        type: Boolean,
        default: false,
      },
    },
    setup(props) {
      return () => h(
        'div',
        {
          class: 'plugin-agent-modal-stub',
          'data-open': String(Boolean(props.modelValue)),
        },
        'PluginAgentModal stub',
      )
    },
  }),
}))

vi.mock('@/components/common/BaseModal.vue', () => ({
  default: defineComponent({
    name: 'BaseModal',
    props: {
      modelValue: {
        type: Boolean,
        default: false,
      },
      customClass: {
        type: String,
        default: '',
      },
      frameVariant: {
        type: String,
        default: '',
      },
      dividerVariant: {
        type: String,
        default: '',
      },
      footerTone: {
        type: String,
        default: '',
      },
    },
    emits: ['close', 'update:modelValue'],
    setup(props, { emit, slots }) {
      return () => props.modelValue
        ? h('div', {
            class: props.customClass,
            'data-testid': 'plugin-config-base-modal',
            'data-frame': props.frameVariant,
            'data-divider': props.dividerVariant,
            'data-footer-tone': props.footerTone,
          }, [
            h('button', {
              type: 'button',
              class: 'base-modal-close',
              'aria-label': '关闭',
              onClick: () => {
                emit('update:modelValue', false)
                emit('close')
              },
            }, ''),
            h('div', { class: 'base-modal-default-slot' }, slots.default ? slots.default() : []),
            h('div', { class: 'base-modal-footer-slot' }, slots.footer ? slots.footer() : []),
          ])
        : null
    },
  }),
}))

import PluginManager from '@/components/settings/PluginManager.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiField from '@/components/ui/UiField.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiSwitch from '@/components/ui/UiSwitch.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'

function makePlugin(overrides: Record<string, unknown> = {}) {
  return {
    pluginId: 'plugin_one',
    displayName: 'Plugin One',
    author: 'Tests',
    description: 'desc',
    state: 'disabled',
    defaultEnabled: false,
    runtimeEnabled: false,
    config: {},
    configRevision: 1,
    errorMessage: null,
    pluginVersionId: 'plugin-version-one',
    packageVersion: '1.0.0',
    currentRevision: 1,
    manifest: {
      schema_version: 3,
      plugin_id: 'plugin_one',
      display_name: 'Plugin One',
      package_version: '1.0.0',
      entrypoint: 'plugin:run',
      hooks: [],
      supported_steps: ['ocr'],
      supported_modes: ['standard'],
      priority: 0,
      failure_policy: 'continue',
      author: 'Tests',
      description: 'desc',
      default_enabled: false,
      config_schema: {},
    },
    configSchema: {},
    ...overrides,
  }
}

describe('PluginManager', () => {
  enableAutoUnmount(afterEach)

  beforeEach(() => {
    getPluginsMock.mockReset()
    getPluginDefaultStatesMock.mockReset()
    refreshPluginsMock.mockReset()
    enablePluginMock.mockReset()
    disablePluginMock.mockReset()
    deletePluginMock.mockReset()
    exportPluginMock.mockReset()
    importPluginMock.mockReset()
    getPluginConfigSchemaMock.mockReset()
    getPluginConfigMock.mockReset()
    savePluginConfigMock.mockReset()
    setPluginDefaultStateMock.mockReset()
    toastSuccessMock.mockReset()
    toastErrorMock.mockReset()
    toastWarningMock.mockReset()
    confirmProductActionMock.mockReset()
    confirmProductActionMock.mockResolvedValue(true)

    getPluginsMock.mockResolvedValue([makePlugin()])
    getPluginDefaultStatesMock.mockResolvedValue({
      plugin_one: false,
    })
    exportPluginMock.mockResolvedValue({
      blob: new Blob(['zip-bytes'], { type: 'application/zip' }),
      filename: 'plugin_one.zip',
    })
    importPluginMock.mockResolvedValue(undefined)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('refreshes plugins from the refresh button and warns on partial success', async () => {
    refreshPluginsMock.mockResolvedValue({
      partialSuccess: true,
      plugins: [makePlugin({
        pluginId: 'plugin_two',
        displayName: 'Plugin Two',
        description: 'updated',
        packageVersion: '2.0.0',
        runtimeEnabled: true,
        defaultEnabled: true,
      })],
      defaultStates: {
        plugin_two: true,
      },
      summary: {
        added: 1,
        reloaded: 0,
        removed: 0,
        failed: 1,
      },
      failures: [{
        plugin_name: 'broken_plugin',
        error: 'bad import',
      }],
    })

    const wrapper = mount(PluginManager)
    await flushPromises()

    const refreshButton = wrapper.findAll('button').find(
      button => button.text().includes('刷新插件')
    )
    expect(refreshButton).toBeTruthy()

    await refreshButton!.trigger('click')
    await flushPromises()

    expect(refreshPluginsMock).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain('Plugin Two')
    expect(toastWarningMock).toHaveBeenCalledWith(expect.stringContaining('部分插件刷新失败'))
  })

  it('opens the plugin agent modal from the auto-generate button', async () => {
    const wrapper = mount(PluginManager)
    await flushPromises()

    const agentButton = wrapper.findAll('button').find(
      button => button.text().includes('自动生成插件'),
    )
    expect(agentButton).toBeTruthy()

    await agentButton!.trigger('click')
    await flushPromises()

    expect(wrapper.find('.plugin-agent-modal-stub').attributes('data-open')).toBe('true')
  })

  it('keeps scoped manager colors on semantic tokens instead of raw owner palettes', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginManager.vue'), 'utf8')
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(styleBlock).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
    expect(styleBlock).not.toMatch(/var\(--color-[a-z0-9-]+,\s*var\(--[a-z0-9-]+\)\)/)
    expect(styleBlock).not.toContain('--plugin-manager-config-body-background-base')
    expect(styleBlock).toContain('--color-overlay-inverse-muted')
    expect(styleBlock).toContain('--color-action-brand')
  })

  it('renders installed-plugin loading and empty states through product status feedback', async () => {
    getPluginsMock.mockReturnValueOnce(new Promise(() => undefined))

    const loadingWrapper = mount(PluginManager)
    await flushPromises()

    const loadingState = loadingWrapper.getComponent(ProductStatusBanner)
    expect(loadingState.props()).toMatchObject({
      tone: 'info',
      role: 'status',
      iconName: 'loading',
      title: '正在加载插件',
    })
    expect(loadingWrapper.text()).toContain('正在读取已安装插件列表...')

    getPluginsMock.mockReset()
    getPluginsMock.mockResolvedValue([])
    getPluginDefaultStatesMock.mockResolvedValue({})

    const emptyWrapper = mount(PluginManager)
    await flushPromises()

    const emptyState = emptyWrapper.getComponent(ProductStatusBanner)
    expect(emptyState.props()).toMatchObject({
      tone: 'neutral',
      role: 'note',
      iconName: 'settings',
      title: '暂无已安装的插件',
    })
    expect(emptyWrapper.text()).toContain('导入插件或使用自动生成插件开始。')

    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginManager.vue'), 'utf8')
    expect(source).toContain("import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'")
    expect(source).not.toContain('loading-hint')
    expect(source).not.toContain('empty-hint')
  })

  it('exports a plugin package from the export button', async () => {
    const objectUrlSpy = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:plugin-one')
    const revokeSpy = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})

    const wrapper = mount(PluginManager)
    await flushPromises()

    const exportButton = wrapper.findAll('button').find(button => button.attributes('title') === '导出')
    expect(exportButton).toBeTruthy()

    await exportButton!.trigger('click')
    await flushPromises()

    expect(exportPluginMock).toHaveBeenCalledWith('plugin_one')
    expect(objectUrlSpy).toHaveBeenCalledTimes(1)
    expect(clickSpy).toHaveBeenCalledTimes(1)
    expect(revokeSpy).toHaveBeenCalledWith('blob:plugin-one')
    expect(toastSuccessMock).toHaveBeenCalledWith('已导出 Plugin One')

    clickSpy.mockRestore()
    revokeSpy.mockRestore()
    objectUrlSpy.mockRestore()
  })

  it('uses product record cards for installed plugin rows', async () => {
    const wrapper = mount(PluginManager)
    await flushPromises()

    const cards = wrapper.findAllComponents(ProductRecordCard)
    expect(cards).toHaveLength(1)
    expect(cards[0]!.classes()).toContain('plugin-manager__plugin-card')

    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginManager.vue'), 'utf8')
    expect(source).not.toContain('class="plugin-item"')
  })

  it('keeps plugin manager hooks under the component owner', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginManager.vue'), 'utf8')

    for (const currentHook of [
      'plugin-manager__header-actions',
      'plugin-manager__import-input',
      'plugin-manager__status',
      'plugin-manager__list',
      'plugin-manager__plugin-card',
      'plugin-manager__plugin-header',
      'plugin-manager__plugin-name',
      'plugin-manager__plugin-version',
      'plugin-manager__plugin-description',
      'plugin-manager__plugin-meta',
      'plugin-manager__plugin-controls',
      'plugin-manager__settings-hint',
      'plugin-manager__default-state-item',
      'plugin-manager__config-body',
      'plugin-manager__config-field-card',
      'plugin-manager__config-field-key',
      'plugin-manager__config-switch-row',
      'plugin-manager__config-switch-text',
    ]) {
      expect(source).toContain(currentHook)
    }

    const classTokens = [...source.matchAll(/class="([^"]+)"/g)]
      .flatMap(match => match[1]!.split(/\s+/).filter(Boolean))
    for (const legacyHook of [
      'plugin-header-actions',
      'plugin-manager-status',
      'plugin-list',
      'plugin-item-card',
      'plugin-header',
      'plugin-name',
      'plugin-version',
      'plugin-description',
      'plugin-meta',
      'plugin-controls',
      'settings-hint',
      'default-state-item',
      'plugin-config-body',
      'config-field-card',
      'config-field-key',
      'config-switch-row',
      'config-switch-text',
      'sr-only',
    ]) {
      expect(classTokens).not.toContain(legacyHook)
      expect(source).not.toMatch(new RegExp(`\\.${legacyHook}\\b`))
    }

    expect(source).not.toMatch(/\.plugin-manager \.[a-z0-9_-]+/)
  })

  it('closes plugin configuration through the product modal close action', async () => {
    getPluginsMock.mockResolvedValue([makePlugin({ configSchema: { configured: { type: 'boolean' } } })])
    getPluginConfigSchemaMock.mockResolvedValue({})
    getPluginConfigMock.mockResolvedValue({})

    const wrapper = mount(PluginManager)
    await flushPromises()

    const configButton = wrapper.find('button[aria-label="配置插件 Plugin One"]')
    expect(configButton.exists()).toBe(true)

    await configButton.trigger('click')
    await flushPromises()

    const closeButton = wrapper.find('.base-modal-close')
    expect(closeButton.element.tagName).toBe('BUTTON')
    expect(closeButton.attributes('aria-label')).toBe('关闭')
    expect(closeButton.text()).toBe('')

    await closeButton.trigger('click')
    expect(wrapper.find('.plugin-config-modal').exists()).toBe(false)

    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginManager.vue'), 'utf8')
    expect(source).not.toContain('close-btn')
  })

  it('uses the product modal shell for plugin configuration', async () => {
    getPluginsMock.mockResolvedValue([makePlugin({ configSchema: { configured: { type: 'boolean' } } })])
    getPluginConfigSchemaMock.mockResolvedValue({})
    getPluginConfigMock.mockResolvedValue({})

    const wrapper = mount(PluginManager)
    await flushPromises()

    const configButton = wrapper.find('button[aria-label="配置插件 Plugin One"]')
    expect(configButton.exists()).toBe(true)

    await configButton.trigger('click')
    await flushPromises()

    const modal = wrapper.get('[data-testid="plugin-config-base-modal"]')
    expect(modal.classes()).toContain('plugin-config-modal')
    expect(modal.attributes('data-frame')).toBe('outlined')
    expect(modal.attributes('data-divider')).toBe('soft')
    expect(modal.attributes('data-footer-tone')).toBe('muted')

    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginManager.vue'), 'utf8')
    expect(source).not.toContain('import OverlayLayer')
    expect(source).not.toContain('<OverlayLayer')
    expect(source).not.toContain('plugin-config-content')
  })

  it('uses product field primitives for plugin configuration fields', async () => {
    getPluginsMock.mockResolvedValue([makePlugin({ configSchema: { configured: { type: 'boolean' } } })])
    getPluginConfigSchemaMock.mockResolvedValue({
      retries: {
        type: 'number',
        label: 'Retries',
        description: 'Retry count',
        min: 0,
        max: 5,
      },
      endpoint: {
        type: 'text',
        label: 'Endpoint',
        description: 'Service URL',
      },
    })
    getPluginConfigMock.mockResolvedValue({
      retries: 2,
      endpoint: 'https://example.test',
    })

    const wrapper = mount(PluginManager)
    await flushPromises()

    const configButton = wrapper.find('button[aria-label="配置插件 Plugin One"]')
    expect(configButton.exists()).toBe(true)

    await configButton.trigger('click')
    await flushPromises()

    const configCards = wrapper.findAllComponents(ProductRecordCard)
      .filter(card => card.classes().includes('plugin-manager__config-field-card'))
    expect(configCards).toHaveLength(2)

    const fields = wrapper.findAllComponents(UiField)
      .filter(field => field.props('variant') === 'settings')
    expect(fields.map(field => field.props('label'))).toEqual(['Retries', 'Endpoint'])
    expect(fields.map(field => field.props('description'))).toEqual(['Retry count', 'Service URL'])

    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginManager.vue'), 'utf8')
    expect(source).not.toContain('class="config-field"')
    expect(source).not.toContain('field-description')
    expect(source).not.toContain('config-input')
    expect(source).not.toMatch(/--ui-(?:input|select)-/)
  })

  it('uses the shared nullable number primitive for plugin number configuration fields', async () => {
    getPluginsMock.mockResolvedValue([makePlugin({ configSchema: { configured: { type: 'boolean' } } })])
    getPluginConfigSchemaMock.mockResolvedValue({
      retries: {
        type: 'number',
        label: 'Retries',
        min: 0,
        max: 10,
      },
    })
    getPluginConfigMock.mockResolvedValue({
      retries: 2,
    })
    savePluginConfigMock.mockResolvedValue(undefined)

    const wrapper = mount(PluginManager)
    await flushPromises()

    const configButton = wrapper.find('button[aria-label="配置插件 Plugin One"]')
    expect(configButton.exists()).toBe(true)

    await configButton.trigger('click')
    await flushPromises()

    const numberField = wrapper.getComponent(UiNumberField)
    expect(numberField.props('modelValue')).toBe(2)
    expect(numberField.props('nullable')).toBe(true)
    expect(numberField.props('min')).toBe(0)
    expect(numberField.props('max')).toBe(10)

    numberField.vm.$emit('update:modelValue', null)
    await flushPromises()

    const saveButton = wrapper.findAll('button').find(button => button.text().includes('保存'))
    expect(saveButton).toBeTruthy()
    await saveButton!.trigger('click')
    await flushPromises()

    expect(savePluginConfigMock).toHaveBeenCalledWith('plugin_one', { retries: null })

    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginManager.vue'), 'utf8')
    expect(source).toContain('UiNumberField')
    expect(source).not.toMatch(/<UiInput\b[^\n]*\btype="number"|<UiInput\b[^\n]*\bv-model\.number/)
  })

  it('retries import with replace after conflict confirmation', async () => {
    importPluginMock
      .mockRejectedValueOnce({
        message: '插件已存在',
        status: 409,
        details: {
          pluginId: 'plugin_one',
          currentRevision: 1,
        },
      })
      .mockResolvedValueOnce(undefined)

    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    const wrapper = mount(PluginManager)
    await flushPromises()

    const fileInput = wrapper.find('input[type="file"]')
    expect(fileInput.exists()).toBe(true)

    const file = new File(['zip-bytes'], 'plugin_one.zip', { type: 'application/zip' })
    Object.defineProperty(fileInput.element, 'files', {
      value: [file],
      configurable: true,
    })
    await fileInput.trigger('change')
    await flushPromises()

    expect(importPluginMock).toHaveBeenNthCalledWith(1, file, false)
    expect(confirmProductActionMock).toHaveBeenCalledWith({
      title: '替换插件',
      message: '插件 "plugin_one" 已存在，是否替换？',
      confirmText: '替换',
      cancelText: '取消',
      tone: 'danger',
    })
    expect(confirmSpy).not.toHaveBeenCalled()
    expect(importPluginMock).toHaveBeenNthCalledWith(2, file, true)
    expect(refreshPluginsMock).toHaveBeenCalledTimes(1)
    expect(toastSuccessMock).toHaveBeenCalledWith('插件导入成功')

    confirmSpy.mockRestore()
  })

  it('imports plugin packages through the typed file-input boundary', async () => {
    importPluginMock.mockResolvedValue(undefined)

    const wrapper = mount(PluginManager)
    await flushPromises()

    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginManager.vue'), 'utf8')
    expect(source).toContain('@files-change="handleImportFiles"')
    expect(source).not.toContain('target.files')
    expect(source).not.toContain("target.value = ''")

    const file = new File(['zip-bytes'], 'plugin_two.zip', { type: 'application/zip' })
    wrapper.getComponent(UiFileInput).vm.$emit('files-change', [file])
    await flushPromises()

    expect(importPluginMock).toHaveBeenCalledWith(file, false)
    expect(refreshPluginsMock).toHaveBeenCalledTimes(1)
    expect(toastSuccessMock).toHaveBeenCalledWith('插件导入成功')
  })

  it('uses product confirmation before deleting a plugin', async () => {
    deletePluginMock.mockResolvedValue(undefined)
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    const wrapper = mount(PluginManager)
    await flushPromises()

    const deleteButton = wrapper.find('button[aria-label="删除插件 Plugin One"]')
    expect(deleteButton.exists()).toBe(true)

    await deleteButton.trigger('click')
    await flushPromises()

    expect(confirmProductActionMock).toHaveBeenCalledWith({
      title: '删除插件',
      message: '确定要删除插件 "Plugin One" 吗？',
      confirmText: '删除',
      cancelText: '取消',
      tone: 'danger',
    })
    expect(confirmSpy).not.toHaveBeenCalled()
    expect(deletePluginMock).toHaveBeenCalledWith('plugin_one')
    expect(toastSuccessMock).toHaveBeenCalledWith('插件删除成功')
  })

  it('uses icon-button primitives instead of root button skins for plugin row icon actions', async () => {
    getPluginsMock.mockResolvedValue([makePlugin({ configSchema: { configured: { type: 'boolean' } } })])

    const wrapper = mount(PluginManager)
    await flushPromises()
    const source = readFileSync(resolve(process.cwd(), 'src/components/settings/PluginManager.vue'), 'utf8')
    const rootStyle = source.match(/\.plugin-manager \{(?<body>[\s\S]*?)\n\}/)

    expect(source).toContain("import UiIconButton from '@/components/ui/UiIconButton.vue'")
    expect(rootStyle?.groups?.body ?? '').not.toMatch(/--ui-button-/)
    expect(wrapper.findAllComponents(UiIconButton).length).toBeGreaterThanOrEqual(2)
    expect(wrapper.get('button[aria-label="配置插件 Plugin One"]').getComponent(UiIconButton).props()).toMatchObject({
      label: '配置插件 Plugin One',
      variant: 'soft',
      size: 'sm',
    })
    expect(wrapper.get('button[aria-label="删除插件 Plugin One"]').getComponent(UiIconButton).props()).toMatchObject({
      label: '删除插件 Plugin One',
      variant: 'danger',
      size: 'sm',
    })
  })

  it('does not assert shared icon-button primitives through internal class names', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/pluginManager.spec.ts'), 'utf8')
    const buttonClassPrefix = 'ui-' + 'button--'
    const iconButtonClassPrefix = 'ui-' + 'icon-button--'

    expect(source).not.toContain(buttonClassPrefix)
    expect(source).not.toContain(iconButtonClassPrefix)
  })

  it('uses shared switches for plugin enabled default and boolean config controls', async () => {
    getPluginsMock.mockResolvedValue([makePlugin({ configSchema: { configured: { type: 'boolean' } } })])
    enablePluginMock.mockResolvedValue(undefined)
    setPluginDefaultStateMock.mockResolvedValue(undefined)
    getPluginConfigSchemaMock.mockResolvedValue({
      feature_enabled: {
        type: 'boolean',
        label: 'Feature enabled',
      },
    })
    getPluginConfigMock.mockResolvedValue({
      feature_enabled: false,
    })
    savePluginConfigMock.mockResolvedValue(undefined)

    const wrapper = mount(PluginManager)
    await flushPromises()

    const stateSwitches = wrapper.findAllComponents(UiSwitch)
    expect(stateSwitches.map(switchControl => switchControl.props('accessibilityLabel'))).toEqual([
      '启用插件 Plugin One',
      '开启 Plugin One 默认启用状态',
    ])

    stateSwitches[0]!.vm.$emit('change', true)
    await flushPromises()
    expect(enablePluginMock).toHaveBeenCalledWith('plugin_one')

    stateSwitches[1]!.vm.$emit('change', true)
    await flushPromises()
    expect(setPluginDefaultStateMock).toHaveBeenCalledWith('plugin_one', true)

    const configButton = wrapper.find('button[aria-label="配置插件 Plugin One"]')
    expect(configButton.exists()).toBe(true)
    await configButton.trigger('click')
    await flushPromises()

    const configSwitch = wrapper.findAllComponents(UiSwitch).at(-1)
    expect(configSwitch?.props('accessibilityLabel')).toBe('Feature enabled：启用')
    configSwitch?.vm.$emit('change', true)
    await wrapper.findAll('button').find(button => button.text() === '保存')!.trigger('click')
    await flushPromises()

    expect(savePluginConfigMock).toHaveBeenCalledWith('plugin_one', {
      feature_enabled: true,
    })
  })

  it('uses the fixed select primitive for plugin config select fields', async () => {
    getPluginsMock.mockResolvedValue([makePlugin({ configSchema: { configured: { type: 'boolean' } } })])
    getPluginConfigSchemaMock.mockResolvedValue({
      mode: {
        type: 'select',
        label: 'Mode',
        options: [
          { label: 'Fast', value: 'fast' },
          { label: 'Safe', value: 'safe' },
        ],
      },
    })
    getPluginConfigMock.mockResolvedValue({
      mode: 'fast',
    })
    savePluginConfigMock.mockResolvedValue(undefined)

    const wrapper = mount(PluginManager)
    await flushPromises()

    const configButton = wrapper.find('button[aria-label="配置插件 Plugin One"]')
    expect(configButton.exists()).toBe(true)
    await configButton.trigger('click')
    await flushPromises()

    const configSelect = wrapper.getComponent(UiSelect)
    expect(configSelect.props('modelValue')).toBe('fast')
    expect(configSelect.props('options')).toEqual([
      { label: 'Fast', value: 'fast' },
      { label: 'Safe', value: 'safe' },
    ])

    configSelect.vm.$emit('change', 'safe')
    await wrapper.findAll('button').find(button => button.text() === '保存')!.trigger('click')
    await flushPromises()

    expect(savePluginConfigMock).toHaveBeenCalledWith('plugin_one', {
      mode: 'safe',
    })
  })
})
