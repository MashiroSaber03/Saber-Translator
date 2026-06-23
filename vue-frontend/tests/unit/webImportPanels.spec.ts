import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import WebImportLogsPanel from '@/components/translate/web-import/WebImportLogsPanel.vue'
import WebImportSettingsPanel from '@/components/translate/web-import/WebImportSettingsPanel.vue'
import type { WebImportSettings } from '@/types/webImport'

function createDraftSettings(): WebImportSettings {
  return {
    firecrawl: {
      apiKey: '',
      apiUrl: '',
    },
    agent: {
      provider: 'custom',
      apiKey: '',
      customBaseUrl: '',
      modelName: '',
      forceJsonOutput: true,
      useStream: false,
    },
    extraction: {
      prompt: '',
      maxIterations: 3,
    },
    download: {
      concurrency: 3,
      timeout: 30,
      retries: 2,
      delay: 0,
      useReferer: true,
    },
    advanced: {
      customCookie: '',
      customHeaders: '',
      bypassProxy: false,
    },
    preprocess: {
      enabled: false,
      maxWidth: 0,
      outputFormat: 'jpeg',
      quality: 90,
      grayscale: false,
      sharpen: false,
    },
    ui: {
      showAgentLogs: true,
      autoImport: false,
    },
  }
}

describe('WebImport panels', () => {
  it('uses button semantics for settings disclosure', async () => {
    const wrapper = mount(WebImportSettingsPanel, {
      props: {
        activeSettingsTab: 'basic',
        agentProviderOptions: [],
        draftSettings: createDraftSettings(),
        hasUnsavedSettings: false,
        isFetchingModels: false,
        isSavingSettings: false,
        modelList: [],
        modelListOptions: [],
        providerRequiresApiKey: () => false,
        settingsExpanded: false,
        showAgentKey: false,
        showCustomUrl: false,
        showFirecrawlKey: false,
        supportsFetchModels: false,
        testingAgent: false,
        testingFirecrawl: false,
        webImportStore: {} as any,
      },
    })

    const header = wrapper.get('.web-import-modal__settings-header')
    expect(header.element.tagName).toBe('BUTTON')
    expect(header.attributes('aria-expanded')).toBe('false')

    await header.trigger('click')
    expect(wrapper.emitted('update:settingsExpanded')?.[0]).toEqual([true])
  })

  it('uses button semantics for logs disclosure', async () => {
    const wrapper = mount(WebImportLogsPanel, {
      props: {
        expanded: false,
        status: 'extracting',
        logs: [{ type: 'info', timestamp: '12:00', message: 'hello' }],
      },
    })

    const header = wrapper.get('.logs-header')
    expect(header.element.tagName).toBe('BUTTON')
    expect(header.attributes('aria-expanded')).toBe('false')

    await header.trigger('click')
    expect(wrapper.emitted('toggle')).toBeTruthy()
  })
})
