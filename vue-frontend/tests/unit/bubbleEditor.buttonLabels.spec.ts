import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { createPinia, setActivePinia } from 'pinia'
import BubbleEditor from '@/components/edit/BubbleEditor.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import EditColorPopover from '@/components/edit/EditColorPopover.vue'
import type { BubbleState } from '@/types/bubble'

vi.mock('@/api/v2/settings', async importOriginal => ({
  ...await importOriginal<typeof import('@/api/v2/settings')>(),
  listV2Fonts: vi.fn().mockResolvedValue([]),
}))

function makeBubble(): BubbleState {
  return {
    originalText: '原文',
    translatedText: '译文',
    textboxText: '',
    coords: [10, 20, 110, 220],
    polygon: [],
    fontSize: 24,
    fontFamily: 'fonts/STXIHEI.TTF',
    textDirection: 'vertical',
    autoTextDirection: 'vertical',
    textColor: '#000000',
    fillColor: '#FFFFFF',
    rotationAngle: 0,
    position: { x: 0, y: 0 },
    strokeEnabled: true,
    strokeColor: '#FFFFFF',
    strokeWidth: 3,
    lineSpacing: 1.2,
    inlineAlign: 'center',
    blockAlign: 'end',
    inpaintMethod: 'solid',
    textlines: [],
    ocrResult: null,
  }
}

describe('BubbleEditor button labels', () => {
  afterEach(() => vi.unstubAllGlobals())
  beforeEach(() => {
    vi.stubGlobal('ResizeObserver', class {
      observe = vi.fn()
      disconnect = vi.fn()
    })
    setActivePinia(createPinia())

    Object.defineProperty(globalThis.navigator, 'clipboard', {
      value: { writeText: vi.fn() },
      configurable: true,
    })
  })

  it('removes duplicate apply buttons and keeps only the clearer bulk style action label', () => {
    const wrapper = mount(BubbleEditor, {
      props: {
        bubble: makeBubble(),
        bubbleIndex: 0,
        isOcrLoading: false,
        isTranslateLoading: false,
      },
      global: {
        stubs: {
          UiCombobox: {
            template: '<div class="ui-combobox-stub">{{ modelValue }}</div>',
            props: {
              modelValue: {
                type: String,
                default: '',
              },
            },
          },
          JapaneseKeyboard: {
            template: '<div class="jp-keyboard-stub"></div>',
          },
        },
      },
    })

    const buttonLabels = wrapper.findAll('button').map(button => button.text().trim())

    expect(buttonLabels).not.toContain('✓ 应用文本')
    expect(buttonLabels).not.toContain('应用')
    expect(buttonLabels).toContain('样式同步到本页全部气泡')
  })

  it('uses the shorter panel titles for source and translation text', () => {
    const wrapper = mount(BubbleEditor, {
      props: {
        bubble: makeBubble(),
        bubbleIndex: 0,
        isOcrLoading: false,
        isTranslateLoading: false,
      },
      global: {
        stubs: {
          UiCombobox: {
            template: '<div class="ui-combobox-stub">{{ modelValue }}</div>',
            props: {
              modelValue: {
                type: String,
                default: '',
              },
            },
          },
          JapaneseKeyboard: {
            template: '<div class="jp-keyboard-stub"></div>',
          },
        },
      },
    })

    const columnTitles = wrapper.findAll('.bubble-editor__text-panel-title').map(title => title.text().trim())

    expect(columnTitles).toContain('漫画原文')
    expect(columnTitles).toContain('译文')
    expect(columnTitles).not.toContain('🇯🇵 日语原文')
    expect(columnTitles).not.toContain('🇨🇳 中文译文')
  })

  it('defers translated text updates until IME composition commits', async () => {
    const wrapper = mount(BubbleEditor, {
      props: {
        bubble: makeBubble(),
        bubbleIndex: 0,
        isOcrLoading: false,
        isTranslateLoading: false,
      },
      global: {
        stubs: {
          UiCombobox: {
            template: '<div class="ui-combobox-stub">{{ modelValue }}</div>',
            props: {
              modelValue: {
                type: String,
                default: '',
              },
            },
          },
          JapaneseKeyboard: {
            template: '<div class="jp-keyboard-stub"></div>',
          },
        },
      },
    })

    const translatedTextarea = wrapper.findAll('textarea')[1]
    expect(translatedTextarea).toBeDefined()

    await translatedTextarea.trigger('compositionstart')
    translatedTextarea.element.value = 'nihao'
    await translatedTextarea.trigger('input')

    expect(wrapper.emitted('update')).toBeUndefined()

    translatedTextarea.element.value = '你好'
    await translatedTextarea.trigger('compositionend')

    expect(wrapper.emitted('update')).toEqual([[{ translatedText: '你好' }]])
  })

  it('applies style to all bubbles without routine console logs', async () => {
    const wrapper = mount(BubbleEditor, {
      props: {
        bubble: makeBubble(),
        bubbleIndex: 0,
        isOcrLoading: false,
        isTranslateLoading: false,
      },
      global: {
        stubs: {
          UiCombobox: {
            template: '<div class="ui-combobox-stub">{{ modelValue }}</div>',
            props: {
              modelValue: {
                type: String,
                default: '',
              },
            },
          },
          JapaneseKeyboard: {
            template: '<div class="jp-keyboard-stub"></div>',
          },
        },
      },
    })

    const applyButton = wrapper.findAll('button')
      .find(button => button.text().trim() === '样式同步到本页全部气泡')
    expect(applyButton).toBeDefined()

    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)

    try {
      const strokeWidth = wrapper.get('input[aria-label="描边宽度"]')
      expect(strokeWidth.attributes('step')).toBe('0.1')
      await strokeWidth.setValue('1.2')
      expect(wrapper.emitted('update')?.at(-1)).toEqual([{ strokeWidth: 1.2 }])
      await applyButton?.trigger('click')
      expect(logSpy).not.toHaveBeenCalled()
      expect(wrapper.emitted('reRender')).toBeUndefined()
      expect(wrapper.emitted('applyToAllStyle')).toEqual([[
        expect.objectContaining({
          fontSize: 24,
          fontFamily: 'fonts/STXIHEI.TTF',
          textDirection: 'vertical',
          textColor: '#000000',
          fillColor: '#FFFFFF',
          strokeEnabled: true,
          strokeColor: '#FFFFFF',
          strokeWidth: 1.2,
          inpaintMethod: 'solid',
          lineSpacing: 1.2,
          inlineAlign: 'center',
          blockAlign: 'end',
        }),
      ]])
    } finally {
      logSpy.mockRestore()
    }
  })

  it('keeps durable bubble state writes outside the editor owner', () => {
    const editorSource = readFileSync(
      resolve(process.cwd(), 'src/components/edit/useBubbleEditor.ts'),
      'utf8',
    )
    const componentSource = readFileSync(
      resolve(process.cwd(), 'src/components/edit/BubbleEditor.vue'),
      'utf8',
    )

    expect(editorSource).not.toContain("from '@/stores/bubbleStore'")
    expect(editorSource).not.toContain('useBubbleStore(')
    expect(editorSource).toContain("'applyToAllStyle'")
    expect(componentSource).toContain('@click="applyToAll"')
  })

  it('uses shared button variants for editor footer actions', () => {
    const wrapper = mount(BubbleEditor, {
      props: {
        bubble: makeBubble(),
        bubbleIndex: 0,
        isOcrLoading: false,
        isTranslateLoading: false,
      },
      global: {
        stubs: {
          UiCombobox: {
            template: '<div class="ui-combobox-stub">{{ modelValue }}</div>',
            props: {
              modelValue: {
                type: String,
                default: '',
              },
            },
          },
          JapaneseKeyboard: {
            template: '<div class="jp-keyboard-stub"></div>',
          },
        },
      },
    })

    const buttons = wrapper.findAllComponents(UiButton)
    const applyButton = buttons.find(button => button.text().trim() === '样式同步到本页全部气泡')
    const resetButton = buttons.find(button => button.text().trim() === '重置')

    expect(applyButton?.props()).toMatchObject({
      variant: 'primary',
      tone: 'success',
      block: true,
    })
    expect(applyButton?.classes()).not.toContain('btn-apply-all')
    expect(resetButton).toBeUndefined()

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/BubbleEditor.vue'),
      'utf8',
    )

    expect(source).not.toMatch(/btn-(apply-all|reset)/)
    expect(source).not.toContain('bubble-editor-apply-all-button')
  })

  it('inserts kana at the start of the original text when the caret is at zero', async () => {
    const wrapper = mount(BubbleEditor, {
      props: {
        bubble: makeBubble(),
        bubbleIndex: 0,
        isOcrLoading: false,
        isTranslateLoading: false,
      },
      global: {
        stubs: {
          UiCombobox: {
            template: '<div class="ui-combobox-stub">{{ modelValue }}</div>',
            props: {
              modelValue: {
                type: String,
                default: '',
              },
            },
          },
          JapaneseKeyboard: {
            name: 'JapaneseKeyboard',
            template: '<div class="jp-keyboard-stub"></div>',
            props: {
              visible: Boolean,
              defaultTarget: String,
            },
            emits: ['insert', 'delete', 'close'],
          },
        },
      },
    })

    const originalTextarea = wrapper.findAll('textarea')[0].element as HTMLTextAreaElement
    originalTextarea.setSelectionRange(0, 0)

    wrapper.findComponent({ name: 'JapaneseKeyboard' }).vm.$emit('insert', 'あ', 'original')
    await nextTick()
    await nextTick()

    const updates = wrapper.emitted('update')
    expect(updates?.at(-1)).toEqual([{ originalText: 'あ原文' }])
    expect(originalTextarea.selectionStart).toBe(1)
  })

  it('does not delete from the end of the original text when the caret is at zero', async () => {
    const wrapper = mount(BubbleEditor, {
      props: {
        bubble: makeBubble(),
        bubbleIndex: 0,
        isOcrLoading: false,
        isTranslateLoading: false,
      },
      global: {
        stubs: {
          UiCombobox: {
            template: '<div class="ui-combobox-stub">{{ modelValue }}</div>',
            props: {
              modelValue: {
                type: String,
                default: '',
              },
            },
          },
          JapaneseKeyboard: {
            name: 'JapaneseKeyboard',
            template: '<div class="jp-keyboard-stub"></div>',
            props: {
              visible: Boolean,
              defaultTarget: String,
            },
            emits: ['insert', 'delete', 'close'],
          },
        },
      },
    })

    const originalTextarea = wrapper.findAll('textarea')[0].element as HTMLTextAreaElement
    originalTextarea.setSelectionRange(0, 0)

    wrapper.findComponent({ name: 'JapaneseKeyboard' }).vm.$emit('delete', 'original')
    await nextTick()
    await nextTick()

    const updates = wrapper.emitted('update')
    expect(updates).toBeUndefined()
    expect(originalTextarea.value).toBe('原文')
    expect(originalTextarea.selectionStart).toBe(0)
  })

  it('uses the fixed select primitive for background repair method', () => {
    const wrapper = mount(BubbleEditor, {
      props: {
        bubble: makeBubble(),
        bubbleIndex: 0,
        isOcrLoading: false,
        isTranslateLoading: false,
      },
      global: {
        stubs: {
          UiCombobox: {
            template: '<div class="ui-combobox-stub">{{ modelValue }}</div>',
            props: {
              modelValue: {
                type: String,
                default: '',
              },
            },
          },
          JapaneseKeyboard: {
            template: '<div class="jp-keyboard-stub"></div>',
          },
        },
      },
    })

    const inpaintSelect = wrapper.getComponent(UiSelect)
    expect(inpaintSelect.props('modelValue')).toBe('solid')
    expect(inpaintSelect.props('options')).toEqual(expect.arrayContaining([
      expect.objectContaining({ value: 'solid' }),
      expect.objectContaining({ value: 'lama_mpe' }),
      expect.objectContaining({ value: 'litelama' }),
    ]))
    expect(inpaintSelect.get('button').attributes('aria-label')).toBe('背景修复方式')
  })

  it('gives icon-only editing toolbar buttons explicit accessible names', () => {
    const wrapper = mount(BubbleEditor, {
      props: {
        bubble: makeBubble(),
        bubbleIndex: 0,
        isOcrLoading: false,
        isTranslateLoading: false,
      },
      global: {
        stubs: {
          UiCombobox: {
            template: '<div class="ui-combobox-stub">{{ modelValue }}</div>',
            props: {
              modelValue: {
                type: String,
                default: '',
              },
            },
          },
          JapaneseKeyboard: {
            template: '<div class="jp-keyboard-stub"></div>',
          },
        },
      },
    })

    const unlabeledIconButtons = wrapper
      .findAll('button')
      .filter((button) => button.text().trim() === '')
      .filter((button) => !button.attributes('aria-label'))
      .map((button) => button.attributes('title') || button.classes().join('.'))

    expect(unlabeledIconButtons).toEqual([])
    expect(wrapper.get('button[title="竖向排版"]').attributes('aria-label')).toBe('竖向排版')
    expect(wrapper.get('button[title="文字颜色"]').attributes('aria-label')).toBe('文字颜色')
  })

  it('renders icon-only editing tools through the shared icon-button primitive', () => {
    const wrapper = mount(BubbleEditor, {
      props: {
        bubble: makeBubble(),
        bubbleIndex: 0,
        isOcrLoading: false,
        isTranslateLoading: false,
      },
      global: {
        stubs: {
          UiCombobox: {
            template: '<div class="ui-combobox-stub">{{ modelValue }}</div>',
            props: {
              modelValue: {
                type: String,
                default: '',
              },
            },
          },
          JapaneseKeyboard: {
            template: '<div class="jp-keyboard-stub"></div>',
          },
        },
      },
    })
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/BubbleEditor.vue'),
      'utf8',
    )

    expect(source).toContain("import UiIconButton from '@/components/ui/UiIconButton.vue'")
    expect(wrapper.findAllComponents(UiIconButton).map(button => button.props('label'))).toEqual([
      '重新OCR此气泡',
      '重新翻译此气泡',
      '竖向排版',
      '横向排版',
      '文字颜色',
      '背景填充颜色',
      '文字描边',
      '描边颜色',
      '顶部对齐',
      '行内居中',
      '底部对齐',
      '文本块靠右',
      '文本块居中',
      '文本块靠左',
      '逆时针旋转',
      '顺时针旋转',
      '重置旋转',
      '左移',
      '右移',
      '上移',
      '下移',
      '重置位置',
    ])
    expect(source).not.toMatch(/<UiButton[\s\S]{0,180}bubble-editor-toolbar-action/)
    expect(source).not.toMatch(/<UiButton[\s\S]{0,140}re-ocr-btn/)
    expect(source).not.toMatch(/<UiButton[\s\S]{0,140}re-translate-btn/)
  })

  it('exposes pressed state for icon-only formatting mode buttons', () => {
    const wrapper = mount(BubbleEditor, {
      props: {
        bubble: makeBubble(),
        bubbleIndex: 0,
        isOcrLoading: false,
        isTranslateLoading: false,
      },
      global: {
        stubs: {
          UiCombobox: {
            template: '<div class="ui-combobox-stub">{{ modelValue }}</div>',
            props: {
              modelValue: {
                type: String,
                default: '',
              },
            },
          },
          JapaneseKeyboard: {
            template: '<div class="jp-keyboard-stub"></div>',
          },
        },
      },
    })

    expect(wrapper.get('button[title="竖向排版"]').attributes('aria-pressed')).toBe('true')
    expect(wrapper.get('button[title="横向排版"]').attributes('aria-pressed')).toBe('false')
    expect(wrapper.get('button[title="文字描边"]').attributes('aria-pressed')).toBe('true')
    expect(wrapper.get('button[title="行内居中"]').attributes('aria-pressed')).toBe('true')
    expect(wrapper.get('button[title="文本块靠左"]').attributes('aria-pressed')).toBe('true')
  })

  it('updates the two alignment axes independently on the selected bubble', async () => {
    const wrapper = mount(BubbleEditor, {
      props: {
        bubble: makeBubble(),
        bubbleIndex: 0,
      },
      global: {
        stubs: {
          UiCombobox: {
            template: '<div class="ui-combobox-stub"></div>',
            props: ['modelValue'],
          },
          JapaneseKeyboard: {
            template: '<div class="jp-keyboard-stub"></div>',
          },
        },
      },
    })

    await wrapper.get('button[title="顶部对齐"]').trigger('click')
    await wrapper.get('button[title="文本块居中"]').trigger('click')

    expect(wrapper.emitted('update')).toEqual([
      [{ inlineAlign: 'start' }],
      [{ blockAlign: 'center' }],
    ])
  })

  it('exposes pressed state for the selected font-size preset', () => {
    const wrapper = mount(BubbleEditor, {
      props: {
        bubble: makeBubble(),
        bubbleIndex: 0,
        isOcrLoading: false,
        isTranslateLoading: false,
      },
      global: {
        stubs: {
          UiCombobox: {
            template: '<div class="ui-combobox-stub">{{ modelValue }}</div>',
            props: {
              modelValue: {
                type: String,
                default: '',
              },
            },
          },
          JapaneseKeyboard: {
            template: '<div class="jp-keyboard-stub"></div>',
          },
        },
      },
    })

    const presets = wrapper.findAll('.bubble-editor__font-size-preset')
    const selectedPreset = presets.find(button => button.text().trim() === '24')
    const unselectedPreset = presets.find(button => button.text().trim() !== '24')

    expect(selectedPreset?.attributes('aria-pressed')).toBe('true')
    expect(unselectedPreset?.attributes('aria-pressed')).toBe('false')
  })

  it('uses the shared number field primitive for editor numeric controls', () => {
    const wrapper = mount(BubbleEditor, {
      props: {
        bubble: makeBubble(),
        bubbleIndex: 0,
        isOcrLoading: false,
        isTranslateLoading: false,
      },
      global: {
        stubs: {
          UiCombobox: {
            template: '<div class="ui-combobox-stub">{{ modelValue }}</div>',
            props: {
              modelValue: {
                type: String,
                default: '',
              },
            },
          },
          JapaneseKeyboard: {
            template: '<div class="jp-keyboard-stub"></div>',
          },
        },
      },
    })

    const numberFields = wrapper.findAllComponents(UiNumberField)

    expect(numberFields).toHaveLength(4)
    expect(numberFields.map(field => field.props('modelValue'))).toEqual(expect.arrayContaining([24, 3, 1.2, 0]))
    expect(numberFields.every(field => field.props('variant') === 'editor')).toBe(true)

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/BubbleEditor.vue'),
      'utf8',
    )

    expect(source).not.toMatch(/toolbar-fontsize-input|toolbar-mini-input|linespacing-input|toolbar-rotation-input/)
    expect(source).not.toMatch(/<UiInput\s+type="number"/)
    expect(source).not.toMatch(/--ui-input-/)
    expect(source).toMatch(/<UiNumberField[\s\S]*variant="editor"/)
  })

  it('uses editor field primitives for toolbar field labels', () => {
    const wrapper = mount(BubbleEditor, {
      props: {
        bubble: makeBubble(),
        bubbleIndex: 0,
        isOcrLoading: false,
        isTranslateLoading: false,
      },
      global: {
        stubs: {
          UiCombobox: {
            template: '<div class="ui-combobox-stub">{{ modelValue }}</div>',
            props: {
              modelValue: {
                type: String,
                default: '',
              },
            },
          },
          JapaneseKeyboard: {
            template: '<div class="jp-keyboard-stub"></div>',
          },
        },
      },
    })

    const toolbarFields = wrapper
      .findAllComponents(UiField)
      .filter(field => ['字体', '字号', '行间距'].includes(String(field.props('label'))))

    expect(toolbarFields.map(field => field.props('variant'))).toEqual(['editor', 'editor', 'editor'])
    expect(toolbarFields.map(field => field.props('label'))).toEqual(['字体', '字号', '行间距'])

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/BubbleEditor.vue'),
      'utf8',
    )
    expect(source).not.toMatch(/<label\b/)
    expect(source).not.toContain('combo-control label')
  })

  it('uses owner-specific icon hooks for text refresh loading states', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/BubbleEditor.vue'),
      'utf8',
    )

    expect(source).toContain('bubble-editor__refresh-icon')
    expect(source).not.toContain('bubble-editor-refresh-icon')
    expect(source).not.toContain('class="button-icon"')
    expect(source).not.toContain('.button-icon')
  })

  it('lets UiIconButton own icon-only editor toolbar chrome', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/BubbleEditor.vue'),
      'utf8',
    )

    expect(source).toContain("import UiIconButton from '@/components/ui/UiIconButton.vue'")
    expect(source).toContain('bubble-editor__toolbar-action')
    expect(source).toContain('--ui-icon-button-active-background')
    expect(source).not.toMatch(/:data-active=/)
    expect(source).not.toMatch(/\.(?:re-ocr-btn|re-translate-btn|bubble-editor__toolbar-action|bubble-editor-toolbar-action|bubble-editor-toolbar-small-action)[\s\S]{0,260}\b(?:width|height|border|background|cursor|transition|box-shadow|transform)\s*:/)
    expect(source).not.toMatch(/\.bubble-editor__toolbar-action:hover/)
    expect(source).not.toMatch(/\.bubble-editor__toolbar-action\[data-active/)
    expect(source).not.toContain('bubble-editor-toolbar-action')
  })

  it('uses owner-specific toolbar action hooks instead of generic button names', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/BubbleEditor.vue'),
      'utf8',
    )

    expect(source).toContain('class="bubble-editor"')
    expect(source).toContain('bubble-editor__text-panel')
    expect(source).toContain('bubble-editor__text-panel--original')
    expect(source).toContain('bubble-editor__text-panel--translated')
    expect(source).toContain('bubble-editor__toolbar')
    expect(source).toContain('bubble-editor__toolbar-row')
    expect(source).toContain('bubble-editor__toolbar-action')
    expect(source).toContain('bubble-editor__toolbar-color-action')
    expect(source).toContain('bubble-editor__refresh-action')
    expect(source).toContain('bubble-editor__style-section')
    expect(source).toContain('bubble-editor__font-size-preset')
    expect(source).toContain('bubble-editor__footer-actions')
    expect(source).not.toContain('bubble-editor-toolbar-small-action')
    expect(source).not.toContain('bubble-editor-toolbar-action')
    expect(source).not.toContain('bubble-editor-toolbar-color-action')
    expect(source).not.toContain('bubble-editor-refresh-action')
    expect(source).not.toContain('bubble-editor-style-section')
    expect(source).not.toContain('style-settings-section')
    expect(source).not.toContain('class="style-settings-section text-block"')
    expect(source).not.toMatch(/\.style-settings-section\b/)
    expect(source).not.toContain('edit-panel-content')
    expect(source).not.toMatch(/class="[^"]*\b(?:text-column|original-text-column|translated-text-column|text-block|office-toolbar|combo-control|font-control|size-control|linespacing-control|fontsize-presets-panel|font-size-presets|preset-btn|edit-action-buttons|editor-number-field)\b/)
    expect(source).not.toMatch(/\.(?:text-column-header|column-title|text-editor|original-editor|translated-editor|text-actions|text-action-btn|copy-btn|keyboard-toggle-btn|office-toolbar|combo-control|toolbar-divider|toolbar-icon-group|toolbar-color-group|toolbar-color-picker|color-indicator|toolbar-inpaint-group|toolbar-solid-color-options|toolbar-stroke-cluster|toolbar-stroke-options|toolbar-stroke-width|toolbar-unit|toolbar-rotation-group|toolbar-position-group|toolbar-position-value|fontsize-presets-panel|font-size-presets|preset-btn|edit-action-buttons|editor-number-field)\b/)
    expect(source).not.toMatch(/toolbar-(btn|color-btn|small-btn)/)
  })

  it('uses an explicit owner hook for the font-size preset disclosure title', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/BubbleEditor.vue'),
      'utf8',
    )

    expect(source).toContain('bubble-editor__font-size-presets-title')
    expect(source).not.toContain('.bubble-editor__font-size-presets-panel summary')
  })

  it('maps editor owner colors through semantic tokens', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/BubbleEditor.vue'),
      'utf8',
    )

    expect(source).not.toMatch(/#[0-9A-Fa-f]{3,8}\b|rgba?\(/)
    expect(source).not.toMatch(/var\(--color-[a-z0-9-]+,\s*var\(--[a-z0-9-]+\)\)/)
    expect(source).not.toMatch(/--bubble-editor-(text-column-divider|column-title-text|text-action-background)/)
    expect(source).not.toMatch(/<svg[\s>]/)
    expect(source).toContain('--bubble-editor-style-panel-border: color-mix')
    expect(source).toContain('--bubble-editor-translated-title-text: var(--color-surface-success)')
  })

  it('uses typed model updates and an anchored color popover for text and color controls', async () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/BubbleEditor.vue'),
      'utf8',
    )

    expect(source).toContain("import EditColorPopover from './EditColorPopover.vue'")
    expect(source).not.toContain("import UiInput from '@/components/ui/UiInput.vue'")
    expect(source).not.toMatch(/@input="handle(OriginalText|Text|TextColor)Change"/)
    expect(source).not.toMatch(/<UiInput[\s\S]*type="color"/)
    expect(source).not.toContain('hidden-color-input')

    const wrapper = mount(BubbleEditor, {
      props: {
        bubble: makeBubble(),
        bubbleIndex: 0,
        isOcrLoading: false,
        isTranslateLoading: false,
      },
      global: {
        stubs: {
          UiCombobox: {
            template: '<div class="ui-combobox-stub">{{ modelValue }}</div>',
            props: {
              modelValue: {
                type: String,
                default: '',
              },
            },
          },
          JapaneseKeyboard: {
            template: '<div class="jp-keyboard-stub"></div>',
          },
        },
      },
    })

    expect(wrapper.find('input[type="color"]').exists()).toBe(false)
    await wrapper.get('button[aria-label="文字颜色"]').trigger('click')
    wrapper.getComponent(EditColorPopover).vm.$emit('apply', 'textColor', '#123456')
    expect(wrapper.emitted('update')?.at(-1)).toEqual([{ textColor: '#123456' }])
    wrapper.unmount()
  })
})
