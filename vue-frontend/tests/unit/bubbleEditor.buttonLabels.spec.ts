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
import RichTextInput from '@/components/edit/RichTextInput.vue'
import { nativeColor } from '@/components/edit/richTextDocument'
import { copiedTextStyle } from '@/components/edit/textStyleClipboard'
import type { BubbleState } from '@/types/bubble'

vi.mock('@/api/client', () => ({ apiClient: { get: vi.fn().mockResolvedValue({ items: [] }) } }))

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
  it('shows mixed values for multiple frames and formats the selection through the shared batch action', async () => {
    const a = makeBubble(), b = { ...makeBubble(), fontSize: 36, translatedText: '另一个框' }
    const wrapper = mount(BubbleEditor, { props: { bubble: a, bubbleIndex: 0, selectedCount: 2, selectedBubbles: [a, b] } })
    await nextTick()
    expect(wrapper.text()).toContain('已选 2 个框')
    expect((wrapper.get('input[aria-label="字号"]').element as HTMLInputElement).value).toBe('')
    await wrapper.get('input[aria-label="字号"]').setValue('32')
    expect(wrapper.emitted('formatText')?.at(-1)).toEqual([{ size_pt: 32 }])
    expect(wrapper.emitted('update')).toBeUndefined()
    wrapper.unmount()
  })

  it('pastes only character formatting into an active text selection instead of changing the whole frame', async () => {
    const format = vi.fn()
    const wrapper = mount(BubbleEditor, { props: { bubble: makeBubble(), bubbleIndex: 0 }, global: {
      stubs: { RichTextInput: { name: 'RichTextInput', template: '<div />', methods: { format } } },
    } })
    const rich = wrapper.findComponent({ name: 'RichTextInput' })
    rich.vm.$emit('selection', { active: true, count: 2, styles: [{ size_pt: 24 }] })
    copiedTextStyle.value = { character: { size_pt: 40, underline: true }, box: { textDirection: 'horizontal' } }
    await nextTick()
    await wrapper.get('button[aria-label="粘贴样式"]').trigger('click')
    expect(format).toHaveBeenCalledWith({ size_pt: 40, underline: true })
    expect(wrapper.emitted('applyStyle')).toBeUndefined()
    copiedTextStyle.value = null
    wrapper.unmount()
  })
  it('edits character metrics through rich text and shows mixed values without resetting them', async () => {
    const bubble = makeBubble()
    bubble.translatedText = '中文'
    bubble.richText = { runs: [
      { len: 3, style: { size_pt: 24, tracking: 125, horizontal_scale: 0.8, vertical_scale: 1.1, language: 'ja' } },
      { len: 3, style: { size_pt: 30, tracking: 0, horizontal_scale: 0.8, vertical_scale: 1.1, language: 'ja' } },
    ], paragraphs: [{ len: 6, style: { align: 'Center' } }] }
    const wrapper = mount(BubbleEditor, { props: { bubble, bubbleIndex: 0 } })
    await nextTick()
    const tracking = wrapper.get('input[aria-label="字距"]')
    expect((tracking.element as HTMLInputElement).value).toBe('')
    expect(tracking.attributes('placeholder')).toBe('混合')
    expect((wrapper.get('input[aria-label="字宽"]').element as HTMLInputElement).value).toBe('80')
    await tracking.setValue('15')
    const value = wrapper.emitted('update')!.at(-1)![0] as Partial<BubbleState>
    expect(value.richText!.runs.map(run => run.style)).toEqual(bubble.richText.runs.map(run => ({ ...run.style, tracking: 150 })))
    expect(value.richText!.paragraphs).toEqual(bubble.richText.paragraphs)
    expect(value).not.toHaveProperty('coords')
    await wrapper.get('input[aria-label="字宽"]').setValue('65')
    expect((wrapper.emitted('update')!.at(-1)![0] as Partial<BubbleState>).richText!.runs[0]!.style.horizontal_scale).toBe(0.65)
    await wrapper.get('button[aria-label="重置字距和字形"]').trigger('click')
    const reset = (wrapper.emitted('update')!.at(-1)![0] as Partial<BubbleState>).richText!
    expect(reset.runs.map(run => run.style)).toEqual(bubble.richText.runs.map(run => ({
      ...run.style, tracking: 0, horizontal_scale: 1, vertical_scale: 1,
    })))
    wrapper.unmount()
  })

  it.each([
    ['字宽', 'horizontal_scale', 'ArrowUp', 1.01],
    ['字宽', 'horizontal_scale', 'ArrowDown', 0.99],
    ['字高', 'vertical_scale', 'ArrowUp', 1.01],
    ['字高', 'vertical_scale', 'ArrowDown', 0.99],
    ['字距', 'tracking', 'ArrowUp', 10],
    ['字距', 'tracking', 'ArrowDown', -10],
  ])('steps mixed %s with %s %s from its normal value', async (label, field, key, expected) => {
    const format = vi.fn()
    const wrapper = mount(BubbleEditor, { props: { bubble: makeBubble(), bubbleIndex: 0 }, global: {
      stubs: { RichTextInput: { name: 'RichTextInput', template: '<div />', methods: { format } } },
    } })
    wrapper.findComponent({ name: 'RichTextInput' }).vm.$emit('selection', {
      active: true, count: 2, styles: [
        { size_pt: 24, horizontal_scale: 0.8, vertical_scale: 1.2, tracking: 0 },
        { size_pt: 30, horizontal_scale: 1.1, vertical_scale: 0.9, tracking: 50 },
      ],
    })
    await nextTick()
    const input = wrapper.get(`input[aria-label="${label}"]`)
    expect((input.element as HTMLInputElement).value).toBe('')
    expect(format).not.toHaveBeenCalled()
    await input.trigger('keydown', { key })
    expect(format).toHaveBeenLastCalledWith({ [field]: expected })
    expect(wrapper.text()).toContain('已选 2 个字')
    expect(wrapper.emitted('update')).toBeUndefined()
    wrapper.unmount()
  })

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

    const input = wrapper.get('[role="textbox"]')
    await input.trigger('compositionstart')
    ;(wrapper.getComponent(RichTextInput).vm as unknown as { insertText: (value: string) => void }).insertText('你好')
    expect(wrapper.emitted('update')).toBeUndefined()
    await input.trigger('compositionend')
    await nextTick()
    expect(wrapper.emitted('update')?.at(-1)).toEqual([expect.objectContaining({ translatedText: '你好译文', richText: expect.any(Object) })])
    wrapper.unmount()
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
      expect(wrapper.emitted('applyStyle')).toEqual([['page']])
      expect(copiedTextStyle.value?.box).toMatchObject({ textDirection: 'vertical', lineSpacing: 1.2 })
      expect(copiedTextStyle.value?.box).not.toHaveProperty('fillColor')
      expect(copiedTextStyle.value?.box).not.toHaveProperty('inpaintMethod')
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
    expect(editorSource).toContain("'applyStyle'")
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
      variant: 'secondary',
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
      '加粗',
      '斜体',
      '下划线',
      '清除局部样式',
      '适配字号',
      '竖向排版',
      '横向排版',
      '文字颜色',
      '文字描边',
      '描边颜色',
      '顶部对齐',
      '行内居中',
      '底部对齐',
      '文本块靠右',
      '文本块居中',
      '文本块靠左',
      '重置字距和字形',
      '逆时针旋转',
      '顺时针旋转',
      '重置旋转',
      '左移',
      '右移',
      '上移',
      '下移',
      '背景填充颜色',
      '删除样式收藏',
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

    expect(numberFields).toHaveLength(7)
    expect(numberFields.map(field => field.props('modelValue'))).toEqual(expect.arrayContaining([24, 3, 1.2, 0]))
    expect(numberFields.every(field => field.props('variant') === 'editor')).toBe(true)

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/BubbleEditor.vue'),
      'utf8',
    )

    expect(source).not.toMatch(/toolbar-fontsize-input|toolbar-mini-input|linespacing-input|toolbar-rotation-input/)
    expect(source).not.toMatch(/<UiInput\s+type="number"/)
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

  it('uses one editor-owned disclosure style for collapsible groups', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/BubbleEditor.vue'),
      'utf8',
    )

    expect(source).toContain('class="bubble-editor__disclosure"')
    expect(source).not.toContain('.bubble-editor__advanced summary')
  })

  it('maps editor owner colors through semantic tokens', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/BubbleEditor.vue'),
      'utf8',
    )

    expect(source).not.toMatch(/#[0-9A-Fa-f]{3,8}\b|rgba?\(/)
    expect(source).not.toMatch(/--bubble-editor-(text-column-divider|column-title-text|text-action-background)/)
    expect(source).not.toMatch(/<svg[\s>]/)
    expect(source).toContain('background: var(--color-surface-card)')
    expect(source).toContain('color: var(--color-text-strong)')
    expect(source).not.toContain('--bubble-editor-style-panel-')
  })

  it('uses typed model updates and an anchored color popover for text and color controls', async () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/BubbleEditor.vue'),
      'utf8',
    )

    expect(source).toContain("import EditColorPopover from './EditColorPopover.vue'")
    expect(source).toContain('aria-label="样式名称"')
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
    expect(wrapper.emitted('update')?.at(-1)).toEqual([expect.objectContaining({ richText: expect.objectContaining({
      runs: [expect.objectContaining({ len: 6, style: expect.objectContaining({ color: nativeColor('#123456') }) })],
    }) })])
    wrapper.unmount()
  })
})
