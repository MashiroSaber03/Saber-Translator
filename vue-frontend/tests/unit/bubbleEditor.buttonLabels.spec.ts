import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h } from 'vue'
import BubbleEditor from '@/components/edit/BubbleEditor.vue'
import type { BubbleState } from '@/types/bubble'

vi.mock('@/api/config', () => ({
  getFontListApi: vi.fn().mockResolvedValue({ fonts: [] }),
}))

const CustomSelectStub = defineComponent({
  name: 'CustomSelect',
  props: {
    modelValue: {
      type: String,
      default: '',
    },
  },
  emits: ['update:modelValue', 'change'],
  setup(props) {
    return () => h('div', { class: 'custom-select-stub' }, String(props.modelValue))
  },
})

const JapaneseKeyboardStub = defineComponent({
  name: 'JapaneseKeyboard',
  setup() {
    return () => h('div', { class: 'jp-keyboard-stub' })
  },
})

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
    textAlign: 'center',
    inpaintMethod: 'solid',
    textlines: [],
    ocrResult: null,
  }
}

describe('BubbleEditor button labels', () => {
  beforeEach(() => {
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
          CustomSelect: CustomSelectStub,
          JapaneseKeyboard: JapaneseKeyboardStub,
        },
      },
    })

    const buttonLabels = wrapper.findAll('button').map(button => button.text().trim())

    expect(buttonLabels).not.toContain('✓ 应用文本')
    expect(buttonLabels).not.toContain('应用')
    expect(buttonLabels).toContain('样式同步到本页全部气泡')
  })
})
