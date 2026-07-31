import { ref, watch, computed, onMounted, onUnmounted, nextTick } from 'vue'
import {
  FONT_SIZE_PRESETS,
  FONT_SIZE_MIN,
  FONT_SIZE_MAX,
  FONT_SIZE_STEP
} from '@/constants'
import type { BubbleState, TextDirection, InpaintMethod, TextAlign } from '@/types/bubble'
import { listV2Fonts } from '@/api/v2/settings'
import { createBubbleState } from '@/utils/bubbleFactory'
import { copyTextToClipboard } from '@/utils/clipboard'
import { TEXT_STYLE_DEFAULTS } from '@/defaults/textStyleDefaults'

export interface BubbleEditorProps {
  bubble: BubbleState | null
  bubbleIndex: number
  isOcrLoading?: boolean
  isTranslateLoading?: boolean
}

export type BubbleEditorEmit = {
  (e: 'update', updates: Partial<BubbleState>): void
  (e: 'applyToAllStyle', updates: Partial<BubbleState>): void
  (e: 'ocrRecognize', index: number): void
  (e: 'reTranslate', index: number): void
  (e: 'resetCurrent', index: number): void
}

type TextareaFieldRef = {
  focus: () => void
  selectionStart: number | null
  selectionEnd: number | null
}

type ColorInputRef = {
  click: () => void
}

export function useBubbleEditor(props: BubbleEditorProps, emit: BubbleEditorEmit) {
  const defaultBubble: BubbleState = createBubbleState({
    coords: [0, 0, 0, 0],
    polygon: [],
    fontFamily: '',
  })

  const localOriginalText = ref('')
  const localTranslatedText = ref('')
  const localFontSize = ref(TEXT_STYLE_DEFAULTS.fontSize)
  const localFontFamily = ref('')
  const localTextDirection = ref<TextDirection>('vertical')
  const localTextColor = ref(TEXT_STYLE_DEFAULTS.textColor)
  const localFillColor = ref(TEXT_STYLE_DEFAULTS.fillColor)
  const localStrokeEnabled = ref(TEXT_STYLE_DEFAULTS.strokeEnabled)
  const localStrokeColor = ref(TEXT_STYLE_DEFAULTS.strokeColor)
  const localStrokeWidth = ref(TEXT_STYLE_DEFAULTS.strokeWidth)
  const localRotationAngle = ref(0)
  const localInpaintMethod = ref<InpaintMethod>(TEXT_STYLE_DEFAULTS.inpaintMethod)
  const localPositionX = ref(0)
  const localPositionY = ref(0)
  const localLineSpacing = ref(TEXT_STYLE_DEFAULTS.lineSpacing)
  const localTextAlign = ref<TextAlign>(TEXT_STYLE_DEFAULTS.textAlign)

  const originalTextInput = ref<TextareaFieldRef | null>(null)
  const translatedTextInput = ref<TextareaFieldRef | null>(null)

  const textColorInput = ref<ColorInputRef | null>(null)
  const fillColorInput = ref<ColorInputRef | null>(null)
  const strokeColorInput = ref<ColorInputRef | null>(null)

  const showJpKeyboard = ref(false)
  const jpKeyboardTarget = ref<'original' | 'translated'>('original')
  let isOwnerMounted = true

  const systemFonts = ref<{ name: string; id: string }[]>([])
  const customFonts = ref<{ name: string; id: string }[]>([])

  const positionX = computed(() => {
    if (!props.bubble) return 0
    return props.bubble.coords[0] + localPositionX.value
  })

  const positionY = computed(() => {
    if (!props.bubble) return 0
    return props.bubble.coords[1] + localPositionY.value
  })

  const fontSelectGroups = computed(() => {
    const knownFontIds = new Set([
      ...systemFonts.value.map(font => font.id),
      ...customFonts.value.map(font => font.id),
    ])
    const currentFontId = localFontFamily.value
    const currentOptions = currentFontId && !knownFontIds.has(currentFontId)
      ? [{ label: `当前字体 (${currentFontId})`, value: currentFontId }]
      : []
    const groups = []
    if (systemFonts.value.length > 0 || currentOptions.length > 0) {
      groups.push({
        label: '系统字体',
        options: [
          ...currentOptions,
          ...systemFonts.value.map(font => ({ label: font.name, value: font.id })),
        ],
      })
    }
    if (customFonts.value.length > 0) {
      groups.push({
        label: '自定义字体',
        options: customFonts.value.map(font => ({ label: font.name, value: font.id })),
      })
    }
    return groups
  })

  const inpaintMethodOptions = [
    { label: '纯色填充', value: 'solid' },
    { label: 'LAMA修复(漫画)', value: 'lama_mpe' },
    { label: 'LAMA修复(通用)', value: 'litelama' },
  ]

  function syncFromBubble(bubble: BubbleState | null): void {
    const b = bubble || defaultBubble
    localOriginalText.value = b.originalText
    localTranslatedText.value = b.translatedText
    localFontSize.value = b.fontSize
    localFontFamily.value = b.fontFamily
    localTextDirection.value = b.textDirection
    localTextColor.value = b.textColor
    localFillColor.value = b.fillColor
    localStrokeEnabled.value = b.strokeEnabled
    localStrokeColor.value = b.strokeColor
    localStrokeWidth.value = b.strokeWidth
    localRotationAngle.value = b.rotationAngle
    localInpaintMethod.value = b.inpaintMethod
    localPositionX.value = b.position?.x || 0
    localPositionY.value = b.position?.y || 0
    localLineSpacing.value = b.lineSpacing ?? TEXT_STYLE_DEFAULTS.lineSpacing
    localTextAlign.value = b.textAlign ?? TEXT_STYLE_DEFAULTS.textAlign
  }

  watch(
    () => props.bubble,
    newBubble => {
      syncFromBubble(newBubble)
    },
    { deep: true, immediate: true }
  )

  function handleOriginalTextChange(value: string): void {
    localOriginalText.value = value
    emit('update', { originalText: localOriginalText.value })
  }

  function handleTextChange(value: string): void {
    localTranslatedText.value = value
    emit('update', { translatedText: localTranslatedText.value })
  }

  function copyText(text: string): void {
    void copyTextToClipboard(text)
  }

  function copyOriginalText(): void {
    copyText(localOriginalText.value)
  }

  function copyTranslatedText(): void {
    copyText(localTranslatedText.value)
  }

  function handleFontSizeChange(): void {
    emit('update', { fontSize: localFontSize.value })
  }

  function setFontSize(size: number): void {
    localFontSize.value = size
    emit('update', { fontSize: size })
  }

  function handleFontFamilyChange(): void {
    emit('update', { fontFamily: localFontFamily.value })
  }

  function setTextDirection(direction: TextDirection): void {
    localTextDirection.value = direction
    emit('update', { textDirection: direction })
  }

  function triggerTextColorPicker(): void {
    textColorInput.value?.click()
  }

  function handleTextColorChange(value: string): void {
    localTextColor.value = value
    emit('update', { textColor: localTextColor.value })
  }

  function triggerFillColorPicker(): void {
    fillColorInput.value?.click()
  }

  function handleFillColorChange(value: string): void {
    localFillColor.value = value
    emit('update', { fillColor: localFillColor.value })
  }

  function triggerStrokeColorPicker(): void {
    strokeColorInput.value?.click()
  }

  function handleStrokeColorChange(value: string): void {
    localStrokeColor.value = value
    emit('update', { strokeColor: localStrokeColor.value })
  }

  function toggleStroke(): void {
    localStrokeEnabled.value = !localStrokeEnabled.value
    emit('update', { strokeEnabled: localStrokeEnabled.value })
  }

  function handleStrokeWidthChange(): void {
    emit('update', { strokeWidth: localStrokeWidth.value })
  }

  function handleInpaintMethodChange(): void {
    emit('update', { inpaintMethod: localInpaintMethod.value })
  }

  function handleLineSpacingChange(): void {
    let v = Number(localLineSpacing.value)
    if (!Number.isFinite(v) || v <= 0) v = TEXT_STYLE_DEFAULTS.lineSpacing
    v = Math.max(0.5, Math.min(3.0, v))
    localLineSpacing.value = v
    emit('update', { lineSpacing: v })
  }

  function setTextAlign(align: TextAlign): void {
    localTextAlign.value = align
    emit('update', { textAlign: align })
  }

  function handleRotationChange(): void {
    emit('update', { rotationAngle: localRotationAngle.value })
  }

  function rotateLeft(): void {
    localRotationAngle.value = Math.max(-180, localRotationAngle.value - 5)
    emit('update', { rotationAngle: localRotationAngle.value })
  }

  function rotateRight(): void {
    localRotationAngle.value = Math.min(180, localRotationAngle.value + 5)
    emit('update', { rotationAngle: localRotationAngle.value })
  }

  function resetRotation(): void {
    localRotationAngle.value = 0
    emit('update', { rotationAngle: 0 })
  }

  const MOVE_STEP = 2

  function moveLeft(): void {
    localPositionX.value -= MOVE_STEP
    emit('update', { position: { x: localPositionX.value, y: localPositionY.value } })
  }

  function moveRight(): void {
    localPositionX.value += MOVE_STEP
    emit('update', { position: { x: localPositionX.value, y: localPositionY.value } })
  }

  function moveUp(): void {
    localPositionY.value -= MOVE_STEP
    emit('update', { position: { x: localPositionX.value, y: localPositionY.value } })
  }

  function moveDown(): void {
    localPositionY.value += MOVE_STEP
    emit('update', { position: { x: localPositionX.value, y: localPositionY.value } })
  }

  function resetPosition(): void {
    localPositionX.value = 0
    localPositionY.value = 0
    emit('update', { position: { x: 0, y: 0 } })
  }

  function applyToAll(): void {
    emit('applyToAllStyle', {
      fontSize: localFontSize.value,
      fontFamily: localFontFamily.value,
      textDirection: localTextDirection.value,
      textColor: localTextColor.value,
      fillColor: localFillColor.value,
      strokeEnabled: localStrokeEnabled.value,
      strokeColor: localStrokeColor.value,
      strokeWidth: localStrokeWidth.value,
      inpaintMethod: localInpaintMethod.value,
      lineSpacing: localLineSpacing.value,
      textAlign: localTextAlign.value,
    })
  }

  function resetBubbleEdit(): void {
    // 父级编辑工作区持有进入编辑模式时的气泡快照。
    emit('resetCurrent', props.bubbleIndex)
  }

  function handleOcrRecognize(): void {
    emit('ocrRecognize', props.bubbleIndex)
  }

  function handleReTranslate(): void {
    emit('reTranslate', props.bubbleIndex)
  }

  function toggleJpKeyboard(): void {
    showJpKeyboard.value = !showJpKeyboard.value
  }

  function handleKanaInsert(char: string, target: 'original' | 'translated'): void {
    if (target === 'original') {
      const input = originalTextInput.value
      if (input) {
        const start = input.selectionStart ?? localOriginalText.value.length
        const end = input.selectionEnd ?? localOriginalText.value.length
        const text = localOriginalText.value
        localOriginalText.value = text.slice(0, start) + char + text.slice(end)
        nextTick(() => {
          input.selectionStart = input.selectionEnd = start + char.length
          input.focus()
        })
        emit('update', { originalText: localOriginalText.value })
      }
    } else {
      const input = translatedTextInput.value
      if (input) {
        const start = input.selectionStart ?? localTranslatedText.value.length
        const end = input.selectionEnd ?? localTranslatedText.value.length
        const text = localTranslatedText.value
        localTranslatedText.value = text.slice(0, start) + char + text.slice(end)
        nextTick(() => {
          input.selectionStart = input.selectionEnd = start + char.length
          input.focus()
        })
        emit('update', { translatedText: localTranslatedText.value })
      }
    }
  }

  function handleKanaDelete(target: 'original' | 'translated'): void {
    if (target === 'original') {
      const input = originalTextInput.value
      if (input && localOriginalText.value.length > 0) {
        const start = input.selectionStart ?? localOriginalText.value.length
        const end = input.selectionEnd ?? localOriginalText.value.length
        const text = localOriginalText.value
        let didChange = false
        if (start === end && start > 0) {
          localOriginalText.value = text.slice(0, start - 1) + text.slice(end)
          didChange = true
          nextTick(() => {
            input.selectionStart = input.selectionEnd = start - 1
            input.focus()
          })
        } else if (start !== end) {
          localOriginalText.value = text.slice(0, start) + text.slice(end)
          didChange = true
          nextTick(() => {
            input.selectionStart = input.selectionEnd = start
            input.focus()
          })
        }
        if (didChange) {
          emit('update', { originalText: localOriginalText.value })
        }
      }
    } else {
      const input = translatedTextInput.value
      if (input && localTranslatedText.value.length > 0) {
        const start = input.selectionStart ?? localTranslatedText.value.length
        const end = input.selectionEnd ?? localTranslatedText.value.length
        const text = localTranslatedText.value
        let didChange = false
        if (start === end && start > 0) {
          localTranslatedText.value = text.slice(0, start - 1) + text.slice(end)
          didChange = true
          nextTick(() => {
            input.selectionStart = input.selectionEnd = start - 1
            input.focus()
          })
        } else if (start !== end) {
          localTranslatedText.value = text.slice(0, start) + text.slice(end)
          didChange = true
          nextTick(() => {
            input.selectionStart = input.selectionEnd = start
            input.focus()
          })
        }
        if (didChange) {
          emit('update', { translatedText: localTranslatedText.value })
        }
      }
    }
  }

  async function loadFontList(): Promise<void> {
    try {
      const fonts = await listV2Fonts()
      if (!isOwnerMounted) return
      const system: { name: string; id: string }[] = []
      const custom: { name: string; id: string }[] = []

      for (const font of fonts) {
        const fontItem = {
          name: font.displayName,
          id: font.id,
        }
        if (font.kind === 'builtin') {
          system.push(fontItem)
        } else {
          custom.push(fontItem)
        }
      }

      systemFonts.value = system
      customFonts.value = custom
    } catch {
      if (isOwnerMounted) {
        systemFonts.value = []
        customFonts.value = []
      }
    }
  }

  onMounted(() => {
    loadFontList()
  })

  onUnmounted(() => {
    isOwnerMounted = false
  })

  return {
    FONT_SIZE_PRESETS,
    FONT_SIZE_MIN,
    FONT_SIZE_MAX,
    FONT_SIZE_STEP,
    localOriginalText,
    localTranslatedText,
    localFontSize,
    localFontFamily,
    localTextDirection,
    localTextColor,
    localFillColor,
    localStrokeEnabled,
    localStrokeColor,
    localStrokeWidth,
    localRotationAngle,
    localInpaintMethod,
    localPositionX,
    localPositionY,
    localLineSpacing,
    localTextAlign,
    originalTextInput,
    translatedTextInput,
    textColorInput,
    fillColorInput,
    strokeColorInput,
    showJpKeyboard,
    jpKeyboardTarget,
    positionX,
    positionY,
    fontSelectGroups,
    inpaintMethodOptions,
    handleOriginalTextChange,
    handleTextChange,
    copyOriginalText,
    copyTranslatedText,
    handleFontSizeChange,
    setFontSize,
    handleFontFamilyChange,
    setTextDirection,
    triggerTextColorPicker,
    handleTextColorChange,
    triggerFillColorPicker,
    handleFillColorChange,
    triggerStrokeColorPicker,
    handleStrokeColorChange,
    toggleStroke,
    handleStrokeWidthChange,
    handleInpaintMethodChange,
    handleLineSpacingChange,
    setTextAlign,
    handleRotationChange,
    rotateLeft,
    rotateRight,
    resetRotation,
    moveLeft,
    moveRight,
    moveUp,
    moveDown,
    resetPosition,
    applyToAll,
    resetBubbleEdit,
    handleOcrRecognize,
    handleReTranslate,
    toggleJpKeyboard,
    handleKanaInsert,
    handleKanaDelete,
  }
}
