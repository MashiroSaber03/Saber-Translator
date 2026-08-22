import { ref, watch, computed, onMounted, onUnmounted, nextTick } from 'vue'
import {
  FONT_SIZE_PRESETS,
  FONT_SIZE_MIN,
  FONT_SIZE_STEP
} from '@/constants'
import type {
  BubbleState,
  ResolvedTextDirection,
  InpaintMethod,
  LogicalAlign,
} from '@/types/bubble'
import { listV2Fonts } from '@/api/v2/settings'
import { createBubbleState } from '@/utils/bubbleFactory'
import { copyTextToClipboard } from '@/utils/clipboard'
import { inpaintMethodOptions as rawInpaintMethodOptions } from '@/utils/textStyleForm'
import { TEXT_STYLE_DEFAULTS } from '@/defaults/textStyleDefaults'
import { usePublicUserAccess } from '@/composables/usePublicUserAccess'

export interface BubbleEditorProps {
  bubble: BubbleState | null
  bubbleIndex: number
  disabled?: boolean
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
  const publicAccess = usePublicUserAccess()
  const inpaintMethodOptions = computed(() => publicAccess.modelOptions(
    rawInpaintMethodOptions,
    {
      lama_mpe: 'lama_mpe',
      litelama: 'litelama',
    },
  ))
  const defaultBubble: BubbleState = createBubbleState({
    coords: [0, 0, 0, 0],
    polygon: [],
    fontFamily: '',
  })

  const localOriginalText = ref('')
  const localTranslatedText = ref('')
  const localFontSize = ref(TEXT_STYLE_DEFAULTS.fontSize)
  const localFontFamily = ref('')
  const localTextDirection = ref<ResolvedTextDirection>('vertical')
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
  const localInlineAlign = ref<LogicalAlign>(TEXT_STYLE_DEFAULTS.inlineAlign)
  const localBlockAlign = ref<LogicalAlign>(TEXT_STYLE_DEFAULTS.blockAlign)

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
    localPositionX.value = b.position.x
    localPositionY.value = b.position.y
    localLineSpacing.value = b.lineSpacing
    localInlineAlign.value = b.inlineAlign
    localBlockAlign.value = b.blockAlign
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

  function setTextDirection(direction: ResolvedTextDirection): void {
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
    localLineSpacing.value = v
    emit('update', { lineSpacing: v })
  }

  function setInlineAlign(align: LogicalAlign): void {
    localInlineAlign.value = align
    emit('update', { inlineAlign: align })
  }

  function setBlockAlign(align: LogicalAlign): void {
    localBlockAlign.value = align
    emit('update', { blockAlign: align })
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
    if (!props.bubble || props.bubbleIndex < 0) return
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
      inlineAlign: localInlineAlign.value,
      blockAlign: localBlockAlign.value,
    })
  }

  function resetBubbleEdit(): void {
    if (!props.bubble || props.bubbleIndex < 0) return
    emit('resetCurrent', props.bubbleIndex)
  }

  function handleOcrRecognize(): void {
    if (!props.bubble || props.bubbleIndex < 0) return
    emit('ocrRecognize', props.bubbleIndex)
  }

  function handleReTranslate(): void {
    if (!props.bubble || props.bubbleIndex < 0) return
    emit('reTranslate', props.bubbleIndex)
  }

  function toggleJpKeyboard(): void {
    showJpKeyboard.value = !showJpKeyboard.value
  }

  type KeyboardTarget = 'original' | 'translated'

  function targetInput(target: KeyboardTarget): TextareaFieldRef | null {
    return target === 'original' ? originalTextInput.value : translatedTextInput.value
  }

  function targetText(target: KeyboardTarget): string {
    return target === 'original' ? localOriginalText.value : localTranslatedText.value
  }

  function updateTargetText(target: KeyboardTarget, value: string): void {
    if (target === 'original') {
      localOriginalText.value = value
      emit('update', { originalText: value })
    } else {
      localTranslatedText.value = value
      emit('update', { translatedText: value })
    }
  }

  function restoreTargetSelection(input: TextareaFieldRef, position: number): void {
    nextTick(() => {
      input.selectionStart = position
      input.selectionEnd = position
      input.focus()
    })
  }

  function handleKanaInsert(char: string, target: KeyboardTarget): void {
    const input = targetInput(target)
    if (!input) return
    const text = targetText(target)
    const start = input.selectionStart ?? text.length
    const end = input.selectionEnd ?? text.length
    updateTargetText(target, text.slice(0, start) + char + text.slice(end))
    restoreTargetSelection(input, start + char.length)
  }

  function handleKanaDelete(target: KeyboardTarget): void {
    const input = targetInput(target)
    const text = targetText(target)
    if (!input || text.length === 0) return
    const start = input.selectionStart ?? text.length
    const end = input.selectionEnd ?? text.length
    if (start !== end) {
      updateTargetText(target, text.slice(0, start) + text.slice(end))
      restoreTargetSelection(input, start)
      return
    }
    if (start <= 0) return
    const previousCodePoint = Array.from(text.slice(0, start)).at(-1)
    if (!previousCodePoint) return
    const deleteStart = start - previousCodePoint.length
    updateTargetText(target, text.slice(0, deleteStart) + text.slice(end))
    restoreTargetSelection(input, deleteStart)
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
    void loadFontList()
  })

  onUnmounted(() => {
    isOwnerMounted = false
  })

  return {
    FONT_SIZE_PRESETS,
    FONT_SIZE_MIN,
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
    localInlineAlign,
    localBlockAlign,
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
    setInlineAlign,
    setBlockAlign,
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
