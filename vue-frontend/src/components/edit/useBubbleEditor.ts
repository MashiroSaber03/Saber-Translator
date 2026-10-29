import { ref, shallowRef, watch, computed, onMounted, onUnmounted, nextTick } from 'vue'
import {
  FONT_SIZE_PRESETS,
  FONT_SIZE_MIN,
  FONT_SIZE_STEP
} from '@/constants'
import type {
  BubbleState,
  BubbleColorField,
  ResolvedTextDirection,
  InpaintMethod,
  LogicalAlign,
} from '@/types/bubble'
import { apiClient } from '@/api/client'
import { defaultRun, plainRun, nativeColor, textColor, type EditorFont, type RunStyle } from './richTextDocument'
import { copiedTextStyle, copyTextStyle, textStyleTemplate, type TextStyleTemplate } from './textStyleClipboard'
import { useTextStyleFavorites } from './useTextStyleFavorites'
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
  selectedCount?: number
  selectedBubbles?: BubbleState[]
}

export type BubbleEditorEmit = {
  (e: 'update', updates: Partial<BubbleState>): void
  (e: 'formatText', style: RunStyle, reset?: boolean): void
  (e: 'fitText'): void
  (e: 'moveText', dx: number, dy: number): void
  (e: 'applyStyle', scope: 'selected' | 'page', template?: TextStyleTemplate): void
  (e: 'ocrRecognize', index: number): void
  (e: 'reTranslate', index: number): void
  (e: 'pickColor', field: BubbleColorField): void
}

type TextareaFieldRef = {
  focus: () => void
  selectionStart: number | null
  selectionEnd: number | null
  format?: (patch: RunStyle, replace?: boolean) => void
  insertText?: (text: string) => void
  deleteBackward?: () => void
  getSelectedText?: () => string
}

export function useBubbleEditor(props: BubbleEditorProps, emit: BubbleEditorEmit) {
  const publicAccess = usePublicUserAccess()
  const { favorites, name: favoriteName, selected: favoriteSelected, save: saveFavorite, remove: removeFavorite } = useTextStyleFavorites()
  const inpaintMethodOptions = computed(() => publicAccess.modelOptions(
    rawInpaintMethodOptions,
    {
      lama_mpe: 'lama_mpe',
      litelama: 'litelama',
      lama_manga: 'lama_manga',
    },
  ))
  const defaultBubble: BubbleState = createBubbleState({
    coords: [0, 0, 0, 0],
    polygon: [],
    fontFamily: '',
  })

  const localOriginalText = ref('')
  const localTranslatedText = ref('')
  const localFontSize = ref<number | null>(props.bubble?.fontSize ?? TEXT_STYLE_DEFAULTS.fontSize)
  const localFontFamily = ref(props.bubble?.fontFamily ?? '')
  const localTextDirection = ref<ResolvedTextDirection>('vertical')
  const localTextColor = ref(props.bubble?.textColor ?? TEXT_STYLE_DEFAULTS.textColor)
  const localFillColor = ref(TEXT_STYLE_DEFAULTS.fillColor)
  const localStrokeEnabled = ref(TEXT_STYLE_DEFAULTS.strokeEnabled)
  const localStrokeColor = ref(TEXT_STYLE_DEFAULTS.strokeColor)
  const localStrokeWidth = ref(TEXT_STYLE_DEFAULTS.strokeWidth)
  const localRotationAngle = ref(0)
  const localInpaintMethod = ref<InpaintMethod>(TEXT_STYLE_DEFAULTS.inpaintMethod)
  const localLineSpacing = ref(TEXT_STYLE_DEFAULTS.lineSpacing)
  const localInlineAlign = ref<LogicalAlign>(TEXT_STYLE_DEFAULTS.inlineAlign)
  const localBlockAlign = ref<LogicalAlign>(TEXT_STYLE_DEFAULTS.blockAlign)

  const originalTextInput = ref<TextareaFieldRef | null>(null)
  const translatedTextInput = ref<TextareaFieldRef | null>(null)
  const editorFonts = ref<EditorFont[]>([])
  const textSelection = ref<{ active: boolean; count: number; styles: RunStyle[] }>({ active: false, count: 0, styles: [] })
  const textScope = computed(() => textSelection.value.active
    ? textSelection.value.count ? `已选 ${textSelection.value.count} 个字` : '输入位置'
    : (props.selectedBubbles?.length ?? 0) > 1 ? `已选 ${props.selectedBubbles!.length} 个框` : '整框文字')
  const pasteLabel = computed(() => textSelection.value.active
    ? textSelection.value.count ? '粘贴到选字' : '粘贴到输入位置'
    : (props.selectedCount ?? 1) > 1 ? `粘贴到 ${props.selectedCount} 个框` : '粘贴样式')
  const copyLabel = computed(() => textSelection.value.active && textSelection.value.count ? '复制选中' : '复制整框')
  const selectedStyles = computed(() => !textSelection.value.active && (props.selectedBubbles?.length ?? 0) > 1
    ? props.selectedBubbles!.flatMap(bubble => bubble.richText?.runs.map(run => run.style)
      ?? [defaultRun(bubble, editorFonts.value.find(font => font.id === bubble.fontFamily)?.style)])
    : textSelection.value.styles)
  const bold = computed(() => selectedStyles.value.length > 0 && selectedStyles.value.every(style => style.faux_bold))
  const italic = computed(() => selectedStyles.value.length > 0 && selectedStyles.value.every(style => style.faux_italic))
  const underline = computed(() => selectedStyles.value.length > 0 && selectedStyles.value.every(style => style.underline))
  const characterMetrics = computed(() => [
    { field: 'tracking', label: '字距', initial: 0, factor: 10, min: -50, max: 200, title: '字间额外间距占字号的百分比，0% 为正常字距；横排和竖排均有效' },
    { field: 'horizontal_scale', label: '字宽', initial: 1, factor: 0.01, min: 10, max: 400, title: '文字宽度，100% 为正常字宽；调低可将文字压窄' },
    { field: 'vertical_scale', label: '字高', initial: 1, factor: 0.01, min: 10, max: 400, title: '文字高度，100% 为正常字高；调高可将文字拉长' },
  ].map(metric => {
    const values = selectedStyles.value.map(style => Number(style[metric.field] ?? metric.initial))
    return { ...metric, value: values.length && values.every(value => value === values[0])
      ? Number((values[0]! / metric.factor).toFixed(2)) : null }
  }))

  function setCharacterMetric(field: string, value: number | null): void {
    const metric = characterMetrics.value.find(item => item.field === field)
    if (metric && value !== null && Number.isFinite(value))
      formatText({ [field]: Number((Math.max(metric.min, Math.min(metric.max, value)) * metric.factor).toFixed(5)) })
  }

  const canCopyStyle = computed(() => {
    const styles = selectedStyles.value
    return styles.length > 0 && styles.every(style => JSON.stringify(style) === JSON.stringify(styles[0]))
  })

  function copyStyle(): void {
    if (props.bubble && canCopyStyle.value) copyTextStyle(props.bubble, selectedStyles.value[0]!)
  }

  function pasteStyle(): void {
    if (!copiedTextStyle.value) return
    if (textSelection.value.active) formatText(copiedTextStyle.value.character)
    else emit('applyStyle', 'selected')
  }

  function clearFormatting(): void {
    if (!props.bubble) return
    const base = plainRun(props.bubble, editorFonts.value.find(font => font.id === props.bubble!.fontFamily)?.style)
    if (!textSelection.value.active && (props.selectedBubbles?.length ?? 0) > 1) emit('formatText', {}, true)
    else translatedTextInput.value?.format?.(base, true)
  }

  function saveStyleFavorite(): void {
    if (props.bubble && canCopyStyle.value) saveFavorite(textStyleTemplate(props.bubble, selectedStyles.value[0]!))
  }
  function applyStyleFavorite(): void {
    const favorite = favorites.value.find(item => item.name === favoriteSelected.value)
    if (!favorite) return
    if (textSelection.value.active) formatText(favorite.style.character)
    else emit('applyStyle', 'selected', favorite.style)
  }

  function handleRichSelection(value: typeof textSelection.value): void {
    textSelection.value = value
  }

  watch(selectedStyles, styles => {
    if (!styles.length) return
    const size = styles[0]!.size_pt
    localFontSize.value = styles.every(style => style.size_pt === size) ? Number(Number(size).toFixed(2)) : null
    const first = styles[0]!
    const font = editorFonts.value.find(item => first.postscript_name
      ? item.style.postscript_name === first.postscript_name : item.style.font_family === first.font_family)
    localFontFamily.value = styles.every(style => style.font_family === first.font_family && style.postscript_name === first.postscript_name)
      ? font?.id ?? props.bubble?.fontFamily ?? '' : ''
    localTextColor.value = textColor(first) ?? props.bubble?.textColor ?? TEXT_STYLE_DEFAULTS.textColor
  }, { deep: true })

  function handleRichChange(value: Partial<BubbleState>): void { emit('update', value) }
  function formatText(style: RunStyle): void {
    if (!textSelection.value.active && (props.selectedBubbles?.length ?? 0) > 1) emit('formatText', style)
    else translatedTextInput.value?.format?.(style)
  }
  function toggleTextStyle(field: 'faux_bold' | 'faux_italic' | 'underline'): void {
    formatText({ [field]: !selectedStyles.value.every(style => style[field]) })
  }

  const colorPopover = shallowRef<{ field: BubbleColorField; anchor: HTMLElement } | null>(null)
  const colorValues = { textColor: localTextColor, fillColor: localFillColor, strokeColor: localStrokeColor }
  const activeColorValue = computed(() => colorValues[colorPopover.value?.field ?? 'textColor'].value)

  function closeColorPopover(): void {
    colorPopover.value = null
  }

  watch([
    () => props.bubble?.backendBubbleId ?? props.bubble?.clientMutationId ?? props.bubble,
    () => props.bubbleIndex,
    () => props.disabled,
  ], closeColorPopover, { flush: 'sync' })

  const showJpKeyboard = ref(false)
  const jpKeyboardTarget = ref<'original' | 'translated'>('original')
  let isOwnerMounted = true

  const positionX = computed(() => {
    if (!props.bubble) return 0
    return props.bubble.coords[0] + props.bubble.position.x
  })

  const positionY = computed(() => {
    if (!props.bubble) return 0
    return props.bubble.coords[1] + props.bubble.position.y
  })

  const fontSelectGroups = computed(() => {
    const sharedFonts = editorFonts.value.filter(font => font.scope === 'shared')
    const privateFonts = editorFonts.value.filter(font => font.scope === 'private')
    const knownFontIds = new Set(editorFonts.value.map(font => font.id))
    const currentFontId = localFontFamily.value
    const currentOptions = currentFontId && !knownFontIds.has(currentFontId)
      ? [{ label: String(textSelection.value.styles[0]?.font_family || '当前字体'), value: currentFontId }]
      : []
    const groups = []
    if (sharedFonts.length > 0 || currentOptions.length > 0) {
      groups.push({
        label: '共享字体',
        options: [
          ...currentOptions,
          ...sharedFonts.map(font => ({ label: font.displayName, value: font.id })),
        ],
      })
    }
    if (privateFonts.length > 0) {
      groups.push({
        label: '我的字体',
        options: privateFonts.map(font => ({ label: font.displayName, value: font.id })),
      })
    }
    return groups
  })

  function syncFromBubble(bubble: BubbleState | null): void {
    const b = bubble || defaultBubble
    localOriginalText.value = b.originalText
    localTranslatedText.value = b.translatedText
    localTextDirection.value = b.textDirection
    localFillColor.value = b.fillColor
    localStrokeEnabled.value = b.strokeEnabled
    localStrokeColor.value = b.strokeColor
    localStrokeWidth.value = b.strokeWidth
    localRotationAngle.value = b.rotationAngle
    localInpaintMethod.value = b.inpaintMethod
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

  function copyText(text: string): void {
    void copyTextToClipboard(text)
  }

  function copyOriginalText(): void {
    copyText(localOriginalText.value)
  }

  function copyTranslatedText(): void {
    copyText(translatedTextInput.value?.getSelectedText?.() || localTranslatedText.value)
  }

  function handleFontSizeChange(): void {
    if (localFontSize.value !== null) formatText({ size_pt: localFontSize.value })
  }

  function setFontSize(size: number): void {
    localFontSize.value = size
    formatText({ size_pt: size })
  }

  function handleFontFamilyChange(): void {
    const font = editorFonts.value.find(item => item.id === localFontFamily.value)
    if (font) formatText(font.style)
  }

  function setTextDirection(direction: ResolvedTextDirection): void {
    localTextDirection.value = direction
    emit('update', { textDirection: direction })
  }

  function openColorPopover(field: BubbleColorField, event: MouseEvent): void {
    if (props.disabled || !props.bubble || !(event.currentTarget instanceof HTMLElement)) return
    colorPopover.value = colorPopover.value?.field === field ? null : { field, anchor: event.currentTarget }
  }

  function applyColor(field: BubbleColorField, value: string): void {
    if (props.disabled || !props.bubble || colorPopover.value?.field !== field) return
    colorValues[field].value = value
    if (field === 'textColor') formatText({ color: nativeColor(value) })
    else emit('update', { [field]: value })
  }

  function requestImageColor(field: BubbleColorField): void {
    if (props.disabled || !props.bubble || colorPopover.value?.field !== field) return
    closeColorPopover()
    emit('pickColor', field)
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

  function moveBy(dx: number, dy: number): void {
    if (props.bubble) emit('moveText', dx, dy)
  }
  const moveLeft = () => moveBy(-MOVE_STEP, 0)
  const moveRight = () => moveBy(MOVE_STEP, 0)
  const moveUp = () => moveBy(0, -MOVE_STEP)
  const moveDown = () => moveBy(0, MOVE_STEP)

  function applyToAll(): void {
    copyStyle()
    if (canCopyStyle.value) emit('applyStyle', 'page')
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
    if (input.insertText) { input.insertText(char); input.focus(); return }
    const text = targetText(target)
    const start = input.selectionStart ?? text.length
    const end = input.selectionEnd ?? text.length
    updateTargetText(target, text.slice(0, start) + char + text.slice(end))
    restoreTargetSelection(input, start + char.length)
  }

  function handleKanaDelete(target: KeyboardTarget): void {
    const input = targetInput(target)
    if (input?.deleteBackward) { input.deleteBackward(); input.focus(); return }
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
      const { items: fonts } = await apiClient.get<{ items: EditorFont[] }>('/api/v2/editor/fonts')
      if (!isOwnerMounted) return
      editorFonts.value = fonts
    } catch {
      if (isOwnerMounted) editorFonts.value = []
    }
  }

  onMounted(() => {
    void loadFontList()
  })

  onUnmounted(() => {
    isOwnerMounted = false
  })

  return {
    favorites, favoriteName, favoriteSelected, removeFavorite, saveStyleFavorite, applyStyleFavorite, clearFormatting,
    copiedTextStyle,
    canCopyStyle,
    copyStyle,
    pasteStyle,
    editorFonts,
    textScope,
    pasteLabel,
    copyLabel,
    bold,
    italic,
    underline,
    characterMetrics,
    setCharacterMetric,
    handleRichSelection,
    handleRichChange,
    toggleTextStyle,
    formatText,
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
    localLineSpacing,
    localInlineAlign,
    localBlockAlign,
    originalTextInput,
    translatedTextInput,
    colorPopover,
    activeColorValue,
    closeColorPopover,
    applyColor,
    requestImageColor,
    showJpKeyboard,
    jpKeyboardTarget,
    positionX,
    positionY,
    fontSelectGroups,
    inpaintMethodOptions,
    handleOriginalTextChange,
    copyOriginalText,
    copyTranslatedText,
    handleFontSizeChange,
    setFontSize,
    handleFontFamilyChange,
    setTextDirection,
    openColorPopover,
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
    applyToAll,
    handleOcrRecognize,
    handleReTranslate,
    toggleJpKeyboard,
    handleKanaInsert,
    handleKanaDelete,
  }
}
