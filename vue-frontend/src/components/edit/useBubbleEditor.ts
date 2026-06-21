import { ref, watch, computed, onMounted, nextTick } from 'vue'
import { useBubbleStore } from '@/stores/bubbleStore'
import {
  FONT_SIZE_PRESETS,
  FONT_SIZE_MIN,
  FONT_SIZE_MAX,
  FONT_SIZE_STEP
} from '@/constants'
import type { BubbleState, TextDirection, InpaintMethod, TextAlign } from '@/types/bubble'
import { getFontListApi } from '@/api/config'
import { createBubbleState } from '@/utils/bubbleFactory'
import { TEXT_STYLE_DEFAULTS } from '@/defaults/textStyleDefaults'

export interface BubbleEditorProps {
  /** 当前选中的气泡；无选中时为 null */
  bubble: BubbleState | null
  /** 当前气泡索引 */
  bubbleIndex: number
  /** OCR 识别中 */
  isOcrLoading?: boolean
  /** 翻译中 */
  isTranslateLoading?: boolean
}

export type BubbleEditorEmit = {
  /** 更新当前气泡 */
  (e: 'update', updates: Partial<BubbleState>): void
  /** 重新渲染 */
  (e: 'reRender'): void
  /** 重新 OCR 识别 */
  (e: 'ocrRecognize', index: number): void
  /** 重新翻译当前气泡 */
  (e: 'reTranslate', index: number): void
  /** 重置当前气泡到进入编辑模式时的状态 */
  (e: 'resetCurrent', index: number): void
}

export function useBubbleEditor(props: BubbleEditorProps, emit: BubbleEditorEmit) {
  // ============================================================
  // Store
  // ============================================================

  const bubbleStore = useBubbleStore()

  // ============================================================
  // 默认值
  // ============================================================

  const defaultBubble: BubbleState = createBubbleState({
    coords: [0, 0, 0, 0],
    polygon: [],
  })

  // ============================================================
  // 本地状态（用于双向绑定）
  // ============================================================

  const localOriginalText = ref('')
  const localTranslatedText = ref('')
  const localFontSize = ref(TEXT_STYLE_DEFAULTS.fontSize)
  const localFontFamily = ref(TEXT_STYLE_DEFAULTS.fontFamily)
  const localTextDirection = ref<TextDirection>('vertical')  // 简化设计：不再使用 'auto'
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

  // 文本输入框引用
  const originalTextInput = ref<HTMLTextAreaElement | null>(null)
  const translatedTextInput = ref<HTMLTextAreaElement | null>(null)

  // 颜色选择器引用
  const textColorInput = ref<HTMLInputElement | null>(null)
  const fillColorInput = ref<HTMLInputElement | null>(null)
  const strokeColorInput = ref<HTMLInputElement | null>(null)

  // 日语软键盘状态
  const showJpKeyboard = ref(false)
  const jpKeyboardTarget = ref<'original' | 'translated'>('original')

  // 字体相关
  const systemFonts = ref<{ name: string; path: string }[]>([
    { name: '思源黑体', path: TEXT_STYLE_DEFAULTS.fontFamily },
    { name: '华文楷体', path: 'fonts/STKAITI.TTF' },
    { name: '华文细黑', path: 'fonts/STXIHEI.TTF' },
    { name: '黑体', path: 'fonts/SIMHEI.TTF' },
    { name: '宋体', path: 'fonts/SIMSUN.TTC' },
  ])
  const customFonts = ref<{ name: string; path: string }[]>([])

  // ============================================================
  // 计算属性
  // ============================================================

  /** 位置X */
  const positionX = computed(() => {
    if (!props.bubble) return 0
    return props.bubble.coords[0] + localPositionX.value
  })

  /** 位置Y */
  const positionY = computed(() => {
    if (!props.bubble) return 0
    return props.bubble.coords[1] + localPositionY.value
  })

  /** 字体选择器分组选项（用于CustomSelect） */
  const fontSelectGroups = computed(() => {
    const groups = [
      {
        label: '系统字体',
        options: systemFonts.value.map(f => ({ label: f.name, value: f.path })),
      },
    ]
    if (customFonts.value.length > 0) {
      groups.push({
        label: '自定义字体',
        options: customFonts.value.map(f => ({ label: f.name, value: f.path })),
      })
    }
    return groups
  })

  /** 背景修复方式选项（用于CustomSelect） */
  const inpaintMethodOptions = [
    { label: '纯色填充', value: 'solid' },
    { label: 'LAMA修复(漫画)', value: 'lama_mpe' },
    { label: 'LAMA修复(通用)', value: 'litelama' },
  ]

  // ============================================================
  // 同步本地状态
  // ============================================================

  /** 从气泡数据同步到本地状态 */
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

  // 监听 props 变化，同步本地状态
  watch(
    () => props.bubble,
    newBubble => {
      syncFromBubble(newBubble)
    },
    { deep: true, immediate: true }
  )

  // ============================================================
  // 事件处理 - 文本
  // ============================================================

  /** 处理原文变化 */
  function handleOriginalTextChange(): void {
    emit('update', { originalText: localOriginalText.value })
  }

  /** 处理译文变化 */
  function handleTextChange(): void {
    emit('update', { translatedText: localTranslatedText.value })
  }

  /** 复制原文 */
  function copyOriginalText(): void {
    navigator.clipboard.writeText(localOriginalText.value)
  }

  /** 复制译文 */
  function copyTranslatedText(): void {
    navigator.clipboard.writeText(localTranslatedText.value)
  }

  // ============================================================
  // 事件处理 - 字体和字号
  // ============================================================

  /** 处理字号变化 */
  function handleFontSizeChange(): void {
    emit('update', { fontSize: localFontSize.value })
  }

  /** 设置字号 */
  function setFontSize(size: number): void {
    localFontSize.value = size
    emit('update', { fontSize: size })
  }

  /** 增大字号 */
  function increaseFontSize(): void {
    localFontSize.value = Math.min(FONT_SIZE_MAX, localFontSize.value + FONT_SIZE_STEP)
    emit('update', { fontSize: localFontSize.value })
  }

  /** 减小字号 */
  function decreaseFontSize(): void {
    localFontSize.value = Math.max(FONT_SIZE_MIN, localFontSize.value - FONT_SIZE_STEP)
    emit('update', { fontSize: localFontSize.value })
  }

  /** 处理字体变化 */
  function handleFontFamilyChange(): void {
    emit('update', { fontFamily: localFontFamily.value })
  }

  // ============================================================
  // 事件处理 - 排版方向
  // ============================================================

  /** 设置排版方向 */
  function setTextDirection(direction: TextDirection): void {
    localTextDirection.value = direction
    emit('update', { textDirection: direction })
  }

  // ============================================================
  // 事件处理 - 颜色
  // ============================================================

  /** 触发文字颜色选择器 */
  function triggerTextColorPicker(): void {
    textColorInput.value?.click()
  }

  /** 处理文字颜色变化 */
  function handleTextColorChange(): void {
    emit('update', { textColor: localTextColor.value })
  }

  /** 触发填充颜色选择器 */
  function triggerFillColorPicker(): void {
    fillColorInput.value?.click()
  }

  /** 处理填充颜色变化 */
  function handleFillColorChange(): void {
    emit('update', { fillColor: localFillColor.value })
  }

  /** 触发描边颜色选择器 */
  function triggerStrokeColorPicker(): void {
    strokeColorInput.value?.click()
  }

  /** 处理描边颜色变化 */
  function handleStrokeColorChange(): void {
    emit('update', { strokeColor: localStrokeColor.value })
  }

  // ============================================================
  // 事件处理 - 描边
  // ============================================================

  /** 切换描边 */
  function toggleStroke(): void {
    localStrokeEnabled.value = !localStrokeEnabled.value
    emit('update', { strokeEnabled: localStrokeEnabled.value })
  }

  /** 处理描边宽度变化 */
  function handleStrokeWidthChange(): void {
    emit('update', { strokeWidth: localStrokeWidth.value })
  }

  // ============================================================
  // 事件处理 - 修复方式
  // ============================================================

  /** 处理修复方式变化 */
  function handleInpaintMethodChange(): void {
    emit('update', { inpaintMethod: localInpaintMethod.value })
  }

  // ============================================================
  // 事件处理 - 行间距与对齐
  // ============================================================

  /** 处理行间距变化（限制在 0.5-3.0） */
  function handleLineSpacingChange(): void {
    let v = Number(localLineSpacing.value)
    if (!Number.isFinite(v) || v <= 0) v = TEXT_STYLE_DEFAULTS.lineSpacing
    v = Math.max(0.5, Math.min(3.0, v))
    localLineSpacing.value = v
    emit('update', { lineSpacing: v })
  }

  /** 设置对齐方式 */
  function setTextAlign(align: TextAlign): void {
    localTextAlign.value = align
    emit('update', { textAlign: align })
  }

  // ============================================================
  // 事件处理 - 旋转
  // ============================================================

  /** 处理旋转角度变化 */
  function handleRotationChange(): void {
    emit('update', { rotationAngle: localRotationAngle.value })
  }

  /** 逆时针旋转 */
  function rotateLeft(): void {
    localRotationAngle.value = Math.max(-180, localRotationAngle.value - 5)
    emit('update', { rotationAngle: localRotationAngle.value })
  }

  /** 顺时针旋转 */
  function rotateRight(): void {
    localRotationAngle.value = Math.min(180, localRotationAngle.value + 5)
    emit('update', { rotationAngle: localRotationAngle.value })
  }

  /** 重置旋转 */
  function resetRotation(): void {
    localRotationAngle.value = 0
    emit('update', { rotationAngle: 0 })
  }

  // ============================================================
  // 事件处理 - 位置
  // ============================================================

  const MOVE_STEP = 2

  /** 左移 */
  function moveLeft(): void {
    localPositionX.value -= MOVE_STEP
    emit('update', { position: { x: localPositionX.value, y: localPositionY.value } })
  }

  /** 右移 */
  function moveRight(): void {
    localPositionX.value += MOVE_STEP
    emit('update', { position: { x: localPositionX.value, y: localPositionY.value } })
  }

  /** 上移 */
  function moveUp(): void {
    localPositionY.value -= MOVE_STEP
    emit('update', { position: { x: localPositionX.value, y: localPositionY.value } })
  }

  /** 下移 */
  function moveDown(): void {
    localPositionY.value += MOVE_STEP
    emit('update', { position: { x: localPositionX.value, y: localPositionY.value } })
  }

  /** 重置位置 */
  function resetPosition(): void {
    localPositionX.value = 0
    localPositionY.value = 0
    emit('update', { position: { x: 0, y: 0 } })
  }

  // ============================================================
  // 事件处理 - 操作按钮
  // ============================================================

  /** 应用到全部气泡 */
  function applyToAll(): void {
    bubbleStore.updateAllBubbles({
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
    console.log('样式已应用到所有气泡')
    // 触发重新渲染
    emit('reRender')
  }

  /** 重置气泡编辑 */
  function resetBubbleEdit(): void {
    // 【当前行为 4.3】通知父组件重置当前气泡到初始状态
    // 父级编辑工作区持有进入编辑模式时的气泡快照。
    emit('resetCurrent', props.bubbleIndex)
  }

  /** 重新OCR识别 */
  function handleOcrRecognize(): void {
    emit('ocrRecognize', props.bubbleIndex)
  }

  /** 重新翻译单个气泡 */
  function handleReTranslate(): void {
    emit('reTranslate', props.bubbleIndex)
  }

  // ============================================================
  // 日语软键盘相关
  // ============================================================

  /** 切换日语软键盘显示 */
  function toggleJpKeyboard(): void {
    showJpKeyboard.value = !showJpKeyboard.value
  }

  /** 处理假名插入 */
  function handleKanaInsert(char: string, target: 'original' | 'translated'): void {
    if (target === 'original') {
      const input = originalTextInput.value
      if (input) {
        const start = input.selectionStart || localOriginalText.value.length
        const end = input.selectionEnd || localOriginalText.value.length
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
        const start = input.selectionStart || localTranslatedText.value.length
        const end = input.selectionEnd || localTranslatedText.value.length
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

  /** 处理假名删除 */
  function handleKanaDelete(target: 'original' | 'translated'): void {
    if (target === 'original') {
      const input = originalTextInput.value
      if (input && localOriginalText.value.length > 0) {
        const start = input.selectionStart || localOriginalText.value.length
        const end = input.selectionEnd || localOriginalText.value.length
        const text = localOriginalText.value
        if (start === end && start > 0) {
          localOriginalText.value = text.slice(0, start - 1) + text.slice(end)
          nextTick(() => {
            input.selectionStart = input.selectionEnd = start - 1
            input.focus()
          })
        } else if (start !== end) {
          localOriginalText.value = text.slice(0, start) + text.slice(end)
          nextTick(() => {
            input.selectionStart = input.selectionEnd = start
            input.focus()
          })
        }
        emit('update', { originalText: localOriginalText.value })
      }
    } else {
      const input = translatedTextInput.value
      if (input && localTranslatedText.value.length > 0) {
        const start = input.selectionStart || localTranslatedText.value.length
        const end = input.selectionEnd || localTranslatedText.value.length
        const text = localTranslatedText.value
        if (start === end && start > 0) {
          localTranslatedText.value = text.slice(0, start - 1) + text.slice(end)
          nextTick(() => {
            input.selectionStart = input.selectionEnd = start - 1
            input.focus()
          })
        } else if (start !== end) {
          localTranslatedText.value = text.slice(0, start) + text.slice(end)
          nextTick(() => {
            input.selectionStart = input.selectionEnd = start
            input.focus()
          })
        }
        emit('update', { translatedText: localTranslatedText.value })
      }
    }
  }

  // ============================================================
  // 字体管理
  // ============================================================

  /** 加载字体列表 */
  async function loadFontList(): Promise<void> {
    try {
      const response = await getFontListApi()
      if (response.fonts) {
        const system: { name: string; path: string }[] = []
        const custom: { name: string; path: string }[] = []

        for (const font of response.fonts) {
          // API返回的字段是display_name，需要转换为name
          const fontItem = {
            name: typeof font === 'string' ? font : font.display_name || font.file_name || '',
            path: typeof font === 'string' ? font : font.path,
          }
          if (fontItem.path.startsWith('fonts/')) {
            system.push(fontItem)
          } else {
            custom.push(fontItem)
          }
        }

        if (system.length > 0) {
          systemFonts.value = system
        }
        customFonts.value = custom
      }
    } catch (error) {
      console.error('加载字体列表失败:', error)
    }
  }

  // ============================================================
  // 生命周期
  // ============================================================

  onMounted(() => {
    loadFontList()
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
    increaseFontSize,
    decreaseFontSize,
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
