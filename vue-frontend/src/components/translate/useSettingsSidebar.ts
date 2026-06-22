import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useBookTranslationConstraintsStore } from '@/stores/bookTranslationConstraintsStore'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import {
  getFontList,
  type TranslateWorkflowPreferences,
  getTranslateWorkflowPreferences,
  saveTranslateWorkflowPreferences,
  uploadFont,
} from '@/api/config'
import { showToast } from '@/utils/toast'
import { TEXT_STYLE_DEFAULTS } from '@/defaults/textStyleDefaults'
import type { TextDirection, InpaintMethod, TextAlign } from '@/types/bubble'
import {
  BUILTIN_FONTS,
  clampLineSpacing,
  getFontDisplayName,
  inpaintMethodOptions,
  layoutDirectionOptions,
  textAlignOptions,
} from '@/utils/textStyleForm'
import {
  DEFAULT_WORKFLOW_MODE,
  WORKFLOW_MODE_CONFIGS,
  isWorkflowMode,
  type WorkflowMode,
  type WorkflowModeConfig,
  type WorkflowRunRequest,
} from '@/types/workflow'
import { clampPageSelection, createPageSelectionSummary } from '@/utils/pageSelection'

export interface ApplySettingsOptions {
  fontSize: boolean
  fontFamily: boolean
  layoutDirection: boolean
  textColor: boolean
  fillColor: boolean
  strokeEnabled: boolean
  strokeColor: boolean
  strokeWidth: boolean
  lineSpacing: boolean
  textAlign: boolean
}

export type SettingsSidebarEmit = {
  /** 启动翻译工作流 */
  (e: 'runWorkflow', payload: WorkflowRunRequest): void
  /** 切换到上一张图片 */
  (e: 'previous'): void
  /** 切换到下一张图片 */
  (e: 'next'): void
  /** 将当前设置应用到全部气泡 */
  (e: 'applyToAll', options: ApplySettingsOptions): void
  /** 单项文字样式配置发生变化 */
  (e: 'textStyleChanged', settingKey: string, newValue: unknown): void
  /** 自动字号开关发生变化 */
  (e: 'autoFontSizeChanged', isAutoFontSize: boolean): void
  /** 自动文字颜色开关发生变化 */
  (e: 'autoTextColorChanged', isAutoTextColor: boolean): void
  /** 打开本书术语表 */
  (e: 'openGlossary'): void
  /** 打开本书不翻译列表 */
  (e: 'openNonTranslate'): void
}

export function useSettingsSidebar(emit: SettingsSidebarEmit) {
  const imageStore = useImageStore()
  const settingsStore = useSettingsStore()
  const bookTranslationConstraintsStore = useBookTranslationConstraintsStore()

  // ============================================================
  // 状态定义
  // ============================================================

  /** 应用设置下拉菜单是否显示 */
  const showApplyOptions = ref(false)

  /** 应用设置选项 */
  const applyOptions = ref<ApplySettingsOptions>({
    fontSize: true,
    fontFamily: true,
    layoutDirection: true,
    textColor: true,
    fillColor: true,
    strokeEnabled: true,
    strokeColor: true,
    strokeWidth: true,
    lineSpacing: true,
    textAlign: true,
  })

  /** 是否启用指定翻译页码 */
  const isPageSelectionEnabled = ref(false)

  /** 全局共享的已选页码（1-based） */
  const selectedPages = ref<number[]>([])

  /** 页码选择弹窗显示状态 */
  const showPageSelectionModal = ref(false)

  /** 当前工作流模式 */
  const selectedWorkflowMode = ref<WorkflowMode>(DEFAULT_WORKFLOW_MODE)

  /** 是否记住翻译页操作模式 */
  const rememberWorkflowModeEnabled = ref(false)

  /** 用户是否已经在本次挂载后手动切换过操作模式 */
  const hasUserChangedWorkflowMode = ref(false)

  /** 用户是否已经在本次挂载后手动切换过记忆开关 */
  const hasUserChangedRememberWorkflowMode = ref(false)

  /** 等待保存到后端的最新偏好快照 */
  let pendingWorkflowPreferences: TranslateWorkflowPreferences | null = null

  /** 是否正在写入翻译页操作模式偏好 */
  let isPersistingWorkflowPreferences = false

  // ============================================================
  // 计算属性
  // ============================================================

  /** 当前图片 */
  const currentImage = computed(() => imageStore.currentImage)

  /** 是否有图片 */
  const hasImages = computed(() => imageStore.hasImages)

  /** 总图片数量 */
  const totalImages = computed(() => imageStore.images.length)

  const normalizedSelectedPages = computed(() => clampPageSelection(selectedPages.value, totalImages.value))
  const hasValidPageSelection = computed(() => normalizedSelectedPages.value.length > 0)

  /** 是否可以翻译 */
  const canTranslate = computed(() => hasImages.value && !imageStore.isBatchTranslationInProgress)
  const canUseBookConstraints = computed(() => bookTranslationConstraintsStore.isAvailable)

  /** 是否可以切换上一张 */
  const canGoPrevious = computed(() => imageStore.canGoPrevious)

  /** 是否可以切换下一张 */
  const canGoNext = computed(() => imageStore.canGoNext)

  /** 当前工作流是否可执行 */
  const canRunWorkflow = computed(() => {
    const mode = selectedWorkflowMode.value
    const selectionInvalid = isPageSelectionActiveForCurrentMode.value && !hasValidPageSelection.value

    switch (mode) {
      case 'translate-current':
        return !!currentImage.value && canTranslate.value
      case 'translate-batch':
      case 'hq-batch':
      case 'proofread-batch':
        return canTranslate.value && !selectionInvalid
      case 'remove-current':
      case 'delete-current':
        return !!currentImage.value
      case 'remove-batch':
        return hasImages.value && !selectionInvalid
      case 'clear-all':
        return hasImages.value
      case 'retry-failed':
        return hasFailedImages.value && !imageStore.isBatchTranslationInProgress
      default:
        return false
    }
  })

  /** 文字样式设置 */
  const textStyle = computed(() => settingsStore.textStyle)

  /** 失败图片数量 */
  const failedImageCount = computed(() => imageStore.failedImageCount)

  /** 是否有失败图片 */
  const hasFailedImages = computed(() => failedImageCount.value > 0)

  /** 当前工作流配置 */
  const selectedWorkflowConfig = computed<WorkflowModeConfig>(() => {
    return (
      WORKFLOW_MODE_CONFIGS.find(cfg => cfg.mode === selectedWorkflowMode.value) ??
      WORKFLOW_MODE_CONFIGS[0]!
    )
  })

  /** 当前模式是否支持指定页码 */
  const supportsPageSelectionForCurrentMode = computed(() => selectedWorkflowConfig.value.supportsPageSelection)

  /** 指定页码是否被激活且可用于当前模式 */
  const isPageSelectionActiveForCurrentMode = computed(() => {
    return supportsPageSelectionForCurrentMode.value && isPageSelectionEnabled.value
  })

  /** 工作流选项（用于 CustomSelect） */
  const workflowModeOptions = computed(() => {
    return WORKFLOW_MODE_CONFIGS.map(cfg => ({
      label: cfg.label,
      value: cfg.mode,
    }))
  })

  /** 启动按钮文案 */
  const workflowStartLabel = computed(() => selectedWorkflowConfig.value.startLabel)

  /** 当前模式的范围/对象标签 */
  const workflowContextTag = computed(() => {
    if (isPageSelectionActiveForCurrentMode.value && hasValidPageSelection.value) {
      return `已选 ${normalizedSelectedPages.value.length} 页`
    }

    switch (selectedWorkflowMode.value) {
      case 'translate-current':
      case 'remove-current':
      case 'delete-current':
        return '当前页'
      case 'translate-batch':
      case 'hq-batch':
      case 'proofread-batch':
      case 'remove-batch':
      case 'clear-all':
        return '全量'
      case 'retry-failed':
        return hasFailedImages.value ? `失败 ${failedImageCount.value} 张` : '失败重试'
      default:
        return '流程'
    }
  })

  /** 当前模式类型标签 */
  const workflowModeTag = computed(() => {
    if (isDangerousWorkflow.value) {
      return '高风险'
    }
    return supportsPageSelectionForCurrentMode.value ? '批量流程' : '单页流程'
  })

  /** 当前模式说明文案 */
  const workflowDescription = computed(() => {
    switch (selectedWorkflowMode.value) {
      case 'delete-current':
        return '删除前会弹出确认，建议先检查当前页是否已保存。'
      case 'clear-all':
        return '清除前会弹出确认，此操作会移除所有已加载图片。'
      case 'retry-failed':
        return hasFailedImages.value
          ? `将重试 ${failedImageCount.value} 张失败图片。`
          : '当前没有失败图片可重试。'
      default:
        if (isPageSelectionActiveForCurrentMode.value && hasValidPageSelection.value) {
          return `当前页码：${createPageSelectionSummary(normalizedSelectedPages.value)}。`
        }
        if (isPageSelectionActiveForCurrentMode.value && !hasValidPageSelection.value) {
          return '请至少选择一页。'
        }
        if (supportsPageSelectionForCurrentMode.value) {
          return '当前作用于全部图片（可启用指定翻译页码）。'
        }
        return '当前只作用于当前图片。'
    }
  })

  /** 当前工作流是否危险操作 */
  const isDangerousWorkflow = computed(() => selectedWorkflowConfig.value.isDangerous)

  /** 字体列表（包含内置字体） */
  const fontList = ref<string[]>([])

  /** 字体上传输入框引用 */
  const fontUploadInput = ref<HTMLInputElement | null>(null)

  /** 字体选择选项（用于CustomSelect） */
  const fontSelectOptions = computed(() => {
    const options = fontList.value.map(font => ({
      label: getFontDisplayName(font),
      value: font,
    }))
    options.push({ label: '自定义字体...', value: 'custom-font' })
    return options
  })

  // ============================================================
  // 生命周期
  // ============================================================

  onMounted(async () => {
    void loadWorkflowPreferences()

    // 加载字体列表
    await loadFontList()

    // 确保当前选中的字体在列表中
    const currentFont = textStyle.value.fontFamily
    if (currentFont && !fontList.value.includes(currentFont)) {
      // 如果当前字体不在列表中，添加到列表
      fontList.value = [currentFont, ...fontList.value]
    }

    // 监听点击外部关闭应用选项菜单
    window.addEventListener('click', handleClickOutside)
  })

  onUnmounted(() => {
    window.removeEventListener('click', handleClickOutside)
  })

  watch(supportsPageSelectionForCurrentMode, supports => {
    if (!supports) {
      isPageSelectionEnabled.value = false
    }
  })

  watch(totalImages, (count) => {
    selectedPages.value = clampPageSelection(selectedPages.value, count)
  })

  // ============================================================
  // 方法
  // ============================================================

  /**
   * 加载字体列表
   */
  async function loadFontList() {
    try {
      const response = await getFontList()
      if (response.fonts && Array.isArray(response.fonts) && response.fonts.length > 0) {
        fontList.value = response.fonts.map(font => font.path)
      } else {
        // 如果API失败，至少显示内置字体
        fontList.value = [...BUILTIN_FONTS]
      }
    } catch (error) {
      console.error('加载字体列表失败:', error)
      // 出错时也显示内置字体
      fontList.value = [...BUILTIN_FONTS]
    }
  }

  async function loadWorkflowPreferences() {
    try {
      const response = await getTranslateWorkflowPreferences()
      const preferences = response.preferences
      if (!response.success || !preferences) return

      if (!hasUserChangedRememberWorkflowMode.value) {
        rememberWorkflowModeEnabled.value = preferences.rememberWorkflowModeEnabled
      }

      if (
        preferences.rememberWorkflowModeEnabled &&
        isWorkflowMode(preferences.lastWorkflowMode) &&
        !hasUserChangedWorkflowMode.value &&
        !hasUserChangedRememberWorkflowMode.value
      ) {
        selectedWorkflowMode.value = preferences.lastWorkflowMode
      }
    } catch (error) {
      console.warn('加载翻译页操作模式偏好失败:', error)
    }
  }

  async function persistWorkflowPreferences(
    rememberEnabled: boolean,
    workflowMode: WorkflowMode
  ): Promise<void> {
    pendingWorkflowPreferences = {
      rememberWorkflowModeEnabled: rememberEnabled,
      lastWorkflowMode: workflowMode,
    }

    if (isPersistingWorkflowPreferences) return

    isPersistingWorkflowPreferences = true
    while (pendingWorkflowPreferences) {
      const nextPreferences = pendingWorkflowPreferences
      pendingWorkflowPreferences = null

      try {
        await saveTranslateWorkflowPreferences(nextPreferences)
      } catch (error) {
        console.warn('保存翻译页操作模式偏好失败:', error)
      }
    }
    isPersistingWorkflowPreferences = false
  }

  /**
   * 更新字号
   */
  function updateFontSize(event: Event) {
    const value = parseInt((event.target as HTMLInputElement).value)
    if (!isNaN(value)) {
      settingsStore.updateTextStyle({ fontSize: value })
      emit('textStyleChanged', 'fontSize', value)
    }
  }

  /**
   * 更新自动字号
   * 切换后触发 autoFontSizeChanged 事件，由父组件决定是否重绘。
   */
  function updateAutoFontSize(event: Event) {
    const checked = (event.target as HTMLInputElement).checked
    settingsStore.updateTextStyle({ autoFontSize: checked })
    console.log(`自动字号设置变更: ${checked}`)
    emit('autoFontSizeChanged', checked)
  }

  /**
   * 处理字体文件上传
   */
  async function handleFontUpload(event: Event) {
    const input = event.target as HTMLInputElement
    const file = input.files?.[0]
    if (!file) return

    // 验证文件类型
    const validExtensions = ['.ttf', '.ttc', '.otf']
    const fileName = file.name.toLowerCase()
    const isValidType = validExtensions.some(ext => fileName.endsWith(ext))

    if (!isValidType) {
      showToast('请选择 .ttf、.ttc 或 .otf 格式的字体文件', 'error')
      input.value = ''
      return
    }

    try {
      const response = await uploadFont(file)
      if (response.success && response.fontPath) {
        // 更新字体列表
        await loadFontList()
        // 设置新上传的字体为当前字体
        settingsStore.updateTextStyle({ fontFamily: response.fontPath })
        showToast('字体上传成功', 'success')
      } else {
        showToast(response.error || '字体上传失败', 'error')
      }
    } catch (error) {
      console.error('字体上传失败:', error)
      showToast('字体上传失败', 'error')
    } finally {
      // 清空文件输入
      input.value = ''
    }
  }

  /**
   * 处理字体选择变化（CustomSelect）
   */
  function handleFontSelectChange(value: string | number) {
    const strValue = String(value)
    if (strValue === 'custom-font') {
      fontUploadInput.value?.click()
      return
    }
    settingsStore.updateTextStyle({ fontFamily: strValue })
    emit('textStyleChanged', 'fontFamily', strValue)
  }

  /**
   * 处理排版方向变化（CustomSelect）
   */
  function handleLayoutDirectionChange(value: string | number) {
    const strValue = String(value)
    settingsStore.updateTextStyle({ layoutDirection: strValue as TextDirection })
    emit('textStyleChanged', 'layoutDirection', strValue)
  }

  /**
   * 处理填充方式变化（CustomSelect）
   */
  function handleInpaintMethodChange(value: string | number) {
    const strValue = String(value)
    settingsStore.updateTextStyle({ inpaintMethod: strValue as InpaintMethod })
  }

  /**
   * 更新文字颜色
   */
  function updateTextColor(event: Event) {
    const value = (event.target as HTMLInputElement).value
    settingsStore.updateTextStyle({ textColor: value })
    emit('textStyleChanged', 'textColor', value)
  }

  /**
   * 更新行间距倍数（0.5 - 3.0）
   */
  function updateLineSpacing(event: Event) {
    const value = clampLineSpacing(Number((event.target as HTMLInputElement).value), TEXT_STYLE_DEFAULTS.lineSpacing)
    settingsStore.updateTextStyle({ lineSpacing: value })
    emit('textStyleChanged', 'lineSpacing', value)
  }

  /**
   * 更新对齐方式
   */
  function updateTextAlign(value: string | number) {
    const strValue = String(value) as TextAlign
    settingsStore.updateTextStyle({ textAlign: strValue })
    emit('textStyleChanged', 'textAlign', strValue)
  }

  /**
   * 更新是否使用自动文字颜色
   */
  function updateUseAutoTextColor(event: Event) {
    const checked = (event.target as HTMLInputElement).checked
    settingsStore.updateTextStyle({ useAutoTextColor: checked })
    emit('autoTextColorChanged', checked)
  }

  /**
   * 更新描边启用状态
   */
  function updateStrokeEnabled(event: Event) {
    const checked = (event.target as HTMLInputElement).checked
    settingsStore.updateTextStyle({ strokeEnabled: checked })
    emit('textStyleChanged', 'strokeEnabled', checked)
  }

  /**
   * 更新描边颜色
   */
  function updateStrokeColor(event: Event) {
    const value = (event.target as HTMLInputElement).value
    settingsStore.updateTextStyle({ strokeColor: value })
    emit('textStyleChanged', 'strokeColor', value)
  }

  /**
   * 更新描边宽度
   */
  function updateStrokeWidth(event: Event) {
    const value = parseInt((event.target as HTMLInputElement).value)
    if (!isNaN(value)) {
      settingsStore.updateTextStyle({ strokeWidth: value })
      emit('textStyleChanged', 'strokeWidth', value)
    }
  }

  /**
   * 更新填充颜色
   */
  function updateFillColor(event: Event) {
    const value = (event.target as HTMLInputElement).value
    settingsStore.updateTextStyle({ fillColor: value })
    emit('textStyleChanged', 'fillColor', value)
  }

  /**
   * 切换应用设置下拉菜单
   */
  function toggleApplyOptions() {
    showApplyOptions.value = !showApplyOptions.value
  }

  /**
   * 全选/取消全选应用选项
   */
  function toggleSelectAll() {
    const allSelected = Object.values(applyOptions.value).every(v => v)
    const newValue = !allSelected
    applyOptions.value = {
      fontSize: newValue,
      fontFamily: newValue,
      layoutDirection: newValue,
      textColor: newValue,
      fillColor: newValue,
      strokeEnabled: newValue,
      strokeColor: newValue,
      strokeWidth: newValue,
      lineSpacing: newValue,
      textAlign: newValue,
    }
  }

  /**
   * 应用设置到全部
   */
  function handleApplyToAll() {
    emit('applyToAll', { ...applyOptions.value })
    showApplyOptions.value = false
  }

  /**
   * 点击外部关闭下拉菜单
   */
  function handleClickOutside(event: MouseEvent) {
    const target = event.target as HTMLElement
    if (!target.closest('.settings-sidebar__apply-group')) {
      showApplyOptions.value = false
    }
  }

  function openPageSelectionModal(): void {
    if (totalImages.value === 0 || !supportsPageSelectionForCurrentMode.value) return
    showPageSelectionModal.value = true
  }

  function handlePageSelectionConfirm(pages: number[]): void {
    selectedPages.value = clampPageSelection(pages, totalImages.value)
  }

  /**
   * 处理工作流模式切换
   */
  function handleWorkflowModeChange(value: string | number) {
    const workflowMode = String(value)
    if (!isWorkflowMode(workflowMode)) return

    hasUserChangedWorkflowMode.value = true
    selectedWorkflowMode.value = workflowMode
    void persistWorkflowPreferences(rememberWorkflowModeEnabled.value, workflowMode)
  }

  function handleRememberWorkflowModeChange(event: Event) {
    const checked = (event.target as HTMLInputElement).checked
    hasUserChangedRememberWorkflowMode.value = true
    rememberWorkflowModeEnabled.value = checked
    void persistWorkflowPreferences(checked, selectedWorkflowMode.value)
  }

  /**
   * 启动当前工作流
   */
  function handleRunWorkflow() {
    if (!canRunWorkflow.value) return

    const payload: WorkflowRunRequest = {
      mode: selectedWorkflowMode.value,
    }

    if (isPageSelectionActiveForCurrentMode.value && hasValidPageSelection.value) {
      payload.pageSelection = {
        pages: normalizedSelectedPages.value,
      }
    }

    emit('runWorkflow', payload)
  }

  function handleOpenGlossary(): void {
    emit('openGlossary')
  }

  function handleOpenNonTranslate(): void {
    emit('openNonTranslate')
  }

  return {
    showApplyOptions,
    applyOptions,
    isPageSelectionEnabled,
    selectedPages,
    showPageSelectionModal,
    selectedWorkflowMode,
    rememberWorkflowModeEnabled,
    currentImage,
    hasImages,
    totalImages,
    normalizedSelectedPages,
    hasValidPageSelection,
    canTranslate,
    canUseBookConstraints,
    canGoPrevious,
    canGoNext,
    canRunWorkflow,
    textStyle,
    failedImageCount,
    hasFailedImages,
    selectedWorkflowConfig,
    supportsPageSelectionForCurrentMode,
    isPageSelectionActiveForCurrentMode,
    workflowModeOptions,
    workflowStartLabel,
    workflowContextTag,
    workflowModeTag,
    workflowDescription,
    isDangerousWorkflow,
    fontList,
    fontUploadInput,
    fontSelectOptions,
    layoutDirectionOptions,
    inpaintMethodOptions,
    textAlignOptions,
    createPageSelectionSummary,
    updateFontSize,
    updateAutoFontSize,
    handleFontUpload,
    handleFontSelectChange,
    handleLayoutDirectionChange,
    handleInpaintMethodChange,
    updateTextColor,
    updateLineSpacing,
    updateTextAlign,
    updateUseAutoTextColor,
    updateStrokeEnabled,
    updateStrokeColor,
    updateStrokeWidth,
    updateFillColor,
    toggleApplyOptions,
    toggleSelectAll,
    handleApplyToAll,
    openPageSelectionModal,
    handlePageSelectionConfirm,
    handleWorkflowModeChange,
    handleRememberWorkflowModeChange,
    handleRunWorkflow,
    handleOpenGlossary,
    handleOpenNonTranslate,
  }
}
