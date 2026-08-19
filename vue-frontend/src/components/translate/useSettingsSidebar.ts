import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import type UiFileInput from '@/components/ui/UiFileInput.vue'
import { useBookTranslationConstraintsStore } from '@/stores/bookTranslationConstraintsStore'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import {
  uploadV2Font,
  type V2WorkflowPreferences,
} from '@/api/v2/settings'
import { showToast } from '@/utils/toast'
import {
  blockAlignOptions,
  inlineAlignOptions,
  inpaintMethodOptions,
  layoutDirectionOptions,
} from '@/utils/textStyleForm'
import {
  DEFAULT_WORKFLOW_MODE,
  WORKFLOW_MODE_CONFIGS,
  isWorkflowMode,
  type WorkflowModeConfig,
} from './workflowModeConfig'
import {
  type WorkflowMode,
  type WorkflowRunRequest,
} from '@/types/workflow'
import { clampPageSelection, createPageSelectionSummary } from '@/utils/pageSelection'
import {
  FONT_FILE_FORMATS_LABEL,
  isSupportedFontFileName,
} from '@/utils/fontFiles'
import type { TextStyleMutationArgs } from '@/types/settings'

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
  inlineAlign: boolean
  blockAlign: boolean
}

export type SettingsSidebarEmit = {
  (e: 'runWorkflow', payload: WorkflowRunRequest): void
  (e: 'previous'): void
  (e: 'next'): void
  (e: 'applyToAll', options: ApplySettingsOptions): void
  (e: 'textStyleChanged', ...args: TextStyleMutationArgs): void
  (e: 'autoFontSizeChanged', isAutoFontSize: boolean): void
  (e: 'autoTextColorChanged', isAutoTextColor: boolean): void
  (e: 'openGlossary'): void
  (e: 'openNonTranslate'): void
}

export function useSettingsSidebar(emit: SettingsSidebarEmit) {
  const imageStore = useImageStore()
  const settingsStore = useSettingsStore()
  const bookTranslationConstraintsStore = useBookTranslationConstraintsStore()

  const showApplyOptions = ref(false)

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
    inlineAlign: true,
    blockAlign: true,
  })

  const isPageSelectionEnabled = ref(false)

  const selectedPages = ref<number[]>([])

  const showPageSelectionModal = ref(false)

  const selectedWorkflowMode = ref<WorkflowMode>(DEFAULT_WORKFLOW_MODE)

  const rememberWorkflowModeEnabled = ref(false)

  const hasUserChangedWorkflowMode = ref(false)

  const hasUserChangedRememberWorkflowMode = ref(false)

  let pendingWorkflowPreferences: V2WorkflowPreferences | null = null

  let isPersistingWorkflowPreferences = false

  const currentImage = computed(() => imageStore.currentImage)

  const isCurrentPageReady = computed(() => (
    currentImage.value !== null
    && currentImage.value.bubbleStates !== null
  ))

  const hasImages = computed(() => imageStore.hasImages)

  const totalImages = computed(() => imageStore.images.length)

  const normalizedSelectedPages = computed(() =>
    clampPageSelection(selectedPages.value, totalImages.value)
  )
  const hasValidPageSelection = computed(() => normalizedSelectedPages.value.length > 0)

  const canTranslate = computed(() => hasImages.value && !imageStore.isTranslationInProgress)
  const canUseBookConstraints = computed(() => bookTranslationConstraintsStore.isAvailable)

  const canGoPrevious = computed(() => imageStore.canGoPrevious)

  const canGoNext = computed(() => imageStore.canGoNext)

  const canRunWorkflow = computed(() => {
    const mode = selectedWorkflowMode.value
    const isIdle = !imageStore.isTranslationInProgress
    const selectionInvalid =
      isPageSelectionActiveForCurrentMode.value && !hasValidPageSelection.value

    switch (mode) {
      case 'translate-current':
        return !!currentImage.value && isIdle
      case 'translate-batch':
      case 'hq-batch':
      case 'proofread-batch':
        return canTranslate.value && !selectionInvalid
      case 'remove-current':
      case 'delete-current':
        return !!currentImage.value && isIdle
      case 'remove-batch':
        return hasImages.value && isIdle && !selectionInvalid
      case 'clear-all':
        return hasImages.value && isIdle
      case 'retry-failed':
        return hasFailedImages.value && isIdle
      default:
        return false
    }
  })

  const textStyle = computed(() => settingsStore.textStyle)

  const failedImageCount = computed(() => imageStore.failedImageCount)

  const hasFailedImages = computed(() => failedImageCount.value > 0)

  const selectedWorkflowConfig = computed<WorkflowModeConfig>(() => {
    return (
      WORKFLOW_MODE_CONFIGS.find(cfg => cfg.mode === selectedWorkflowMode.value) ??
      WORKFLOW_MODE_CONFIGS[0]!
    )
  })

  const supportsPageSelectionForCurrentMode = computed(
    () => selectedWorkflowConfig.value.supportsPageSelection
  )

  const isPageSelectionActiveForCurrentMode = computed(() => {
    return supportsPageSelectionForCurrentMode.value && isPageSelectionEnabled.value
  })

  const workflowModeOptions = computed(() => {
    return WORKFLOW_MODE_CONFIGS.map(cfg => ({
      label: cfg.label,
      value: cfg.mode,
    }))
  })

  const workflowStartLabel = computed(() => selectedWorkflowConfig.value.startLabel)

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

  const workflowModeTag = computed(() => {
    if (isDangerousWorkflow.value) {
      return '高风险'
    }
    return supportsPageSelectionForCurrentMode.value ? '批量流程' : '单页流程'
  })

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

  const isDangerousWorkflow = computed(() => selectedWorkflowConfig.value.isDangerous)

  const fontUploadInput = ref<InstanceType<typeof UiFileInput> | null>(null)
  const isUploadingFont = ref(false)

  const fontSelectOptions = computed(() => {
    const options = settingsStore.fontCatalog.map(font => ({
      label: font.displayName,
      value: font.id,
    }))
    options.push({ label: '自定义字体...', value: 'custom-font' })
    return options
  })

  onMounted(() => {
    window.addEventListener('click', handleClickOutside)

    applyWorkflowPreferences(settingsStore.workflowPreferences)
  })

  onUnmounted(() => {
    window.removeEventListener('click', handleClickOutside)
  })

  watch(supportsPageSelectionForCurrentMode, supports => {
    if (!supports) {
      isPageSelectionEnabled.value = false
    }
  })

  watch(totalImages, count => {
    selectedPages.value = clampPageSelection(selectedPages.value, count)
  })

  watch(
    () => settingsStore.workflowPreferences,
    applyWorkflowPreferences,
    { deep: true },
  )

  function applyWorkflowPreferences(preferences: V2WorkflowPreferences): void {
    if (!hasUserChangedRememberWorkflowMode.value) {
      rememberWorkflowModeEnabled.value = preferences.rememberWorkflowModeEnabled
    }

    if (
      preferences.rememberWorkflowModeEnabled
      && isWorkflowMode(preferences.lastWorkflowMode)
      && !hasUserChangedWorkflowMode.value
      && !hasUserChangedRememberWorkflowMode.value
    ) {
      selectedWorkflowMode.value = preferences.lastWorkflowMode
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
        await settingsStore.saveWorkflowPreferences(nextPreferences)
      } catch {
        // Preference persistence is best-effort and must not interrupt translation.
      }
    }
    isPersistingWorkflowPreferences = false
  }

  function updateFontSize(value: number) {
    if (Number.isInteger(value) && value >= 1) {
      settingsStore.updateTextStyle({ fontSize: value })
      emit('textStyleChanged', 'fontSize', value)
    }
  }

  function updateAutoFontSize(checked: boolean) {
    settingsStore.updateTextStyle({ autoFontSize: checked })
    emit('autoFontSizeChanged', checked)
  }

  async function handleFontUpload(files: File[]) {
    const file = files[0]
    if (!file) return
    if (isUploadingFont.value) {
      fontUploadInput.value?.clear()
      return
    }

    if (!isSupportedFontFileName(file.name)) {
      showToast(`请选择 ${FONT_FILE_FORMATS_LABEL} 格式的字体文件`, 'error')
      fontUploadInput.value?.clear()
      return
    }

    isUploadingFont.value = true
    try {
      const uploadedFont = await uploadV2Font(file)
      settingsStore.upsertFont(uploadedFont)
      settingsStore.updateTextStyle({ fontFamily: uploadedFont.id })
      emit('textStyleChanged', 'fontFamily', uploadedFont.id)
      showToast('字体上传成功', 'success')
    } catch {
      showToast('字体上传失败', 'error')
    } finally {
      isUploadingFont.value = false
      fontUploadInput.value?.clear()
    }
  }

  function handleFontSelectChange(value: string | number) {
    if (typeof value !== 'string') return
    if (value === 'custom-font') {
      fontUploadInput.value?.click()
      return
    }
    if (!value) return
    settingsStore.updateTextStyle({ fontFamily: value })
    emit('textStyleChanged', 'fontFamily', value)
  }

  function handleLayoutDirectionChange(value: string | number) {
    if (value !== 'auto' && value !== 'vertical' && value !== 'horizontal') return
    settingsStore.updateTextStyle({ layoutDirection: value })
    emit('textStyleChanged', 'layoutDirection', value)
  }

  function handleInpaintMethodChange(value: string | number) {
    if (value !== 'solid' && value !== 'lama_mpe' && value !== 'litelama') return
    settingsStore.updateTextStyle({ inpaintMethod: value })
    emit('textStyleChanged', 'inpaintMethod', value)
  }

  function updateTextColor(value: string) {
    settingsStore.updateTextStyle({ textColor: value })
    emit('textStyleChanged', 'textColor', value)
  }

  function updateLineSpacing(nextValue: number) {
    if (!Number.isFinite(nextValue) || nextValue <= 0) return
    settingsStore.updateTextStyle({ lineSpacing: nextValue })
    emit('textStyleChanged', 'lineSpacing', nextValue)
  }

  function updateInlineAlign(value: string | number) {
    if (value !== 'start' && value !== 'center' && value !== 'end') return
    settingsStore.updateTextStyle({ inlineAlign: value })
    emit('textStyleChanged', 'inlineAlign', value)
  }

  function updateBlockAlign(value: string | number) {
    if (value !== 'start' && value !== 'center' && value !== 'end') return
    settingsStore.updateTextStyle({ blockAlign: value })
    emit('textStyleChanged', 'blockAlign', value)
  }

  function updateUseAutoTextColor(checked: boolean) {
    settingsStore.updateTextStyle({ useAutoTextColor: checked })
    emit('autoTextColorChanged', checked)
  }

  function updateStrokeEnabled(checked: boolean) {
    settingsStore.updateTextStyle({ strokeEnabled: checked })
    emit('textStyleChanged', 'strokeEnabled', checked)
  }

  function updateStrokeColor(value: string) {
    settingsStore.updateTextStyle({ strokeColor: value })
    emit('textStyleChanged', 'strokeColor', value)
  }

  function updateStrokeWidth(value: number) {
    if (Number.isInteger(value) && value >= 0) {
      settingsStore.updateTextStyle({ strokeWidth: value })
      emit('textStyleChanged', 'strokeWidth', value)
    }
  }

  function updateFillColor(value: string) {
    settingsStore.updateTextStyle({ fillColor: value })
    emit('textStyleChanged', 'fillColor', value)
  }

  function toggleApplyOptions() {
    showApplyOptions.value = !showApplyOptions.value
  }

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
      inlineAlign: newValue,
      blockAlign: newValue,
    }
  }

  function handleApplyToAll() {
    emit('applyToAll', { ...applyOptions.value })
    showApplyOptions.value = false
  }

  function handleClickOutside(event: MouseEvent) {
    const target = event.target
    if (!(target instanceof Element) || !target.closest('.apply-options-section')) {
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

  function handleWorkflowModeChange(value: string | number) {
    if (!isWorkflowMode(value)) return

    hasUserChangedWorkflowMode.value = true
    selectedWorkflowMode.value = value
    void persistWorkflowPreferences(rememberWorkflowModeEnabled.value, value)
  }

  function handleRememberWorkflowModeChange(checked: boolean) {
    hasUserChangedRememberWorkflowMode.value = true
    rememberWorkflowModeEnabled.value = checked
    void persistWorkflowPreferences(checked, selectedWorkflowMode.value)
  }

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
    isCurrentPageReady,
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
    fontUploadInput,
    fontSelectOptions,
    layoutDirectionOptions,
    inpaintMethodOptions,
    inlineAlignOptions,
    blockAlignOptions,
    createPageSelectionSummary,
    updateFontSize,
    updateAutoFontSize,
    handleFontUpload,
    handleFontSelectChange,
    handleLayoutDirectionChange,
    handleInpaintMethodChange,
    updateTextColor,
    updateLineSpacing,
    updateInlineAlign,
    updateBlockAlign,
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
