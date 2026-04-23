<script setup lang="ts">
/**
 * 设置侧边栏组件
 * 翻译页面左侧的设置面板，包含文字设置、操作按钮等
 *
 * 功能：
 * - 文字设置折叠面板（字号、字体、排版、颜色、描边、填充方式）
 * - 翻译操作按钮组
 * - 导航按钮
 */

import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { getFontList, uploadFont } from '@/api/config'
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
  type WorkflowMode,
  type WorkflowModeConfig,
  type WorkflowRunRequest,
} from '@/types/workflow'
import CustomSelect from '@/components/common/CustomSelect.vue'
import CollapsiblePanel from '@/components/common/CollapsiblePanel.vue'

// ============================================================
// Props 和 Emits
// ============================================================

const emit = defineEmits<{
  /** 启动工作流 */
  (e: 'runWorkflow', payload: WorkflowRunRequest): void
  /** 上一张图片 */
  (e: 'previous'): void
  /** 下一张图片 */
  (e: 'next'): void
  /** 应用设置到全部 */
  (e: 'applyToAll', options: ApplySettingsOptions): void
  /** 文字样式设置变更（需要重新渲染） */
  (e: 'textStyleChanged', settingKey: string, newValue: unknown): void
  /** 【复刻原版修复A】自动字号开关变更（需要特殊处理：重新计算字号或应用固定字号） */
  (e: 'autoFontSizeChanged', isAutoFontSize: boolean): void
}>()

// ============================================================
// 类型定义
// ============================================================

/** 应用设置选项 */
interface ApplySettingsOptions {
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

// ============================================================
// Stores
// ============================================================

const imageStore = useImageStore()
const settingsStore = useSettingsStore()

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

/** 是否启用范围限制 */
const isRangeEnabled = ref(false)

/** 页面范围起始页 */
const pageRangeStart = ref(1)

/** 页面范围结束页 */
const pageRangeEnd = ref(1)

/** 当前工作流模式 */
const selectedWorkflowMode = ref<WorkflowMode>(DEFAULT_WORKFLOW_MODE)

// ============================================================
// 计算属性
// ============================================================

/** 当前图片 */
const currentImage = computed(() => imageStore.currentImage)

/** 是否有图片 */
const hasImages = computed(() => imageStore.hasImages)

/** 总图片数量 */
const totalImages = computed(() => imageStore.images.length)

/** 页面范围是否有效 */
const isPageRangeValid = computed(() => {
  return (
    pageRangeStart.value >= 1 &&
    pageRangeEnd.value <= totalImages.value &&
    pageRangeStart.value <= pageRangeEnd.value
  )
})

/** 是否可以翻译 */
const canTranslate = computed(() => hasImages.value && !imageStore.isBatchTranslationInProgress)

/** 是否可以切换上一张 */
const canGoPrevious = computed(() => imageStore.canGoPrevious)

/** 是否可以切换下一张 */
const canGoNext = computed(() => imageStore.canGoNext)

/** 当前工作流是否可执行 */
const canRunWorkflow = computed(() => {
  const mode = selectedWorkflowMode.value
  const rangeInvalid = isRangeActiveForCurrentMode.value && !isPageRangeValid.value

  switch (mode) {
    case 'translate-current':
      return !!currentImage.value && canTranslate.value
    case 'translate-batch':
    case 'hq-batch':
    case 'proofread-batch':
      return canTranslate.value && !rangeInvalid
    case 'remove-current':
    case 'delete-current':
      return !!currentImage.value
    case 'remove-batch':
      return hasImages.value && !rangeInvalid
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

/** 当前模式是否支持范围处理 */
const supportsRangeForCurrentMode = computed(() => selectedWorkflowConfig.value.supportsRange)

/** 范围是否被激活且可用于当前模式 */
const isRangeActiveForCurrentMode = computed(() => {
  return supportsRangeForCurrentMode.value && isRangeEnabled.value
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
  if (isRangeActiveForCurrentMode.value && isPageRangeValid.value) {
    return `范围 ${pageRangeStart.value}-${pageRangeEnd.value}`
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
  return supportsRangeForCurrentMode.value ? '批量流程' : '单页流程'
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
      if (isRangeActiveForCurrentMode.value && isPageRangeValid.value) {
        return `当前范围：第 ${pageRangeStart.value}-${pageRangeEnd.value} 页。`
      }
      if (supportsRangeForCurrentMode.value) {
        return '当前范围：全部图片（可启用指定范围）。'
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

watch(supportsRangeForCurrentMode, supports => {
  if (!supports) {
    isRangeEnabled.value = false
  }
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
    // 后端返回的是 { fonts: [{file_name, display_name, path, is_default}, ...] }
    if (response.fonts && Array.isArray(response.fonts) && response.fonts.length > 0) {
      // 检查是新格式（对象数组）还是旧格式（字符串数组）
      const firstItem = response.fonts[0]
      if (typeof firstItem === 'object' && 'path' in firstItem) {
        // 新格式：提取字体路径
        const serverFonts = response.fonts.map(f => (typeof f === 'object' ? f.path : f))
        fontList.value = serverFonts
      } else {
        // 旧格式：直接使用
        fontList.value = response.fonts as string[]
      }
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
 * 【复刻原版修复A】切换后触发 autoFontSizeChanged 事件
 */
function updateAutoFontSize(event: Event) {
  const checked = (event.target as HTMLInputElement).checked
  settingsStore.updateTextStyle({ autoFontSize: checked })
  console.log(`自动字号设置变更: ${checked}`)
  // 【复刻原版】触发事件，由父组件处理重新渲染逻辑
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
  if (!target.closest('.apply-settings-group')) {
    showApplyOptions.value = false
  }
}

/**
 * 页面范围面板展开时初始化范围值
 */
function onPageRangeToggle(expanded: boolean) {
  if (expanded && totalImages.value > 0) {
    // 如果当前范围无效或未初始化（默认值1-1），重置为全范围
    if (!isPageRangeValid.value || (pageRangeStart.value === 1 && pageRangeEnd.value === 1)) {
      pageRangeStart.value = 1
      pageRangeEnd.value = totalImages.value
    }
  }
}

/**
 * 更新页面范围起始页
 * 【简化】只做边界限制，不自动修改结束页，避免用户输入时被干扰
 */
function updatePageRangeStart(event: Event) {
  const value = parseInt((event.target as HTMLInputElement).value)
  if (!isNaN(value)) {
    // 只限制在有效范围内，不联动修改结束页
    pageRangeStart.value = Math.max(1, Math.min(value, totalImages.value))
  }
}

/**
 * 更新页面范围结束页
 * 【简化】只做边界限制，不自动修改起始页，避免用户输入时被干扰
 */
function updatePageRangeEnd(event: Event) {
  const value = parseInt((event.target as HTMLInputElement).value)
  if (!isNaN(value)) {
    // 只限制在有效范围内，不联动修改起始页
    pageRangeEnd.value = Math.max(1, Math.min(value, totalImages.value))
  }
}

/**
 * 处理工作流模式切换
 */
function handleWorkflowModeChange(value: string | number) {
  selectedWorkflowMode.value = String(value) as WorkflowMode
}

/**
 * 启动当前工作流
 */
function handleRunWorkflow() {
  if (!canRunWorkflow.value) return

  const payload: WorkflowRunRequest = {
    mode: selectedWorkflowMode.value,
  }

  if (isRangeActiveForCurrentMode.value && isPageRangeValid.value) {
    payload.range = {
      startPage: pageRangeStart.value,
      endPage: pageRangeEnd.value,
    }
  }

  emit('runWorkflow', payload)
}
</script>

<template>
  <aside id="settings-sidebar" class="settings-sidebar">
    <div class="card settings-card">
      <h2 class="sidebar-title">翻译设置</h2>

      <!-- 文字设置折叠面板 -->
      <CollapsiblePanel
        title="文字设置"
        :default-expanded="true"
        class="settings-panel text-settings-panel"
      >
        <div class="settings-form text-settings-form">
          <section class="setting-group setting-group-typography">
            <div class="group-title-row">
              <h3 class="group-title">字体排版</h3>
              <span class="group-note">影响新翻译文本</span>
            </div>
            <div class="form-group">
              <label for="fontSize">字号</label>
              <input
                type="number"
                id="fontSize"
                :value="textStyle.fontSize"
                min="10"
                :disabled="textStyle.autoFontSize"
                :class="{ 'disabled-input': textStyle.autoFontSize }"
                :title="textStyle.autoFontSize ? '已启用自动字号，首次翻译时将自动计算' : ''"
                @input="updateFontSize"
              />
              <label
                class="toggle-pill auto-fontsize-toggle"
                for="autoFontSize"
                title="勾选后，首次翻译时自动为每个气泡计算合适的字号"
              >
                <input
                  type="checkbox"
                  id="autoFontSize"
                  :checked="textStyle.autoFontSize"
                  @change="updateAutoFontSize"
                />
                <span>自动计算初始字号</span>
              </label>
            </div>

            <div class="form-group">
              <label for="fontFamily">文本字体</label>
              <CustomSelect
                :model-value="textStyle.fontFamily"
                :options="fontSelectOptions"
                @change="handleFontSelectChange"
              />
              <input
                ref="fontUploadInput"
                type="file"
                id="fontUpload"
                accept=".ttf,.ttc,.otf"
                style="display: none"
                @change="handleFontUpload"
              />
            </div>

            <div class="form-group">
              <label for="layoutDirection">排版方向</label>
              <CustomSelect
                :model-value="textStyle.layoutDirection"
                :options="layoutDirectionOptions"
                @change="handleLayoutDirectionChange"
              />
            </div>

            <div class="form-group">
              <label for="lineSpacing">行间距</label>
              <input
                type="number"
                id="lineSpacing"
                :value="textStyle.lineSpacing"
                min="0.5"
                max="3"
                step="0.1"
                title="行间距倍数（0.5 - 3.0）"
                @change="updateLineSpacing"
              />
            </div>

            <div class="form-group">
              <label for="textAlign">对齐方式</label>
              <CustomSelect
                :model-value="textStyle.textAlign"
                :options="textAlignOptions"
                @change="updateTextAlign"
              />
            </div>
          </section>

          <section class="setting-group setting-group-color">
            <div class="group-title-row">
              <h3 class="group-title">颜色与填充</h3>
            </div>
            <div class="form-group">
              <div class="label-row">
                <label for="textColor">文字颜色</label>
                <label class="toggle-pill auto-color-toggle" title="翻译时自动使用识别到的文字颜色">
                  <input
                    type="checkbox"
                    :checked="textStyle.useAutoTextColor"
                    @change="updateUseAutoTextColor"
                  />
                  <span>自动</span>
                </label>
              </div>
              <input
                type="color"
                id="textColor"
                class="color-input"
                :value="textStyle.textColor"
                :disabled="textStyle.useAutoTextColor"
                @input="updateTextColor"
              />
              <div v-if="textStyle.useAutoTextColor" class="inline-hint">
                翻译时将自动使用识别到的文字颜色
              </div>
            </div>

            <div class="form-group">
              <label for="useInpainting">气泡填充方式</label>
              <CustomSelect
                :model-value="textStyle.inpaintMethod"
                :options="inpaintMethodOptions"
                @change="handleInpaintMethodChange"
              />
            </div>

            <Transition name="slide-fade">
              <div
                v-if="textStyle.inpaintMethod === 'solid'"
                id="solidColorOptions"
                class="form-group inline-color-group"
              >
                <label for="fillColor">填充颜色</label>
                <input
                  type="color"
                  id="fillColor"
                  class="color-input compact"
                  :value="textStyle.fillColor"
                  @input="updateFillColor"
                />
              </div>
            </Transition>
          </section>

          <section class="setting-group setting-group-stroke">
            <div class="group-title-row">
              <h3 class="group-title">描边</h3>
              <label class="toggle-pill stroke-toggle" for="strokeEnabled">
                <input
                  type="checkbox"
                  id="strokeEnabled"
                  :checked="textStyle.strokeEnabled"
                  @change="updateStrokeEnabled"
                />
                <span>启用描边</span>
              </label>
            </div>

            <Transition name="stroke-slide">
              <div v-if="textStyle.strokeEnabled" id="strokeOptions" class="stroke-options">
                <div class="stroke-grid">
                  <div class="form-group">
                    <label for="strokeColor">描边颜色</label>
                    <input
                      type="color"
                      id="strokeColor"
                      class="color-input compact"
                      :value="textStyle.strokeColor"
                      @input="updateStrokeColor"
                    />
                  </div>
                  <div class="form-group">
                    <label for="strokeWidth">描边宽度 (px)</label>
                    <input
                      type="number"
                      id="strokeWidth"
                      class="compact-number-input"
                      :value="textStyle.strokeWidth"
                      min="0"
                      max="10"
                      @input="updateStrokeWidth"
                    />
                    <div class="input-hint">0 表示无描边。</div>
                  </div>
                </div>
              </div>
            </Transition>
          </section>
        </div>

        <!-- 应用到全部按钮 -->
        <div class="apply-settings-group">
          <button
            type="button"
            class="settings-button"
            :disabled="!hasImages"
            @click="handleApplyToAll"
          >
            应用到全部
          </button>
          <button
            type="button"
            class="settings-gear-btn"
            title="选择要应用的参数"
            @click="toggleApplyOptions"
          >
            ⚙️
          </button>

          <!-- 应用选项下拉菜单 -->
          <div v-if="showApplyOptions" class="apply-options-dropdown">
            <div class="apply-option">
              <input
                type="checkbox"
                id="apply_selectAll"
                :checked="Object.values(applyOptions).every(v => v)"
                @change="toggleSelectAll"
              />
              <label for="apply_selectAll">全选</label>
            </div>
            <hr />
            <div class="apply-option">
              <input type="checkbox" id="apply_fontSize" v-model="applyOptions.fontSize" />
              <label for="apply_fontSize">字号</label>
            </div>
            <div class="apply-option">
              <input type="checkbox" id="apply_fontFamily" v-model="applyOptions.fontFamily" />
              <label for="apply_fontFamily">字体</label>
            </div>
            <div class="apply-option">
              <input
                type="checkbox"
                id="apply_layoutDirection"
                v-model="applyOptions.layoutDirection"
              />
              <label for="apply_layoutDirection">排版方向</label>
            </div>
            <div class="apply-option">
              <input type="checkbox" id="apply_lineSpacing" v-model="applyOptions.lineSpacing" />
              <label for="apply_lineSpacing">行间距</label>
            </div>
            <div class="apply-option">
              <input type="checkbox" id="apply_textAlign" v-model="applyOptions.textAlign" />
              <label for="apply_textAlign">对齐方式</label>
            </div>
            <div class="apply-option">
              <input type="checkbox" id="apply_textColor" v-model="applyOptions.textColor" />
              <label for="apply_textColor">文字颜色</label>
            </div>
            <div class="apply-option">
              <input type="checkbox" id="apply_fillColor" v-model="applyOptions.fillColor" />
              <label for="apply_fillColor">填充颜色</label>
            </div>
            <div class="apply-option">
              <input
                type="checkbox"
                id="apply_strokeEnabled"
                v-model="applyOptions.strokeEnabled"
              />
              <label for="apply_strokeEnabled">描边开关</label>
            </div>
            <div class="apply-option">
              <input type="checkbox" id="apply_strokeColor" v-model="applyOptions.strokeColor" />
              <label for="apply_strokeColor">描边颜色</label>
            </div>
            <div class="apply-option">
              <input type="checkbox" id="apply_strokeWidth" v-model="applyOptions.strokeWidth" />
              <label for="apply_strokeWidth">描边宽度</label>
            </div>
          </div>
        </div>
      </CollapsiblePanel>

      <CollapsiblePanel
        title="指定范围"
        :default-expanded="false"
        class="settings-panel"
        @toggle="onPageRangeToggle"
      >
        <div class="settings-form page-range-form">
          <!-- 启用开关 + 图片数 -->
          <div class="range-header-row">
            <label class="range-toggle-compact">
              <input
                type="checkbox"
                v-model="isRangeEnabled"
                :disabled="totalImages === 0 || !supportsRangeForCurrentMode"
              />
              <span>启用</span>
            </label>
            <span class="total-count">共 {{ totalImages }} 张</span>
          </div>

          <div v-if="!supportsRangeForCurrentMode" class="range-note">当前模式不支持指定范围</div>

          <!-- 页面范围输入（启用时显示） -->
          <div v-show="isRangeActiveForCurrentMode" class="page-range-inputs-compact">
            <input
              type="number"
              id="pageRangeStart"
              :value="pageRangeStart"
              @input="updatePageRangeStart"
              placeholder="起始"
            />
            <span class="range-sep">~</span>
            <input
              type="number"
              id="pageRangeEnd"
              :value="pageRangeEnd"
              @input="updatePageRangeEnd"
              placeholder="结束"
            />
          </div>

          <!-- 范围错误提示 -->
          <div
            v-if="isRangeActiveForCurrentMode && !isPageRangeValid && totalImages > 0"
            class="range-error-compact"
          >
            范围无效
          </div>
        </div>
      </CollapsiblePanel>

      <!-- 工作流启动区 -->
      <div class="action-buttons workflow-controls">
        <div class="form-group">
          <label for="workflowModeSelect">操作模式:</label>
          <CustomSelect
            id="workflowModeSelect"
            :model-value="selectedWorkflowMode"
            :options="workflowModeOptions"
            @change="handleWorkflowModeChange"
          />
        </div>
        <div class="workflow-meta">
          <span class="workflow-chip">{{ workflowContextTag }}</span>
          <span class="workflow-chip" :class="{ 'danger-chip': isDangerousWorkflow }">
            {{ workflowModeTag }}
          </span>
        </div>
        <button
          id="runWorkflowButton"
          class="settings-button workflow-run-button"
          :class="{ 'danger-button': isDangerousWorkflow }"
          :disabled="!canRunWorkflow"
          @click="handleRunWorkflow"
        >
          {{ workflowStartLabel }}
        </button>
        <div class="workflow-description">
          {{ workflowDescription }}
        </div>
      </div>

      <!-- 导航按钮 -->
      <div class="navigation-buttons">
        <button id="prevImageButton" :disabled="!canGoPrevious" @click="emit('previous')">
          上一张
        </button>
        <button id="nextImageButton" :disabled="!canGoNext" @click="emit('next')">下一张</button>
      </div>
    </div>
  </aside>
</template>

<style scoped>
/* 侧边栏容器 */
.settings-sidebar {
  position: fixed;
  top: 70px;
  left: 20px;
  width: 300px;
  height: calc(100vh - 90px);
  overflow-y: auto;
  padding-top: 10px;
  display: flex;
  flex-direction: column;
  direction: rtl;
  z-index: 50;
  scrollbar-width: thin;
  scrollbar-color: #c7d5e7 #eef3f9;
}

.settings-sidebar > * {
  direction: ltr;
}

.settings-sidebar::-webkit-scrollbar {
  width: 8px;
}

.settings-sidebar::-webkit-scrollbar-track {
  background: #eef3f9;
  border-radius: 999px;
}

.settings-sidebar::-webkit-scrollbar-thumb {
  background: #c7d5e7;
  border-radius: 999px;
}

/* 顶层卡片 */
.settings-card {
  background: #fff;
  border: 1px solid #dbe4ef;
  border-radius: 14px;
  box-shadow: 0 8px 20px rgba(28, 45, 72, 0.07);
  padding: 18px;
  margin-bottom: 14px;
}

.sidebar-title {
  margin: 0 0 14px;
  padding-bottom: 12px;
  border-bottom: 1px solid #e2e9f2;
  color: #20314f;
  font-size: 24px;
  font-weight: 700;
  text-align: center;
}

/* 面板层：恢复轻量容器，形成“2层结构” */
.settings-panel {
  margin: 0 0 12px;
  padding: 12px;
  border: 1px solid #d8e3f1;
  border-radius: 12px;
  background: #f5f8fd;
}

.settings-panel :deep(.collapsible-header) {
  margin: 0 0 10px;
  padding: 0;
  color: #304464;
  border-bottom: 1px solid #dde7f4;
  padding-bottom: 8px;
}

.settings-panel :deep(.collapsible-title) {
  font-size: 17px;
  font-weight: 700;
}

.settings-panel :deep(.toggle-icon) {
  color: #6e81a2;
  font-size: 12px;
}

.settings-panel :deep(.collapsible-content) {
  padding-top: 2px;
}

.settings-form {
  display: flex;
  flex-direction: column;
}

.setting-group {
  --group-divider-color: #d4deeb;
  margin: 0;
  padding: 10px 0;
  border-radius: 0;
  background: transparent;
  box-shadow: none;
}

.setting-group:last-child {
  margin-bottom: 0;
}

.setting-group + .setting-group {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 3px solid var(--group-divider-color);
}

.group-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 12px;
  padding: 0 0 10px;
  border-bottom: 1px solid #dfe8f4;
}

.setting-group-typography {
  --group-divider-color: #d4deeb;
}

.setting-group-color {
  --group-divider-color: #24a87a;
}

.setting-group-stroke {
  --group-divider-color: #dc9a2f;
}

.group-title {
  margin: 0;
  color: #273959;
  font-size: 14px;
  font-weight: 700;
  line-height: 1.2;
}

.group-note {
  color: #7d8ba4;
  font-size: 11px;
  line-height: 1.2;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 11px;
}

.form-group:last-child {
  margin-bottom: 0;
}

.form-group label {
  margin: 0;
  color: #2f3d56;
  font-size: 13px;
  font-weight: 600;
}

.label-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.settings-sidebar input[type='number'],
.settings-sidebar input[type='text'],
.settings-sidebar select {
  width: 100%;
  min-height: 40px;
  padding: 9px 10px;
  border: 1px solid #cfdcec;
  border-radius: 8px;
  font-size: 14px;
  color: #1f2f47;
  background: #fbfdff;
  transition:
    border-color 0.2s ease,
    box-shadow 0.2s ease;
}

.settings-sidebar input[type='number']:focus,
.settings-sidebar input[type='text']:focus,
.settings-sidebar select:focus {
  border-color: #4a82ce;
  box-shadow: 0 0 0 3px rgba(74, 130, 206, 0.18);
  outline: none;
}

.disabled-input {
  opacity: 0.55;
  cursor: not-allowed;
}

.toggle-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  width: fit-content;
  padding: 5px 10px;
  border: 1px solid #d3deed;
  border-radius: 999px;
  background: #f4f8fd;
  color: #5b6f8e;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  user-select: none;
}

.toggle-pill input[type='checkbox'] {
  width: 14px;
  height: 14px;
  margin: 0;
  accent-color: #4a82ce;
  cursor: pointer;
}

.toggle-pill:has(input:checked) {
  border-color: #94b5e5;
  background: #e9f2ff;
  color: #21579c;
}

.auto-fontsize-toggle {
  margin-top: 2px;
}

.color-input {
  width: 58px;
  height: 34px;
  padding: 2px;
  border: 1px solid #cfdcec;
  border-radius: 8px;
  background: #fff;
  cursor: pointer;
}

.color-input.compact {
  width: 72px;
}

.color-input:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.inline-color-group {
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
}

.inline-hint {
  color: #3a6ea7;
  font-size: 12px;
  line-height: 1.35;
  padding: 6px 8px;
  border: 1px solid #d2e2fa;
  border-radius: 8px;
  background: #edf4ff;
}

.stroke-options {
  margin-top: 8px;
  padding: 8px 0 0;
  border-top: 1px dashed #d7e2ef;
  border-radius: 0;
  background: transparent;
}

.stroke-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.compact-number-input {
  width: 100%;
  min-height: 36px;
}

.input-hint {
  color: #6f8099;
  font-size: 11px;
  line-height: 1.3;
}

/* 展开/收起动画 */
.slide-fade-enter-active {
  transition: all 0.28s ease-out;
}

.slide-fade-leave-active {
  transition: all 0.2s ease-in;
}

.slide-fade-enter-from,
.slide-fade-leave-to {
  opacity: 0;
  max-height: 0;
  overflow: hidden;
}

.slide-fade-enter-to,
.slide-fade-leave-from {
  opacity: 1;
  max-height: 70px;
}

.stroke-slide-enter-active {
  transition: all 0.3s ease-out;
}

.stroke-slide-leave-active {
  transition: all 0.2s ease-in;
}

.stroke-slide-enter-from,
.stroke-slide-leave-to {
  opacity: 0;
  max-height: 0;
  overflow: hidden;
}

.stroke-slide-enter-to,
.stroke-slide-leave-from {
  opacity: 1;
  max-height: 220px;
}

/* 应用到全部 */
.apply-settings-group {
  display: flex;
  align-items: stretch;
  position: relative;
  margin-top: 14px;
  width: 100%;
  height: 38px;
}

.apply-settings-group .settings-button {
  flex: 1;
  min-width: 0;
  margin: 0;
  border: none;
  border-radius: 8px 0 0 8px;
  background: linear-gradient(135deg, #4b89d0 0%, #316fb6 100%);
  color: #fff;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.apply-settings-group .settings-button:hover:not(:disabled) {
  background: linear-gradient(135deg, #3f7bc4 0%, #2b64a9 100%);
}

.apply-settings-group .settings-button:disabled {
  background: #c2c9d4;
  cursor: not-allowed;
}

.settings-gear-btn {
  width: 38px;
  border: none;
  border-left: 1px solid rgba(255, 255, 255, 0.24);
  border-radius: 0 8px 8px 0;
  background: linear-gradient(135deg, #316fb6 0%, #285d99 100%);
  color: #fff;
  font-size: 14px;
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.settings-gear-btn:hover {
  background: linear-gradient(135deg, #2a64a5 0%, #224f82 100%);
}

.apply-options-dropdown {
  position: absolute;
  inset: auto 0 calc(100% + 6px) 0;
  padding: 10px;
  border: 1px solid #d7e2f2;
  border-radius: 10px;
  background: #fff;
  box-shadow: 0 12px 24px rgba(22, 37, 58, 0.16);
  max-height: 260px;
  overflow-y: auto;
  z-index: var(--z-overlay);
}

.apply-option {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 26px;
  color: #405473;
  font-size: 13px;
  cursor: pointer;
}

.apply-option input[type='checkbox'] {
  width: 14px;
  height: 14px;
  margin: 0;
  accent-color: #4b89d0;
}

.apply-option:hover {
  color: #2b5f9d;
}

.apply-options-dropdown hr {
  margin: 6px 0;
  border: none;
  border-top: 1px solid #e3ebf6;
}

/* 页面范围面板 */
.page-range-form {
  gap: 8px;
}

.range-header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.range-toggle-compact {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border: 1px solid #d4deed;
  border-radius: 999px;
  background: #f4f8fd;
  color: #5d7090;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}

.range-toggle-compact:has(input:checked) {
  border-color: #94b5e5;
  background: #e9f2ff;
  color: #21579c;
}

.range-toggle-compact input[type='checkbox'] {
  width: 14px;
  height: 14px;
  margin: 0;
  accent-color: #4a82ce;
}

.total-count {
  color: #6f809a;
  font-size: 12px;
  font-weight: 500;
}

.range-note {
  color: #6f8099;
  font-size: 12px;
}

.page-range-inputs-compact {
  display: flex;
  align-items: center;
  gap: 6px;
}

.page-range-inputs-compact input {
  width: 58px;
  min-height: 34px;
  padding: 6px 4px;
  border: 1px solid #ccd9ea;
  border-radius: 8px;
  text-align: center;
  font-size: 13px;
  font-weight: 600;
}

.page-range-inputs-compact input:focus {
  border-color: #4a82ce;
  outline: none;
}

.range-sep {
  color: #7a8aa4;
  font-size: 14px;
  font-weight: 600;
}

.range-error-compact {
  color: #b73535;
  font-size: 12px;
  font-weight: 600;
  padding: 4px 8px;
  border: 1px solid #f3cccc;
  border-radius: 8px;
  background: #fff1f1;
  text-align: center;
}

/* 工作流区 */
.action-buttons {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 14px;
}

.workflow-controls {
  padding: 12px;
  border: 1px solid #d8e3f1;
  border-radius: 12px;
  background: #f8fbff;
}

.workflow-controls .form-group {
  margin-bottom: 0;
}

.workflow-controls .form-group label {
  margin-bottom: 6px;
}

.workflow-controls :deep(.custom-select) {
  width: 100%;
  min-width: 0;
}

.workflow-controls :deep(.custom-select-trigger) {
  min-height: 42px;
  border-radius: 10px;
  border-color: #b8c6dd;
}

.workflow-meta {
  display: flex;
  gap: 8px;
  align-items: center;
}

.workflow-chip {
  display: inline-flex;
  align-items: center;
  height: 24px;
  padding: 0 9px;
  border: 1px solid #d3e1f6;
  border-radius: 999px;
  background: #e8f0fd;
  color: #2d4568;
  font-size: 12px;
  font-weight: 600;
}

.workflow-chip.danger-chip {
  border-color: #ffcaca;
  background: #ffe7e7;
  color: #9f2b2b;
}

.workflow-run-button {
  min-height: 54px;
  border: none;
  border-radius: 10px;
  background: linear-gradient(135deg, #3ea94a 0%, #58ba54 100%);
  box-shadow: 0 8px 16px rgba(62, 169, 74, 0.24);
  color: #fff;
  font-size: 16px;
  font-weight: 700;
  cursor: pointer;
  transition:
    transform 0.2s ease,
    box-shadow 0.2s ease;
}

.workflow-run-button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 10px 18px rgba(54, 151, 64, 0.28);
}

.workflow-run-button.danger-button {
  background: linear-gradient(135deg, #d64242 0%, #bf3434 100%);
  box-shadow: 0 8px 16px rgba(214, 66, 66, 0.24);
}

.workflow-run-button.danger-button:hover:not(:disabled) {
  box-shadow: 0 10px 18px rgba(191, 52, 52, 0.28);
}

.workflow-run-button:disabled {
  background: #c1c8d1;
  box-shadow: none;
  cursor: not-allowed;
}

.workflow-description {
  color: #5c6f8f;
  font-size: 13px;
  line-height: 1.45;
}

/* 翻页按钮 */
.navigation-buttons {
  display: flex;
  gap: 10px;
  margin-top: 16px;
}

.navigation-buttons button {
  flex: 1;
  min-height: 38px;
  border: none;
  border-radius: 8px;
  background: #6c7784;
  color: #fff;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.navigation-buttons button:hover:not(:disabled) {
  background: #5a6572;
}

.navigation-buttons button:disabled {
  background: #c2c9d4;
  cursor: not-allowed;
}

@media (max-height: 860px) {
  .settings-sidebar {
    top: 66px;
    height: calc(100vh - 80px);
  }

  .sidebar-title {
    font-size: 22px;
  }
}
</style>
