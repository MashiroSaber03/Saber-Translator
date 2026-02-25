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
import { DEFAULT_FONT_FAMILY } from '@/constants'
import type { TextDirection, InpaintMethod } from '@/types/bubble'
import {
  DEFAULT_WORKFLOW_MODE,
  WORKFLOW_MODE_CONFIGS,
  type WorkflowMode,
  type WorkflowModeConfig,
  type WorkflowRunRequest
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
})


/** 是否启用范围限制 */
const isRangeEnabled = ref(false)

/** 页面范围起始页 */
const pageRangeStart = ref(1)

/** 页面范围结束页 */
const pageRangeEnd = ref(1)

/** 当前工作流模式 */
const selectedWorkflowMode = ref<WorkflowMode>(DEFAULT_WORKFLOW_MODE)

/** 排版方向选项（用于CustomSelect） */
const layoutDirectionOptions = [
  { label: '自动 (根据检测)', value: 'auto' },
  { label: '竖向排版', value: 'vertical' },
  { label: '横向排版', value: 'horizontal' }
]

/** 填充方式选项（用于CustomSelect） */
const inpaintMethodOptions = [
  { label: '纯色填充', value: 'solid' },
  { label: 'LAMA修复 (速度优化)', value: 'lama_mpe' },
  { label: 'LAMA修复 (通用)', value: 'litelama' }
]

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
  return pageRangeStart.value >= 1 && 
         pageRangeEnd.value <= totalImages.value && 
         pageRangeStart.value <= pageRangeEnd.value
})

/** 是否可以翻译 */
const canTranslate = computed(() => 
  hasImages.value && !imageStore.isBatchTranslationInProgress
)

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
  return WORKFLOW_MODE_CONFIGS.find(cfg => cfg.mode === selectedWorkflowMode.value) ?? WORKFLOW_MODE_CONFIGS[0]!
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
    value: cfg.mode
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

/** 内置字体列表（确保始终显示） */
const BUILTIN_FONTS = [
  DEFAULT_FONT_FAMILY,
  'fonts/msyh.ttc',
  'fonts/simhei.ttf',
  'fonts/simsun.ttc'
]

/** 字体上传输入框引用 */
const fontUploadInput = ref<HTMLInputElement | null>(null)

/** 字体选择选项（用于CustomSelect） */
const fontSelectOptions = computed(() => {
  const options = fontList.value.map(font => ({
    label: getFontDisplayName(font),
    value: font
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

watch(supportsRangeForCurrentMode, (supports) => {
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
        const serverFonts = response.fonts.map((f) => 
          typeof f === 'object' ? f.path : f
        )
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
 * 获取字体显示名称
 */
function getFontDisplayName(fontPath: string): string {
  // 内置字体的中文名称映射（与后端保持一致）
  const fontNameMap: Record<string, string> = {
    'fonts/STXINGKA.TTF': '华文行楷',
    'fonts/STXINWEI.TTF': '华文新魏',
    'fonts/STZHONGS.TTF': '华文中宋',
    'fonts/STKAITI.TTF': '楷体',
    'fonts/STLITI.TTF': '隶书',
    'fonts/思源黑体SourceHanSansK-Bold.TTF': '思源黑体',
    'fonts/STSONG.TTF': '华文宋体',
    'fonts/msyh.ttc': '微软雅黑',
    'fonts/msyhbd.ttc': '微软雅黑粗体',
    'fonts/SIMYOU.TTF': '幼圆',
    'fonts/STFANGSO.TTF': '仿宋',
    'fonts/STHUPO.TTF': '华文琥珀',
    'fonts/STXIHEI.TTF': '华文细黑',
    'fonts/simkai.ttf': '中易楷体',
    'fonts/simfang.ttf': '中易仿宋',
    'fonts/simhei.ttf': '中易黑体',
    'fonts/SIMLI.TTF': '中易隶书',
    'fonts/simsun.ttc': '宋体'
  }
  
  // 先检查是否有预定义的中文名称
  if (fontNameMap[fontPath]) {
    return fontNameMap[fontPath]
  }
  
  // 从路径中提取文件名
  const fileName = fontPath.split('/').pop() || fontPath
  
  // 检查文件名是否有预定义名称（不区分大小写）
  for (const [path, name] of Object.entries(fontNameMap)) {
    const mapFileName = path.split('/').pop() || ''
    if (mapFileName.toLowerCase() === fileName.toLowerCase()) {
      return name
    }
  }
  
  // 移除扩展名
  return fileName.replace(/\.(ttf|ttc|otf)$/i, '')
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
    mode: selectedWorkflowMode.value
  }

  if (isRangeActiveForCurrentMode.value && isPageRangeValid.value) {
    payload.range = {
      startPage: pageRangeStart.value,
      endPage: pageRangeEnd.value
    }
  }

  emit('runWorkflow', payload)
}
</script>

<template>
  <aside id="settings-sidebar" class="settings-sidebar">
    <div class="card settings-card">
      <h2>翻译设置</h2>
      
      <!-- 文字设置折叠面板 -->
      <CollapsiblePanel
        title="文字设置"
        :default-expanded="true"
        class="settings-card"
      >
          <div class="settings-form">
            <!-- 字号设置 -->
            <div class="form-group">
              <label for="fontSize">字号大小:</label>
              <input 
                type="number" 
                id="fontSize" 
                :value="textStyle.fontSize"
                min="10"
                :disabled="textStyle.autoFontSize"
                :class="{ 'disabled-input': textStyle.autoFontSize }"
                :title="textStyle.autoFontSize ? '已启用自动字号，首次翻译时将自动计算' : ''"
                @input="updateFontSize"
              >
              <span class="auto-fontSize-option" title="勾选后，首次翻译时自动为每个气泡计算合适的字号">
                <input 
                  type="checkbox" 
                  id="autoFontSize"
                  :checked="textStyle.autoFontSize"
                  @change="updateAutoFontSize"
                >
                <label for="autoFontSize">自动计算初始字号</label>
              </span>
            </div>

            <!-- 字体选择 -->
            <div class="form-group">
              <label for="fontFamily">文本字体:</label>
              <CustomSelect
                :model-value="textStyle.fontFamily"
                :options="fontSelectOptions"
                @change="handleFontSelectChange"
              />
              <!-- 隐藏的字体上传输入框 -->
              <input 
                ref="fontUploadInput"
                type="file" 
                id="fontUpload" 
                accept=".ttf,.ttc,.otf" 
                style="display: none;"
                @change="handleFontUpload"
              >
            </div>

            <!-- 排版设置 -->
            <div class="form-group">
              <label for="layoutDirection">排版设置:</label>
              <CustomSelect
                :model-value="textStyle.layoutDirection"
                :options="layoutDirectionOptions"
                @change="handleLayoutDirectionChange"
              />
            </div>
            
            <!-- 文字颜色 -->
            <div class="form-group">
              <label for="textColor">文字颜色:</label>
              <div class="color-with-auto">
                <label class="auto-color-toggle" title="翻译时自动使用识别到的文字颜色">
                  <input 
                    type="checkbox" 
                    :checked="textStyle.useAutoTextColor"
                    @change="updateUseAutoTextColor"
                  >
                  <span>自动</span>
                </label>
                <input 
                  type="color" 
                  id="textColor" 
                  :value="textStyle.textColor"
                  :disabled="textStyle.useAutoTextColor"
                  @input="updateTextColor"
                >
              </div>
              <div v-if="textStyle.useAutoTextColor" class="auto-color-hint">
                💡 翻译时将自动使用识别到的文字颜色
              </div>
            </div>
            
            <!-- 描边设置 -->
            <div class="form-group">
              <span class="checkbox-label">
                <input 
                  type="checkbox" 
                  id="strokeEnabled"
                  :checked="textStyle.strokeEnabled"
                  @change="updateStrokeEnabled"
                >
                <label for="strokeEnabled">启用文本描边:</label>
              </span>
            </div>
            
            <Transition name="stroke-slide">
              <div v-if="textStyle.strokeEnabled" id="strokeOptions" class="stroke-options">
                <div class="form-group">
                  <label for="strokeColor">描边颜色:</label>
                  <input 
                    type="color" 
                    id="strokeColor" 
                    :value="textStyle.strokeColor"
                    @input="updateStrokeColor"
                  >
                </div>
                <div class="form-group">
                  <label for="strokeWidth">描边宽度 (px):</label>
                  <input 
                    type="number" 
                    id="strokeWidth" 
                    :value="textStyle.strokeWidth"
                    min="0" 
                    max="10"
                    style="width: 60px; padding: 5px;"
                    @input="updateStrokeWidth"
                  >
                  <div class="input-hint">0 表示无描边。</div>
                </div>
              </div>
            </Transition>
            
            <!-- 填充方式 -->
            <div class="form-group">
              <label for="useInpainting">气泡填充方式:</label>
              <CustomSelect
                :model-value="textStyle.inpaintMethod"
                :options="inpaintMethodOptions"
                @change="handleInpaintMethodChange"
              />
            </div>
            
            <!-- 填充颜色（仅纯色填充时显示，带动画） -->
            <Transition name="slide-fade">
              <div v-if="textStyle.inpaintMethod === 'solid'" id="solidColorOptions" class="form-group">
                <label for="fillColor">填充颜色:</label>
                <input 
                  type="color" 
                  id="fillColor" 
                  :value="textStyle.fillColor"
                  @input="updateFillColor"
                >
              </div>
            </Transition>
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
                >
                <label for="apply_selectAll">全选</label>
              </div>
              <hr>
              <div class="apply-option">
                <input type="checkbox" id="apply_fontSize" v-model="applyOptions.fontSize">
                <label for="apply_fontSize">字号</label>
              </div>
              <div class="apply-option">
                <input type="checkbox" id="apply_fontFamily" v-model="applyOptions.fontFamily">
                <label for="apply_fontFamily">字体</label>
              </div>
              <div class="apply-option">
                <input type="checkbox" id="apply_layoutDirection" v-model="applyOptions.layoutDirection">
                <label for="apply_layoutDirection">排版方向</label>
              </div>
              <div class="apply-option">
                <input type="checkbox" id="apply_textColor" v-model="applyOptions.textColor">
                <label for="apply_textColor">文字颜色</label>
              </div>
              <div class="apply-option">
                <input type="checkbox" id="apply_fillColor" v-model="applyOptions.fillColor">
                <label for="apply_fillColor">填充颜色</label>
              </div>
              <div class="apply-option">
                <input type="checkbox" id="apply_strokeEnabled" v-model="applyOptions.strokeEnabled">
                <label for="apply_strokeEnabled">描边开关</label>
              </div>
              <div class="apply-option">
                <input type="checkbox" id="apply_strokeColor" v-model="applyOptions.strokeColor">
                <label for="apply_strokeColor">描边颜色</label>
              </div>
              <div class="apply-option">
                <input type="checkbox" id="apply_strokeWidth" v-model="applyOptions.strokeWidth">
                <label for="apply_strokeWidth">描边宽度</label>
              </div>
            </div>
          </div>
      </CollapsiblePanel>

      <CollapsiblePanel
        title="指定范围"
        :default-expanded="false"
        class="settings-card"
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
                >
                <span>启用</span>
              </label>
              <span class="total-count">共 {{ totalImages }} 张</span>
            </div>

            <div v-if="!supportsRangeForCurrentMode" class="range-note">
              当前模式不支持指定范围
            </div>
            
            <!-- 页面范围输入（启用时显示） -->
            <div v-show="isRangeActiveForCurrentMode" class="page-range-inputs-compact">
              <input 
                type="number" 
                id="pageRangeStart" 
                :value="pageRangeStart"
                @input="updatePageRangeStart"
                placeholder="起始"
              >
              <span class="range-sep">~</span>
              <input 
                type="number" 
                id="pageRangeEnd" 
                :value="pageRangeEnd"
                @input="updatePageRangeEnd"
                placeholder="结束"
              >
            </div>
            
            <!-- 范围错误提示 -->
            <div v-if="isRangeActiveForCurrentMode && !isPageRangeValid && totalImages > 0" class="range-error-compact">
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
        <button 
          id="prevImageButton" 
          :disabled="!canGoPrevious"
          @click="emit('previous')"
        >
          上一张
        </button>
        <button 
          id="nextImageButton" 
          :disabled="!canGoNext"
          @click="emit('next')"
        >
          下一张
        </button>
      </div>
    </div>
  </aside>
</template>

<style scoped>
/* 设置侧边栏样式 - 匹配原版 #settings-sidebar 样式 */
.settings-sidebar {
  position: fixed;
  top: 70px; /* 为顶部导航栏留出空间 */
  left: 20px;
  width: 300px;
  height: calc(100vh - 90px);
  overflow-y: auto;
  padding-top: 10px;
  margin-right: 0;
  order: -1;
  display: flex;
  flex-direction: column;
  scrollbar-width: thin;
  scrollbar-color: #cbd5e0 #f8fafc;
  direction: rtl;
  z-index: 50;
}

/* CollapsiblePanel 组件样式覆盖（匹配 SettingsSidebar 设计风格） */
.settings-card.collapsible-panel {
  margin-top: 20px;
  border-top: 1px solid #eee;
  padding-top: 15px;
  padding-bottom: 15px;
  margin-bottom: 15px;
  transition: transform 0.2s, box-shadow 0.2s;
  border-radius: 8px;
  padding: 15px;
  background-color: #f8fafc;
}

.settings-card.collapsible-panel:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}

.settings-card :deep(.collapsible-header) {
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
  margin: 0;
  user-select: none;
  color: #3a4767;
  font-size: 1.2em;
  margin-bottom: 10px;
  margin-top: 0;
}

.settings-card :deep(.collapsible-header:hover) {
  color: #3498db;
}

.settings-card :deep(.toggle-icon) {
  margin-left: auto;
  color: #8492a6;
  font-size: 1em;
  transition: transform 0.3s ease;
}

.settings-card :deep(.toggle-icon:hover) {
  color: #3498db;
}

/* 表单组 - 匹配原版 .settings-form 样式 */
.settings-form {
  display: flex;
  flex-direction: column;
  gap: 0;
}

.settings-form > div {
  margin-bottom: 15px;
  position: relative;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 0;
  position: relative;
}

.form-group label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

.form-group input[type="number"],
.form-group select {
  width: 100%;
  padding: 12px;
  border: 1px solid #e0e6ed;
  border-radius: 8px;
  font-size: 1em;
  transition: border-color 0.3s, box-shadow 0.3s;
  background-color: #f9fafc;
}

.form-group input[type="number"]:focus,
.form-group select:focus {
  border-color: #3498db;
  box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
  outline: none;
}

.form-group input[type="color"] {
  width: 50px;
  height: 30px;
  padding: 2px;
  border: 1px solid #e0e6ed;
  border-radius: 4px;
  cursor: pointer;
  background-color: #f9fafc;
}

.form-group input[type="color"]:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* 颜色选择器与自动开关容器 */
.color-with-auto {
  display: flex;
  align-items: center;
  gap: 8px;
}

/* 自动颜色开关 */
.auto-color-toggle {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  cursor: pointer;
  white-space: nowrap;
  padding: 4px 8px;
  border-radius: 4px;
  background: #f5f5f5;
  border: 1px solid #e0e6ed;
  transition: all 0.2s ease;
}

.auto-color-toggle:hover {
  border-color: #3498db;
}

.auto-color-toggle:has(input:checked) {
  background: #e3f2fd;
  border-color: #3498db;
  color: #3498db;
}

.auto-color-toggle input {
  cursor: pointer;
}

/* 自动颜色提示 */
.auto-color-hint {
  padding: 6px 10px;
  margin-top: 6px;
  font-size: 11px;
  color: #3498db;
  background: #e3f2fd;
  border-radius: 4px;
  border-left: 3px solid #3498db;
}

/* 填充颜色过渡动画 */
.slide-fade-enter-active {
  transition: all 0.3s ease-out;
}

.slide-fade-leave-active {
  transition: all 0.2s ease-in;
}

.slide-fade-enter-from,
.slide-fade-leave-to {
  opacity: 0;
  max-height: 0;
  margin-bottom: 0;
  overflow: hidden;
}

.slide-fade-enter-to,
.slide-fade-leave-from {
  opacity: 1;
  max-height: 60px;
}

/* 描边选项过渡动画 */
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
  margin-top: 0;
  padding-top: 0;
  padding-bottom: 0;
  overflow: hidden;
}

.stroke-slide-enter-to,
.stroke-slide-leave-from {
  opacity: 1;
  max-height: 150px;
}

/* 自动字号选项 - 匹配原版布局（在字号输入框下方，一行显示） */
.auto-fontSize-option {
  display: flex;
  align-items: center;
  margin-top: 8px;
  width: 100%;
}

.auto-fontSize-option input[type="checkbox"] {
  margin-right: 5px;
  flex-shrink: 0;
  width: auto;
}

.auto-fontSize-option label {
  margin-bottom: 0;
  font-weight: normal;
  font-size: 0.9em;
  white-space: nowrap;
}
/* 复选框标签 */
.checkbox-label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

/* 描边选项 - 匹配原版 #strokeOptions 样式 */
.stroke-options {
  display: block;
  margin-left: 20px;
  border-left: 2px solid #eee;
  padding-left: 15px;
  margin-top: 5px;
}

.stroke-options .form-group {
  margin-bottom: 10px;
}

.stroke-options .form-group label {
  font-size: 0.95em;
}

.stroke-options .input-hint {
  font-size: 0.9em;
  color: #777;
  margin-top: 2px;
  display: block;
}

/* 应用设置组 */
.apply-settings-group {
  display: flex;
  align-items: stretch;
  margin-top: 12px;
  position: relative;
  width: 100%;
  height: 36px;
}

.settings-sidebar .apply-settings-group button.settings-button {
  flex: 1;
  min-width: 0;
  width: auto;
  margin: 0;
  border-radius: 6px 0 0 6px;
  padding: 0 10px;
  background: linear-gradient(135deg, #4a90d9 0%, #357abd 100%);
  color: white;
  border: none;
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  letter-spacing: 0.5px;
  transition: all 0.2s ease;
  text-align: center;
  line-height: 36px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.settings-sidebar .apply-settings-group button.settings-button:hover:not(:disabled) {
  background: linear-gradient(135deg, #357abd 0%, #2a6aad 100%);
  box-shadow: 0 2px 8px rgba(74, 144, 217, 0.35);
}

.settings-sidebar .apply-settings-group button.settings-button:disabled {
  background: #c5c5c5;
  cursor: not-allowed;
  box-shadow: none;
}

.settings-gear-btn {
  flex-shrink: 0;
  width: 36px;
  height: 36px;
  padding: 0;
  background: linear-gradient(135deg, #357abd 0%, #2a6aad 100%);
  color: white;
  border: none;
  border-left: 1px solid rgba(255,255,255,0.15);
  border-radius: 0 6px 6px 0;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.settings-gear-btn:hover {
  background: linear-gradient(135deg, #2a6aad 0%, #1f5a9d 100%);
}

.settings-gear-btn:disabled {
  background: #aaa;
  cursor: not-allowed;
}

/* 应用选项下拉菜单 - 匹配原版 .apply-settings-dropdown 样式 */
.apply-options-dropdown {
  position: absolute;
  bottom: 100%;
  left: 0;
  right: 0;
  background: var(--card-bg, #fff);
  border: 1px solid var(--border-color, #ddd);
  border-radius: 5px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  z-index: var(--z-overlay);
  padding: 10px;
  margin-bottom: 5px;
}

.apply-option {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 0;
  font-size: 0.85em;
  color: var(--text-color, #555);
  cursor: pointer;
}

.apply-option:hover {
  color: var(--color-primary, #007bff);
}

.apply-option input[type="checkbox"] {
  width: 14px;
  height: 14px;
  cursor: pointer;
}

.apply-options-dropdown hr {
  height: 1px;
  background: var(--border-color, #eee);
  margin: 6px 0;
  border: none;
}

/* 操作按钮组 */
.action-buttons {
  display: flex;
  flex-direction: column;
  gap: 0;
  margin-top: 16px;
}

.workflow-controls {
  margin-top: 16px;
  padding: 14px;
  border: 1px solid #d6dfed;
  border-radius: 12px;
  background: linear-gradient(180deg, #f8fbff 0%, #f3f7fd 100%);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.85);
  gap: 8px;
}

.workflow-controls .form-group {
  margin-bottom: 0;
}

.workflow-controls .form-group label {
  margin-bottom: 6px;
  font-size: 14px;
  font-weight: 700;
  color: #2f3d56;
  letter-spacing: 0.2px;
}

.workflow-controls :deep(.custom-select) {
  width: 100%;
  min-width: 0;
}

.workflow-controls :deep(.custom-select-trigger) {
  height: 42px;
  border-radius: 10px;
  border-color: #b8c5dc;
}

.workflow-controls :deep(.custom-select-value) {
  font-size: 15px;
  font-weight: 600;
  color: #1f2f47;
}

.workflow-meta {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-top: 2px;
}

.workflow-chip {
  display: inline-flex;
  align-items: center;
  height: 24px;
  padding: 0 9px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  color: #2d4568;
  background: #e6eefb;
  border: 1px solid #d4e1f7;
}

.workflow-chip.danger-chip {
  color: #9f2b2b;
  background: #ffe7e7;
  border-color: #ffcaca;
}

/* 按钮基础样式 - 匹配原版 */
.action-buttons button {
  width: 100%;
  padding: 14px 25px;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1.1em;
  transition: all 0.3s;
  margin-top: 15px;
  font-weight: 600;
  letter-spacing: 0.5px;
  position: relative;
  overflow: hidden;
}

.action-buttons button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
  box-shadow: none;
  transform: none;
}

.action-buttons button:before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: left 0.7s;
}

.action-buttons button:hover:not(:disabled):before {
  left: 100%;
}

.workflow-controls .workflow-run-button {
  margin-top: 0;
  min-height: 54px;
  padding: 0 14px;
  border-radius: 10px;
  font-size: 16px;
  font-weight: 700;
  letter-spacing: 0;
  line-height: 1.3;
  text-align: center;
  white-space: normal;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #3ea94a 0%, #59ba54 100%);
  box-shadow: 0 8px 16px rgba(62, 169, 74, 0.24);
}

.workflow-controls .workflow-run-button:hover:not(:disabled) {
  background: linear-gradient(135deg, #369740 0%, #50b14b 100%);
  box-shadow: 0 10px 18px rgba(54, 151, 64, 0.28);
  transform: translateY(-1px);
}

.workflow-controls .workflow-run-button.danger-button {
  background: linear-gradient(135deg, #d64242 0%, #bf3434 100%);
  box-shadow: 0 8px 16px rgba(214, 66, 66, 0.24);
}

.workflow-controls .workflow-run-button.danger-button:hover:not(:disabled) {
  background: linear-gradient(135deg, #bf3434 0%, #a92a2a 100%);
  box-shadow: 0 10px 18px rgba(191, 52, 52, 0.28);
}

.workflow-description {
  margin-top: 0;
  font-size: 13px;
  line-height: 1.45;
  color: #5b6f8e;
}
/* 导航按钮 - 匹配原版 .navigation-buttons 样式 */
.navigation-buttons {
  display: flex;
  gap: 10px;
  margin-top: 20px;
  justify-content: space-between;
  width: auto;
}

.navigation-buttons button {
  width: auto;
  padding: 12px 15px;
  background-color: #6c757d;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 0.9em;
  transition: background-color 0.3s;
  box-shadow: none;
  transform: none;
  font-weight: normal;
  white-space: nowrap;
  flex: 1;
}

.navigation-buttons button:hover:not(:disabled) {
  background-color: #5a6268;
}

.navigation-buttons button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

/* 禁用的输入框样式 */
.disabled-input {
  opacity: 0.5;
  cursor: not-allowed;
}

/* ===================================
   设置侧边栏样式 - 完整迁移自 sidebar.css
   =================================== */

.settings-sidebar {
  position: fixed;
  top: 20px;
  left: 20px;
  width: 300px;
  height: calc(100vh - 40px);
  overflow-y: auto;
  padding-top: 20px;
  margin-right: 0;
  order: -1;
  display: flex;
  flex-direction: column;
  scrollbar-width: thin;
  scrollbar-color: #cbd5e0 #f8fafc;
  direction: rtl;
}

.settings-sidebar > * {
  direction: ltr;
}

.settings-sidebar .settings-card {
  background-color: white;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  padding: 25px;
  margin-bottom: 15px;
  transition: box-shadow 0.2s;
}

.settings-sidebar .settings-card:hover {
  transform: none;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.settings-sidebar .settings-card h2 {
  border-bottom: 2px solid #f0f0f0;
  padding-bottom: 12px;
  margin-bottom: 20px;
  color: #2c3e50;
  font-size: 1.6em;
  text-align: center;
}

.settings-sidebar .settings-card h3 {
  color: #3a4767;
  font-size: 1.2em;
  margin-bottom: 15px;
  display: flex;
  align-items: center;
}

.settings-sidebar .settings-card h3 .toggle-icon {
  margin-left: auto;
  color: #8492a6;
  font-size: 1em;
}

.settings-sidebar .settings-card h3 .toggle-icon:hover {
  color: #3498db;
}

.settings-sidebar .settings-card h3 .toggle-icon:before {
  content: none;
}

.settings-sidebar .settings-form > div {
  margin-bottom: 15px;
  position: relative;
}

.settings-sidebar label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

.settings-sidebar select,
.settings-sidebar input[type="number"],
.settings-sidebar input[type="text"] {
  width: 100%;
  padding: 12px;
  border: 1px solid #e0e6ed;
  border-radius: 8px;
  font-size: 1em;
  transition: border-color 0.3s, box-shadow 0.3s;
  background-color: #f9fafc;
}

.settings-sidebar select:focus,
.settings-sidebar input[type="number"]:focus,
.settings-sidebar input[type="text"]:focus {
  border-color: #3498db;
  box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
  outline: none;
}

.settings-sidebar .input-hint {
  font-size: 0.9em;
  color: #777;
  margin-top: 0.2em;
  display: block;
}

.settings-sidebar .navigation-buttons {
  display: flex;
  gap: 10px;
  margin-top: 20px;
  justify-content: space-between;
  width: auto;
}

.settings-sidebar .navigation-buttons button {
  width: auto;
  padding: 12px 15px;
  background-color: #6c757d;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 0.9em;
  transition: background-color 0.3s;
  box-shadow: none;
  transform: none;
  font-weight: normal;
  white-space: nowrap;
}

.settings-sidebar .navigation-buttons button:hover {
  background-color: #5a6268;
}

.settings-sidebar .navigation-buttons button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.settings-sidebar #font-settings,
.settings-sidebar #page-range-settings {
  margin-top: 20px;
  border-top: 1px solid #eee;
  padding-top: 15px;
  padding-bottom: 15px;
  margin-bottom: 15px;
  transition: transform 0.2s, box-shadow 0.2s;
  border-radius: 8px;
  padding: 15px;
  background-color: #f8fafc;
}

.settings-sidebar #font-settings:hover,
.settings-sidebar #page-range-settings:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}

.settings-sidebar #font-settings h3,
.settings-sidebar #page-range-settings h3 {
  margin-bottom: 10px;
  margin-top: 0;
}

.settings-sidebar button#applyFontSettingsToAllButton {
  font-size: 0.9em;
  padding: 8px 15px;
  margin-top: 10px;
}

.settings-sidebar #font-settings + div {
  margin-top: 20px;
}

.settings-sidebar div.settings-card {
  margin-bottom: 15px;
  transition: box-shadow 0.2s;
}


.settings-sidebar #strokeEnabled {
  margin-left: 5px;
  transform: scale(1.1);
}

.settings-sidebar #strokeOptions label {
  font-size: 0.95em;
}

#solidColorOptions, #inpaintingOptions {
  margin-top: 10px;
  margin-bottom: 10px;
  overflow: visible;
}

.settings-sidebar button.settings-button {
  width: 100%;
  padding: 12px 25px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 1.1em;
  transition: background-color 0.3s;
  margin-top: 10px;
  box-shadow: none;
  transform: none;
  font-weight: normal;
  text-align: center;
  white-space: nowrap;
}

#settings-sidebar button.settings-button:hover {
  background-color: #0056b3;
}

#settings-sidebar button.settings-button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

/* ============================================================ */
/* 页面范围设置面板样式 */
/* ============================================================ */

/* ============================================================ */
/* 页面范围设置 - 紧凑样式 */
/* ============================================================ */

.page-range-form {
  padding: 8px 0;
}

/* 头部行：启用开关 + 图片数 */
.range-header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.range-toggle-compact {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  padding: 6px 12px;
  background: #f5f5f5;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 500;
  color: #666;
  transition: all 0.2s ease;
}

.range-toggle-compact:hover {
  border-color: #5c6bc0;
}

.range-toggle-compact:has(input:checked) {
  background: #e8eaf6;
  border-color: #5c6bc0;
  color: #3f51b5;
}

.range-toggle-compact input[type="checkbox"] {
  width: 14px;
  height: 14px;
  cursor: pointer;
  accent-color: #5c6bc0;
}

.total-count {
  font-size: 12px;
  color: #888;
}

.range-note {
  margin-bottom: 8px;
  font-size: 11px;
  color: #6b7c93;
}

/* 紧凑的范围输入 */
.page-range-inputs-compact {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}

.page-range-inputs-compact input {
  width: 52px;
  padding: 6px 4px;
  font-size: 13px;
  font-weight: 600;
  text-align: center;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.page-range-inputs-compact input:focus {
  border-color: #5c6bc0;
  outline: none;
}

.range-sep {
  font-size: 14px;
  color: #999;
}

/* 紧凑的错误提示 */
.range-error-compact {
  padding: 4px 8px;
  background: #ffebee;
  border-radius: 4px;
  font-size: 11px;
  color: #c62828;
  text-align: center;
}

/* 覆盖遗留 settings-button 通用规则，确保工作流按钮视觉和排版稳定 */
.settings-sidebar .workflow-controls button.workflow-run-button {
  margin-top: 0;
  min-height: 54px;
  padding: 0 14px;
  border-radius: 10px;
  font-size: 16px;
  font-weight: 700;
  letter-spacing: 0;
  line-height: 1.3;
  text-align: center;
  white-space: normal;
  display: flex;
  align-items: center;
  justify-content: center;
}

.settings-sidebar .workflow-controls button.workflow-run-button:disabled {
  color: #f3f6fa;
  background: #c1c8d1;
  box-shadow: none;
  transform: none;
}
</style>
