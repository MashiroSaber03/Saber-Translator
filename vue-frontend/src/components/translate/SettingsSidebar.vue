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

import { ref, computed, onMounted } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { getFontList, uploadFont } from '@/api/config'
import { showToast } from '@/utils/toast'
import type { TextDirection, InpaintMethod } from '@/types/bubble'
import CustomSelect from '@/components/common/CustomSelect.vue'

// ============================================================
// Props 和 Emits
// ============================================================

const emit = defineEmits<{
  /** 翻译当前图片 */
  (e: 'translateCurrent'): void
  /** 翻译所有图片 */
  (e: 'translateAll'): void
  /** 高质量翻译 */
  (e: 'hqTranslate'): void
  /** AI 校对 */
  (e: 'proofread'): void
  /** 仅消除文字 */
  (e: 'removeText'): void
  /** 消除所有图片文字 */
  (e: 'removeAllText'): void
  /** 重新翻译失败图片 */
  (e: 'retryFailed'): void
  /** 删除当前图片 */
  (e: 'deleteCurrent'): void
  /** 清除所有图片 */
  (e: 'clearAll'): void
  /** 清理临时文件 */
  (e: 'cleanTemp'): void
  /** 打开插件管理 */
  (e: 'openPlugins'): void
  /** 打开设置 */
  (e: 'openSettings'): void
  /** 上一张图片 */
  (e: 'previous'): void
  /** 下一张图片 */
  (e: 'next'): void
  /** 应用设置到全部 */
  (e: 'applyToAll', options: ApplySettingsOptions): void
  /** 文字样式设置变更（需要重新渲染） */
  (e: 'textStyleChanged', settingKey: string, newValue: unknown): void
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

/** 文字设置面板是否展开 */
const isFontSettingsExpanded = ref(true)

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

/** 是否可以翻译 */
const canTranslate = computed(() => 
  hasImages.value && !imageStore.isBatchTranslationInProgress
)

/** 是否可以切换上一张 */
const canGoPrevious = computed(() => imageStore.canGoPrevious)

/** 是否可以切换下一张 */
const canGoNext = computed(() => imageStore.canGoNext)

/** 是否有翻译失败的图片 */
const hasFailedImages = computed(() => imageStore.failedImageCount > 0)

/** 失败图片数量 */
const failedImageCount = computed(() => imageStore.failedImageCount)

/** 文字样式设置 */
const textStyle = computed(() => settingsStore.textStyle)

/** 字体列表（包含内置字体） */
const fontList = ref<string[]>([])

/** 内置字体列表（确保始终显示） */
const BUILTIN_FONTS = [
  'fonts/STSONG.TTF',
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
 * 切换文字设置面板展开状态
 */
function toggleFontSettings() {
  isFontSettingsExpanded.value = !isFontSettingsExpanded.value
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
 */
function updateAutoFontSize(event: Event) {
  const checked = (event.target as HTMLInputElement).checked
  settingsStore.updateTextStyle({ autoFontSize: checked })
  console.log(`自动字号设置变更: ${checked} (仅影响下次翻译)`)
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
function handleFontSelectChange(value: string) {
  if (value === 'custom-font') {
    fontUploadInput.value?.click()
    return
  }
  settingsStore.updateTextStyle({ fontFamily: value })
  emit('textStyleChanged', 'fontFamily', value)
}

/**
 * 处理排版方向变化（CustomSelect）
 */
function handleLayoutDirectionChange(value: string) {
  settingsStore.updateTextStyle({ layoutDirection: value as TextDirection })
  emit('textStyleChanged', 'layoutDirection', value)
}

/**
 * 处理填充方式变化（CustomSelect）
 */
function handleInpaintMethodChange(value: string) {
  settingsStore.updateTextStyle({ inpaintMethod: value as InpaintMethod })
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

// 监听点击外部事件
if (typeof window !== 'undefined') {
  window.addEventListener('click', handleClickOutside)
}
</script>

<template>
  <aside id="settings-sidebar" class="settings-sidebar">
    <div class="card settings-card">
      <h2>翻译设置</h2>
      
      <!-- 文字设置折叠面板 -->
      <div id="font-settings" class="settings-card collapsible-panel">
        <h3 
          class="collapsible-header"
          @click="toggleFontSettings"
        >
          文字设置 
          <span class="toggle-icon">{{ isFontSettingsExpanded ? '▼' : '▶' }}</span>
        </h3>
        
        <div v-show="isFontSettingsExpanded" class="collapsible-content">
          <div class="settings-form">
            <!-- 字号设置 -->
            <div class="form-group">
              <label for="fontSize">字号大小:</label>
              <input 
                type="number" 
                id="fontSize" 
                :value="textStyle.fontSize"
                min="10" 
                max="100"
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
              <input 
                type="color" 
                id="textColor" 
                :value="textStyle.textColor"
                @input="updateTextColor"
              >
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
        </div>
      </div>

      <!-- 操作按钮组 -->
      <div class="action-buttons">
        <button 
          id="translateButton" 
          :disabled="!canTranslate"
          @click="emit('translateCurrent')"
        >
          翻译当前图片
        </button>
        <button 
          id="translateAllButton" 
          :disabled="!canTranslate"
          @click="emit('translateAll')"
        >
          翻译所有图片
        </button>
        <button 
          id="startHqTranslationBtn" 
          class="settings-button purple-button" 
          :disabled="!canTranslate"
          title="使用高质量翻译模式（需在设置中配置）"
          @click="emit('hqTranslate')"
        >
          高质量翻译
        </button>
        <button 
          id="proofreadButton" 
          :disabled="!canTranslate"
          @click="emit('proofread')"
        >
          AI校对
        </button>
        <!-- 重新翻译失败图片按钮 -->
        <button 
          v-if="hasFailedImages"
          id="retryFailedButton" 
          class="settings-button warning-button"
          :disabled="!canTranslate"
          title="重新翻译所有失败的图片"
          @click="emit('retryFailed')"
        >
          重新翻译失败图片 ({{ failedImageCount }})
        </button>
        <button 
          id="removeTextOnlyButton" 
          :disabled="!currentImage"
          title="消除图片中的气泡文字，无需填写翻译服务商和API Key"
          @click="emit('removeText')"
        >
          仅消除文字
        </button>
        <button 
          id="removeAllTextButton" 
          :disabled="!hasImages"
          title="消除所有图片中的气泡文字，无需填写翻译服务商和API Key"
          @click="emit('removeAllText')"
        >
          消除所有图片文字
        </button>
        <button 
          id="deleteCurrentImageButton" 
          class="settings-button red-button" 
          :disabled="!currentImage"
          @click="emit('deleteCurrent')"
        >
          删除当前图片
        </button>
        <button 
          id="clearAllImagesButton" 
          class="settings-button red-button"
          :disabled="!hasImages"
          @click="emit('clearAll')"
        >
          清除所有图片
        </button>
        <button 
          id="cleanDebugFilesButton" 
          class="settings-button orange-button"
          @click="emit('cleanTemp')"
        >
          清理临时文件
        </button>
        <button 
          id="managePluginsButton" 
          class="settings-button blue-button"
          @click="emit('openPlugins')"
        >
          插件管理
        </button>
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
  overflow-y: auto !important;
  padding-top: 10px;
  box-sizing: border-box;
  margin-right: 0;
  order: -1;
  display: flex;
  flex-direction: column;
  scrollbar-width: thin;
  scrollbar-color: #cbd5e0 #f8fafc;
  direction: rtl;
  z-index: 50;
}

.settings-sidebar > * {
  direction: ltr;
}

/* 设置卡片 - 匹配原版 .settings-card 样式 */
.settings-sidebar :deep(.card),
.settings-sidebar .settings-card {
  background-color: white;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  padding: 25px;
  margin-bottom: 15px;
  transition: box-shadow 0.2s;
}

.settings-sidebar :deep(.card):hover,
.settings-sidebar .settings-card:hover {
  transform: none;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.settings-sidebar :deep(.card) h2,
.settings-sidebar .settings-card h2 {
  border-bottom: 2px solid #f0f0f0;
  padding-bottom: 12px;
  margin-bottom: 20px;
  color: #2c3e50;
  font-size: 1.6em;
  text-align: center;
}

/* 折叠面板 - 匹配原版 #font-settings 样式 */
.collapsible-panel {
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

.collapsible-panel:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}

.collapsible-header {
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

.collapsible-header:hover {
  color: #3498db;
}

.toggle-icon {
  margin-left: auto;
  color: #8492a6;
  font-size: 1em;
  transition: transform 0.3s ease;
}

.toggle-icon:hover {
  color: #3498db;
}

.collapsible-content {
  overflow: visible;
  max-height: none;
  transition: max-height 0.3s ease;
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
  box-sizing: border-box;
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
  margin-bottom: 0 !important;
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
  margin-top: 0 !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
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

/* 应用设置组 - 匹配原版 .apply-settings-group 样式 */
.apply-settings-group {
  display: flex;
  align-items: stretch;
  gap: 0;
  margin-top: 10px;
  position: relative;
}

.apply-settings-group .settings-button {
  flex: 1;
  margin-top: 0 !important;
  border-radius: 5px 0 0 5px !important;
  padding: 12px 25px;
  background-color: #007bff;
  color: white;
  border: none;
  cursor: pointer;
  font-size: 1.1em;
  transition: background-color 0.3s;
  text-align: center;
  white-space: nowrap;
}

.apply-settings-group .settings-button:hover:not(:disabled) {
  background-color: #0056b3;
}

.apply-settings-group .settings-button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.settings-gear-btn {
  padding: 8px 12px;
  background-color: #0056b3;
  color: white;
  border: none;
  border-left: 1px solid rgba(255,255,255,0.3);
  border-radius: 0 5px 5px 0;
  cursor: pointer;
  font-size: 1em;
  transition: background-color 0.3s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.settings-gear-btn:hover {
  background-color: #004494;
}

.settings-gear-btn:disabled {
  background-color: #999;
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
  z-index: 9999;
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
  color: var(--primary-color, #007bff);
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
  background-color: #ccc !important;
  cursor: not-allowed;
  box-shadow: none !important;
  transform: none !important;
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

/* 翻译按钮 - 绿色 */
.action-buttons button#translateButton,
.action-buttons button#translateAllButton {
  background: linear-gradient(135deg, #4cae4c 0%, #5cb85c 100%);
  box-shadow: 0 4px 6px rgba(92, 184, 92, 0.2);
}

.action-buttons button#translateButton:hover:not(:disabled),
.action-buttons button#translateAllButton:hover:not(:disabled) {
  background: linear-gradient(135deg, #449d44 0%, #5cb85c 100%);
  box-shadow: 0 6px 10px rgba(92, 184, 92, 0.3);
  transform: translateY(-2px);
}

/* 校对按钮和高质量翻译按钮 - 紫色 */
.action-buttons button#proofreadButton,
.action-buttons button#startHqTranslationBtn,
.action-buttons .purple-button {
  background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%) !important;
  box-shadow: 0 4px 6px rgba(142, 68, 173, 0.2);
}

.action-buttons button#proofreadButton:hover:not(:disabled),
.action-buttons button#startHqTranslationBtn:hover:not(:disabled),
.action-buttons .purple-button:hover:not(:disabled) {
  background: linear-gradient(135deg, #8e44ad 0%, #6c3483 100%) !important;
  box-shadow: 0 6px 10px rgba(142, 68, 173, 0.3);
  transform: translateY(-2px);
}

/* 消除文字按钮 - 蓝色 */
.action-buttons button#removeTextOnlyButton,
.action-buttons button#removeAllTextButton {
  background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
  box-shadow: 0 4px 6px rgba(52, 152, 219, 0.2);
}

.action-buttons button#removeTextOnlyButton:hover:not(:disabled),
.action-buttons button#removeAllTextButton:hover:not(:disabled) {
  background: linear-gradient(135deg, #2980b9 0%, #1c6ea4 100%);
  box-shadow: 0 6px 10px rgba(52, 152, 219, 0.3);
  transform: translateY(-2px);
}

/* 删除按钮 - 红色 */
.action-buttons button#deleteCurrentImageButton,
.action-buttons button#clearAllImagesButton,
.action-buttons .red-button {
  background: linear-gradient(135deg, #dc3545 0%, #c82333 100%) !important;
  box-shadow: 0 4px 6px rgba(220, 53, 69, 0.2);
}

.action-buttons button#deleteCurrentImageButton:hover:not(:disabled),
.action-buttons button#clearAllImagesButton:hover:not(:disabled),
.action-buttons .red-button:hover:not(:disabled) {
  background: linear-gradient(135deg, #bd2130 0%, #c82333 100%) !important;
  box-shadow: 0 6px 10px rgba(220, 53, 69, 0.3);
  transform: translateY(-2px);
}

/* 清理临时文件按钮 - 橙色 */
.action-buttons button#cleanDebugFilesButton,
.action-buttons .orange-button {
  background: linear-gradient(135deg, #eb9316 0%, #f0ad4e 100%) !important;
  box-shadow: 0 4px 6px rgba(240, 173, 78, 0.2);
}

.action-buttons button#cleanDebugFilesButton:hover:not(:disabled),
.action-buttons .orange-button:hover:not(:disabled) {
  background: linear-gradient(135deg, #e67e22 0%, #d35400 100%) !important;
  box-shadow: 0 6px 10px rgba(211, 84, 0, 0.3);
  transform: translateY(-2px);
}

/* 重试失败按钮 - 橙色 */
.action-buttons button#retryFailedButton,
.action-buttons .warning-button {
  background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%) !important;
  box-shadow: 0 4px 6px rgba(243, 156, 18, 0.2);
}

.action-buttons button#retryFailedButton:hover:not(:disabled),
.action-buttons .warning-button:hover:not(:disabled) {
  background: linear-gradient(135deg, #e67e22 0%, #d35400 100%) !important;
  box-shadow: 0 6px 10px rgba(243, 156, 18, 0.3);
  transform: translateY(-2px);
}

/* 插件管理按钮 - 蓝色 */
.action-buttons button#managePluginsButton,
.action-buttons .blue-button {
  background: linear-gradient(135deg, #3498db 0%, #2980b9 100%) !important;
  box-shadow: 0 4px 6px rgba(52, 152, 219, 0.2);
}

.action-buttons button#managePluginsButton:hover:not(:disabled),
.action-buttons .blue-button:hover:not(:disabled) {
  background: linear-gradient(135deg, #2980b9 0%, #1c6ea4 100%) !important;
  box-shadow: 0 6px 10px rgba(52, 152, 219, 0.3);
  transform: translateY(-2px);
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

/* 暗色主题适配 */
[data-theme="dark"] .settings-sidebar {
  scrollbar-color: #4a5568 #2d3748;
}

[data-theme="dark"] .settings-sidebar :deep(.card),
[data-theme="dark"] .settings-sidebar .settings-card {
  background-color: #2d3748;
}

[data-theme="dark"] .settings-sidebar :deep(.card) h2,
[data-theme="dark"] .settings-sidebar .settings-card h2 {
  color: #e2e8f0;
  border-bottom-color: #4a5568;
}

[data-theme="dark"] .collapsible-panel {
  background-color: #1a202c;
  border-top-color: #4a5568;
}

[data-theme="dark"] .collapsible-header {
  color: #e2e8f0;
}

[data-theme="dark"] .form-group label {
  color: #a0aec0;
}

[data-theme="dark"] .form-group input[type="number"],
[data-theme="dark"] .form-group select {
  background-color: #2d3748;
  border-color: #4a5568;
  color: #e2e8f0;
}

[data-theme="dark"] .form-group input[type="color"] {
  background-color: #2d3748;
  border-color: #4a5568;
}

[data-theme="dark"] .apply-options-dropdown {
  background: #2d3748;
  border-color: #4a5568;
}

[data-theme="dark"] .apply-option {
  color: #a0aec0;
}

[data-theme="dark"] .preset-modal {
  background: #2d3748;
}

[data-theme="dark"] .preset-modal h4 {
  color: #e2e8f0;
}

[data-theme="dark"] .preset-input {
  background-color: #1a202c;
  border-color: #4a5568;
  color: #e2e8f0;
}

[data-theme="dark"] .stroke-options {
  border-left-color: #4a5568;
}

[data-theme="dark"] .stroke-options .input-hint {
  color: #a0aec0;
}

/* ===================================
   设置侧边栏样式 - 完整迁移自 sidebar.css
   =================================== */

#settings-sidebar {
  position: fixed;
  top: 20px;
  left: 20px;
  width: 300px;
  height: calc(100vh - 40px);
  overflow-y: auto !important;
  padding-top: 20px;
  box-sizing: border-box;
  margin-right: 0;
  order: -1;
  display: flex;
  flex-direction: column;
  scrollbar-width: thin;
  scrollbar-color: #cbd5e0 #f8fafc;
  direction: rtl;
}

#settings-sidebar > * {
  direction: ltr;
}

#settings-sidebar .settings-card {
  background-color: white;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  padding: 25px;
  margin-bottom: 15px;
  transition: box-shadow 0.2s;
}

#settings-sidebar .settings-card:hover {
  transform: none;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

#settings-sidebar .settings-card h2 {
  border-bottom: 2px solid #f0f0f0;
  padding-bottom: 12px;
  margin-bottom: 20px;
  color: #2c3e50;
  font-size: 1.6em;
  text-align: center;
}

#settings-sidebar .settings-card h3 {
  color: #3a4767;
  font-size: 1.2em;
  margin-bottom: 15px;
  display: flex;
  align-items: center;
}

#settings-sidebar .settings-card h3 .toggle-icon {
  margin-left: auto;
  color: #8492a6;
  font-size: 1em;
}

#settings-sidebar .settings-card h3 .toggle-icon:hover {
  color: #3498db;
}

#settings-sidebar .settings-card h3 .toggle-icon:before {
  content: none;
}

#settings-sidebar .settings-form > div {
  margin-bottom: 15px;
  position: relative;
}

#settings-sidebar label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

#settings-sidebar select,
#settings-sidebar input[type="number"],
#settings-sidebar input[type="text"] {
  width: 100%;
  padding: 12px;
  border: 1px solid #e0e6ed;
  border-radius: 8px;
  box-sizing: border-box;
  font-size: 1em;
  transition: border-color 0.3s, box-shadow 0.3s;
  background-color: #f9fafc;
}

#settings-sidebar select:focus,
#settings-sidebar input[type="number"]:focus,
#settings-sidebar input[type="text"]:focus {
  border-color: #3498db;
  box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
  outline: none;
}

#settings-sidebar .input-hint {
  font-size: 0.9em;
  color: #777;
  margin-top: 0.2em;
  display: block;
}

#settings-sidebar .navigation-buttons {
  display: flex;
  gap: 10px;
  margin-top: 20px;
  justify-content: space-between;
  width: auto;
}

#settings-sidebar .navigation-buttons button {
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

#settings-sidebar .navigation-buttons button:hover {
  background-color: #5a6268;
}

#settings-sidebar .navigation-buttons button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

#settings-sidebar #font-settings {
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

#settings-sidebar #font-settings:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}

#settings-sidebar #font-settings h3 {
  margin-bottom: 10px;
  margin-top: 0;
}

#settings-sidebar button#applyFontSettingsToAllButton {
  font-size: 0.9em;
  padding: 8px 15px;
  margin-top: 10px;
}

#settings-sidebar #font-settings + div {
  margin-top: 20px;
}

#settings-sidebar div.settings-card {
  margin-bottom: 15px;
  transition: box-shadow 0.2s;
}

#settings-sidebar div.settings-card .collapsible-header.collapsed {
  margin-bottom: 0;
}

#settings-sidebar div.settings-card .collapsible-content.collapsed {
  padding-top: 0;
  padding-bottom: 0;
  margin-bottom: 0;
}

#settings-sidebar #strokeEnabled {
  margin-left: 5px;
  transform: scale(1.1);
}

#settings-sidebar #strokeOptions label {
  font-size: 0.95em;
}

#solidColorOptions, #inpaintingOptions {
  margin-top: 10px;
  margin-bottom: 10px;
  overflow: visible;
}

/* ===================================
   按钮样式 - 完整迁移自 components.css
   =================================== */

.sidebar-btn {
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

.sidebar-btn:hover {
  transform: translateY(-2px);
}

.sidebar-btn:before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: left 0.7s;
}

.sidebar-btn:hover:before {
  left: 100%;
}

#settings-sidebar button.settings-button {
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

#settings-sidebar button#startHqTranslation {
  text-align: center;
  justify-content: center;
  display: flex;
  align-items: center;
}

#settings-sidebar button.settings-button:hover {
  background-color: #0056b3;
}

#settings-sidebar button.settings-button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.apply-settings-group {
  display: flex;
  align-items: stretch;
  gap: 0;
  margin-top: 10px;
  position: relative;
}

.apply-settings-group .settings-button {
  flex: 1;
  margin-top: 0 !important;
  border-radius: 5px 0 0 5px !important;
}

.settings-gear-btn {
  padding: 8px 12px;
  background-color: #0056b3;
  color: white;
  border: none;
  border-left: 1px solid rgba(255,255,255,0.3);
  border-radius: 0 5px 5px 0;
  cursor: pointer;
  font-size: 1em;
  transition: background-color 0.3s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.settings-gear-btn:hover {
  background-color: #004494;
}

.settings-gear-btn:disabled {
  background-color: #999;
  cursor: not-allowed;
}

.apply-settings-dropdown {
  display: none;
  position: absolute;
  bottom: 100%;
  left: 0;
  right: 0;
  background: var(--card-bg, #fff);
  border: 1px solid var(--border-color, #ddd);
  border-radius: 5px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  z-index: 9999;
  padding: 10px;
  margin-bottom: 5px;
}

.apply-settings-dropdown.show {
  display: block;
}

.apply-settings-dropdown .dropdown-title {
  font-size: 0.9em;
  font-weight: bold;
  color: var(--text-color, #333);
  margin-bottom: 8px;
  padding-bottom: 5px;
  border-bottom: 1px solid var(--border-color, #eee);
}

.apply-settings-dropdown label {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 0;
  font-size: 0.85em;
  color: var(--text-color, #555);
  cursor: pointer;
}

.apply-settings-dropdown label:hover {
  color: var(--primary-color, #007bff);
}

.apply-settings-dropdown input[type="checkbox"] {
  width: 14px;
  height: 14px;
  cursor: pointer;
}

.apply-settings-dropdown .select-all-option {
  font-weight: bold;
  color: var(--primary-color, #007bff);
}

.apply-settings-dropdown .dropdown-divider {
  height: 1px;
  background: var(--border-color, #eee);
  margin: 6px 0;
}

/* 翻译按钮 - 绿色 */
#settings-sidebar button#translateButton,
#settings-sidebar button#translateAllButton {
  width: 100%;
  padding: 14px 25px;
  background: linear-gradient(135deg, #4cae4c 0%, #5cb85c 100%);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1.1em;
  transition: all 0.3s;
  margin-top: 15px;
  box-shadow: 0 4px 6px rgba(92, 184, 92, 0.2);
  font-weight: 600;
  letter-spacing: 0.5px;
  position: relative;
  overflow: hidden;
}

#settings-sidebar button#translateButton:hover,
#settings-sidebar button#translateAllButton:hover {
  background: linear-gradient(135deg, #449d44 0%, #5cb85c 100%);
  box-shadow: 0 6px 10px rgba(92, 184, 92, 0.3);
  transform: translateY(-2px);
}

#settings-sidebar button#translateButton:before,
#settings-sidebar button#translateAllButton:before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: left 0.7s;
}

#settings-sidebar button#translateButton:hover:before,
#settings-sidebar button#translateAllButton:hover:before {
  left: 100%;
}

/* 消除文字按钮 - 蓝色 */
#settings-sidebar button#removeTextOnlyButton,
#settings-sidebar button#removeAllTextButton {
  width: 100%;
  padding: 14px 25px;
  background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1.1em;
  transition: all 0.3s;
  margin-top: 15px;
  box-shadow: 0 4px 6px rgba(52, 152, 219, 0.2);
  font-weight: 600;
  letter-spacing: 0.5px;
  position: relative;
  overflow: hidden;
}

#settings-sidebar button#removeTextOnlyButton:hover,
#settings-sidebar button#removeAllTextButton:hover {
  background: linear-gradient(135deg, #2980b9 0%, #1c6ea4 100%);
  box-shadow: 0 6px 10px rgba(52, 152, 219, 0.3);
  transform: translateY(-2px);
}

#settings-sidebar button#removeTextOnlyButton:before,
#settings-sidebar button#removeAllTextButton:before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: left 0.7s;
}

#settings-sidebar button#removeTextOnlyButton:hover:before,
#settings-sidebar button#removeAllTextButton:hover:before {
  left: 100%;
}

/* 删除按钮 - 红色 */
#settings-sidebar button#clearAllImagesButton,
#settings-sidebar button#deleteCurrentImageButton {
  width: 100%;
  padding: 14px 25px;
  background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1.1em;
  transition: all 0.3s;
  margin-top: 15px;
  box-shadow: 0 4px 6px rgba(220, 53, 69, 0.2);
  font-weight: 600;
  letter-spacing: 0.5px;
  position: relative;
  overflow: hidden;
}

#settings-sidebar button#clearAllImagesButton:hover,
#settings-sidebar button#deleteCurrentImageButton:hover {
  background: linear-gradient(135deg, #bd2130 0%, #c82333 100%);
  box-shadow: 0 6px 10px rgba(220, 53, 69, 0.3);
  transform: translateY(-2px);
}

#settings-sidebar button#clearAllImagesButton:before,
#settings-sidebar button#deleteCurrentImageButton:before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: left 0.7s;
}

#settings-sidebar button#clearAllImagesButton:hover:before,
#settings-sidebar button#deleteCurrentImageButton:hover:before {
  left: 100%;
}

/* 校对按钮和高质量翻译按钮 - 紫色 */
#proofreadButton,
#startHqTranslationBtn {
  width: 100%;
  padding: 14px 25px;
  background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1.1em;
  transition: all 0.3s;
  margin-top: 15px;
  box-shadow: 0 4px 6px rgba(142, 68, 173, 0.2);
  font-weight: 600;
  letter-spacing: 0.5px;
  position: relative;
  overflow: hidden;
}

#proofreadButton:hover,
#startHqTranslationBtn:hover {
  background: linear-gradient(135deg, #8e44ad 0%, #6c3483 100%);
  box-shadow: 0 6px 10px rgba(142, 68, 173, 0.3);
  transform: translateY(-2px);
}

#proofreadButton:before,
#startHqTranslationBtn:before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: left 0.7s;
}

#proofreadButton:hover:before,
#startHqTranslationBtn:hover:before {
  left: 100%;
}

/* 按钮颜色变体类 */
#settings-sidebar button.red-button {
  background: linear-gradient(135deg, #d43f3a 0%, #d9534f 100%);
  box-shadow: 0 4px 6px rgba(217, 83, 79, 0.2);
}

#settings-sidebar button.red-button:hover {
  background: linear-gradient(135deg, #c9302c 0%, #d9534f 100%);
  box-shadow: 0 6px 10px rgba(217, 83, 79, 0.3);
}

#settings-sidebar button.orange-button {
  background: linear-gradient(135deg, #eb9316 0%, #f0ad4e 100%);
  box-shadow: 0 4px 6px rgba(240, 173, 78, 0.2);
}

#settings-sidebar button.orange-button:hover {
  background: linear-gradient(135deg, #e67e22 0%, #d35400 100%);
  box-shadow: 0 6px 10px rgba(211, 84, 0, 0.3);
}

#settings-sidebar button.green-button {
  background: linear-gradient(135deg, #5cb85c 0%, #4cae4c 100%);
  box-shadow: 0 4px 6px rgba(92, 184, 92, 0.2);
}

#settings-sidebar button.green-button:hover {
  background: linear-gradient(135deg, #4cae4c 0%, #449d44 100%);
  box-shadow: 0 6px 10px rgba(92, 184, 92, 0.3);
}

#retranslateFailedButton {
  padding: 8px 15px;
  background-color: #f39c12;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  transition: background-color 0.3s;
}

#retranslateFailedButton:hover {
  background-color: #e67e22;
}

#retranslateFailedButton:disabled {
  background-color: #f9d6a5;
  cursor: not-allowed;
}
</style>
