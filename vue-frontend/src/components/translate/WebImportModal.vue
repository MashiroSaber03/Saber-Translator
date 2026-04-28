<script setup lang="ts">
/**
 * 网页导入模态框
 * 核心功能界面：URL输入 → 提取 → 预览 → 下载 → 导入
 * 支持双引擎：Gallery-DL (主流站点高速下载) 和 AI Agent (通用网站)
 */
import { ref, computed, watch } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import { useWebImportStore } from '@/stores/webImportStore'
import { useImageStore } from '@/stores/imageStore'
import { extractImages, downloadImages, checkGalleryDLSupport, getGalleryDLImages, testFirecrawlConnection, testAgentConnection } from '@/api/webImport'
import type { AgentLog, ExtractResult, WebImportEngine } from '@/types/webImport'
import { WEB_IMPORT_AGENT_PROVIDERS } from '@/constants'
import { normalizeProviderId } from '@/config/aiProviders'

const webImportStore = useWebImportStore()
const imageStore = useImageStore()

// 本地状态
const urlInput = ref('')
const logsExpanded = ref(true)
const selectedEngine = ref<WebImportEngine>('auto')
const galleryDLAvailable = ref(false)
const galleryDLSupported = ref(false)
const checkingSupport = ref(false)

// 设置相关状态
const settingsExpanded = ref(false)
const activeSettingsTab = ref<'basic' | 'preprocess' | 'advanced'>('basic')
const testingFirecrawl = ref(false)
const testingAgent = ref(false)
const showFirecrawlKey = ref(false)
const showAgentKey = ref(false)

// 计算属性
const isVisible = computed(() => webImportStore.modalVisible)
const status = computed(() => webImportStore.status)
const logs = computed(() => webImportStore.logs)
const extractResult = computed(() => webImportStore.extractResult)
const selectedPages = computed(() => webImportStore.selectedPages)
const selectedCount = computed(() => webImportStore.selectedCount)
const downloadProgress = computed(() => webImportStore.downloadProgress)
const downloadProgressPercent = computed(() => webImportStore.downloadProgressPercent)
const error = computed(() => webImportStore.error)
const isProcessing = computed(() => webImportStore.isProcessing)
const showAgentLogs = computed(() => webImportStore.settings.ui.showAgentLogs)

// 当前使用的引擎
const currentEngine = computed(() => extractResult.value?.engine || null)

// 引擎显示名称
const engineDisplayName = computed(() => {
  switch (currentEngine.value) {
    case 'gallery-dl': return 'Gallery-DL'
    case 'ai-agent': return 'AI Agent'
    default: return ''
  }
})

// 是否全选
const isAllSelected = computed(() => {
  if (!extractResult.value?.pages) return false
  return selectedCount.value === extractResult.value.pages.length
})

// 获取预览图 URL（gallery-dl 引擎直接使用静态文件服务）
function getPreviewUrl(originalUrl: string): string {
  // gallery-dl 引擎的图片已在本地，直接使用静态服务路径
  if (currentEngine.value === 'gallery-dl') {
    // imageUrl 格式: /api/web-import/static/temp/gallery_dl/xxx.webp
    // 直接返回，不需要代理
    return originalUrl
  }
  return originalUrl
}

// 检查 URL 支持（防抖）
let checkSupportTimeout: ReturnType<typeof setTimeout> | null = null
async function checkUrlSupport(url: string) {
  if (checkSupportTimeout) {
    clearTimeout(checkSupportTimeout)
  }
  
  if (!url.trim()) {
    galleryDLAvailable.value = false
    galleryDLSupported.value = false
    return
  }
  
  checkSupportTimeout = setTimeout(async () => {
    checkingSupport.value = true
    try {
      const result = await checkGalleryDLSupport(url)
      galleryDLAvailable.value = result.available
      galleryDLSupported.value = result.supported
    } catch {
      galleryDLAvailable.value = false
      galleryDLSupported.value = false
    } finally {
      checkingSupport.value = false
    }
  }, 500)
}

// 关闭模态框
function handleClose() {
  if (isProcessing.value) {
    if (!confirm('正在处理中，确定要关闭吗？')) return
  }
  webImportStore.closeModal()
  webImportStore.resetState()
  urlInput.value = ''
}

// 开始提取
async function handleExtract() {
  const url = urlInput.value.trim()
  if (!url) {
    alert('请输入网址')
    return
  }

  // 验证 URL
  try {
    new URL(url)
  } catch {
    alert('请输入有效的网址')
    return
  }

  // 重置状态
  webImportStore.resetState()
  webImportStore.setUrl(url)
  webImportStore.setStatus('extracting')

  try {
    await extractImages(
      url,
      webImportStore.settings,
      (log: AgentLog) => {
        webImportStore.addLog(log)
      },
      (result: ExtractResult) => {
        webImportStore.setExtractResult(result)
        if (result.success) {
          webImportStore.setStatus('extracted')
        } else {
          webImportStore.setError(result.error || '提取失败')
        }
      },
      (errorMsg: string) => {
        webImportStore.setError(errorMsg)
      },
      selectedEngine.value,
      // 新增：每收到一张图片就增量添加
      (page) => {
        webImportStore.addPageIncremental(page)
      }
    )
  } catch (e) {
    webImportStore.setError(e instanceof Error ? e.message : '提取失败')
  }
}

// 切换页面选择
function togglePage(pageNumber: number) {
  webImportStore.togglePageSelection(pageNumber)
}

// 全选/取消全选
function toggleAll() {
  webImportStore.toggleSelectAll()
}

// 开始下载并导入
async function handleImport() {
  if (!extractResult.value?.pages || selectedCount.value === 0) {
    alert('请选择要导入的图片')
    return
  }

  // 获取选中的页面
  const selectedPagesList = extractResult.value.pages.filter((p) =>
    selectedPages.value.has(p.pageNumber)
  )

  webImportStore.setStatus('downloading')
  webImportStore.updateDownloadProgress(0, selectedPagesList.length)

  // 使用提取时使用的引擎
  const engineToUse = currentEngine.value || 'ai-agent'

  try {
    // gallery-dl 引擎：图片已下载到临时目录，直接获取
    if (engineToUse === 'gallery-dl') {
      const galleryResult = await getGalleryDLImages()
      
      if (galleryResult.success && galleryResult.images.length > 0) {
        let importedCount = 0
        const maxImport = Math.min(galleryResult.images.length, selectedPagesList.length)
        
        for (let i = 0; i < maxImport; i++) {
          const img = galleryResult.images[i]
          if (img && img.filename && img.data) {
            imageStore.addImage(img.filename, img.data)
            importedCount++
            webImportStore.updateDownloadProgress(importedCount, maxImport)
          }
        }
        
        webImportStore.setStatus('completed')
        alert(`成功导入 ${importedCount} 张图片`)
        handleClose()
        return
      } else {
        throw new Error(galleryResult.error || '获取图片失败')
      }
    }
    
    // AI Agent 引擎：调用下载接口
    const result = await downloadImages(
      selectedPagesList,
      extractResult.value.sourceUrl,
      webImportStore.settings,
      engineToUse
    )

    if (result.success && result.images.length > 0) {
      webImportStore.setDownloadedImages(result.images)
      webImportStore.updateDownloadProgress(result.images.length, selectedPagesList.length)

      // 导入到 imageStore (参数顺序: fileName, dataUrl)
      for (const img of result.images) {
        imageStore.addImage(img.filename, img.dataUrl)
      }

      webImportStore.setStatus('completed')

      // 提示成功
      const failedMsg = result.failedCount > 0 ? `，${result.failedCount} 张失败` : ''
      alert(`成功导入 ${result.images.length} 张图片${failedMsg}`)

      // 关闭模态框
      handleClose()
    } else {
      webImportStore.setError(result.error || '下载失败')
    }
  } catch (e) {
    webImportStore.setError(e instanceof Error ? e.message : '下载失败')
  }
}

// 监听模态框打开时聚焦输入框
watch(isVisible, (visible) => {
  if (visible) {
    setTimeout(() => {
      const input = document.querySelector('.url-input') as HTMLInputElement
      input?.focus()
    }, 100)
  }
})

// 监听 URL 输入变化，检查 gallery-dl 支持
watch(urlInput, (newUrl) => {
  checkUrlSupport(newUrl)
})

// 计算属性：是否显示自定义 URL
const showCustomUrl = computed(() => normalizeProviderId(webImportStore.settings.agent.provider) === 'custom')

// 测试 Firecrawl 连接
async function handleTestFirecrawl() {
  if (!webImportStore.settings.firecrawl.apiKey) {
    alert('请输入 Firecrawl API Key')
    return
  }

  testingFirecrawl.value = true
  try {
    const result = await testFirecrawlConnection(webImportStore.settings.firecrawl.apiKey)
    if (result.success) {
      alert('✅ Firecrawl 连接成功')
    } else {
      alert(`❌ 连接失败: ${result.error}`)
    }
  } catch (e) {
    alert(`❌ 连接失败: ${e instanceof Error ? e.message : '未知错误'}`)
  } finally {
    testingFirecrawl.value = false
  }
}

// 测试 Agent 连接
async function handleTestAgent() {
  if (!webImportStore.settings.agent.apiKey) {
    alert('请输入 AI Agent API Key')
    return
  }

  testingAgent.value = true
  try {
    const result = await testAgentConnection(
      webImportStore.settings.agent.provider,
      webImportStore.settings.agent.apiKey,
      webImportStore.settings.agent.customBaseUrl,
      webImportStore.settings.agent.modelName
    )
    if (result.success) {
      alert('✅ AI Agent 连接成功')
    } else {
      alert(`❌ 连接失败: ${result.error}`)
    }
  } catch (e) {
    alert(`❌ 连接失败: ${e instanceof Error ? e.message : '未知错误'}`)
  } finally {
    testingAgent.value = false
  }
}

// 重置提示词
function handleResetPrompt() {
  if (confirm('确定要重置为默认提示词吗？')) {
    webImportStore.resetExtractionPrompt()
  }
}
</script>

<template>
  <BaseModal
    :model-value="isVisible"
    title="🌐 从网页导入漫画"
    size="large"
    custom-class="web-import-modal"
    :close-on-overlay="!isProcessing"
    :close-on-esc="!isProcessing"
    @close="handleClose"
  >
          <!-- URL 输入 -->
          <div class="url-section">
            <input
              v-model="urlInput"
              type="url"
              class="url-input"
              placeholder="输入漫画网页 URL，如 https://example.com/chapter-1"
              :disabled="isProcessing"
              @keyup.enter="handleExtract"
            />
            <select
              v-model="selectedEngine"
              class="engine-select"
              :disabled="isProcessing"
            >
              <option value="auto">自动选择</option>
              <option value="gallery-dl">Gallery-DL</option>
              <option value="ai-agent">AI Agent</option>
            </select>
            <button
              class="extract-btn"
              :disabled="isProcessing || !urlInput.trim()"
              @click="handleExtract"
            >
              <span v-if="status === 'extracting'" class="loading-spinner"></span>
              <span v-else>🔍</span>
              {{ status === 'extracting' ? '提取中...' : '开始提取' }}
            </button>
          </div>

          <!-- 引擎支持提示 -->
          <div v-if="urlInput.trim() && !isProcessing" class="engine-hint">
            <span v-if="checkingSupport" class="hint-checking">检查中...</span>
            <span v-else-if="galleryDLSupported" class="hint-supported">✓ 该网站支持 Gallery-DL 高速下载</span>
            <span v-else-if="galleryDLAvailable" class="hint-unsupported">该网站将使用 AI Agent 模式</span>
          </div>

          <!-- 使用须知 -->
          <div class="notice">
            ⚠️ 请仅爬取您有权访问的内容，并遵守目标网站的使用条款。
          </div>

          <!-- 设置区域（可折叠） -->
          <div class="settings-section">
            <div class="settings-header" @click="settingsExpanded = !settingsExpanded">
              <span class="settings-toggle">{{ settingsExpanded ? '▼' : '▶' }}</span>
              <span class="settings-title">⚙️ 设置</span>
              <span class="settings-hint">点击展开配置</span>
            </div>
            
            <div v-if="settingsExpanded" class="settings-content">
              <!-- 选项卡 -->
              <div class="settings-tabs">
                <button
                  class="settings-tab"
                  :class="{ active: activeSettingsTab === 'basic' }"
                  @click="activeSettingsTab = 'basic'"
                >
                  基本设置
                </button>
                <button
                  class="settings-tab"
                  :class="{ active: activeSettingsTab === 'preprocess' }"
                  @click="activeSettingsTab = 'preprocess'"
                >
                  图片预处理
                </button>
                <button
                  class="settings-tab"
                  :class="{ active: activeSettingsTab === 'advanced' }"
                  @click="activeSettingsTab = 'advanced'"
                >
                  高级设置
                </button>
              </div>

              <!-- 基本设置 -->
              <div v-show="activeSettingsTab === 'basic'" class="settings-tab-content">
                <!-- Firecrawl 配置 -->
                <div class="settings-group">
                  <h4 class="group-title">Firecrawl 配置</h4>
                  <div class="form-row">
                    <label class="form-label">API Key</label>
                    <div class="input-group">
                      <input
                        :type="showFirecrawlKey ? 'text' : 'password'"
                        class="form-input"
                        :value="webImportStore.settings.firecrawl.apiKey"
                        @input="webImportStore.setFirecrawlApiKey(($event.target as HTMLInputElement).value)"
                        placeholder="fc-xxxxxxxxxxxxxxxx"
                      />
                      <button class="toggle-btn" @click="showFirecrawlKey = !showFirecrawlKey">
                        {{ showFirecrawlKey ? '👁' : '👁‍🗨' }}
                      </button>
                      <button
                        class="test-btn"
                        :disabled="testingFirecrawl || !webImportStore.settings.firecrawl.apiKey"
                        @click="handleTestFirecrawl"
                      >
                        {{ testingFirecrawl ? '测试中...' : '测试连接' }}
                      </button>
                    </div>
                  </div>
                </div>

                <!-- AI Agent 配置 -->
                <div class="settings-group">
                  <h4 class="group-title">AI Agent 配置</h4>
                  
                  <div class="form-row">
                    <label class="form-label">服务商</label>
                    <select
                      class="form-select"
                      :value="webImportStore.settings.agent.provider"
                      @change="webImportStore.setAgentProvider(($event.target as HTMLSelectElement).value)"
                    >
                      <option
                        v-for="provider in WEB_IMPORT_AGENT_PROVIDERS"
                        :key="provider.value"
                        :value="provider.value"
                      >
                        {{ provider.label }}
                      </option>
                    </select>
                  </div>

                  <div class="form-row">
                    <label class="form-label">API Key</label>
                    <div class="input-group">
                      <input
                        :type="showAgentKey ? 'text' : 'password'"
                        class="form-input"
                        :value="webImportStore.settings.agent.apiKey"
                        @input="webImportStore.setAgentApiKey(($event.target as HTMLInputElement).value)"
                        placeholder="sk-xxxxxxxxxxxxxxxx"
                      />
                      <button class="toggle-btn" @click="showAgentKey = !showAgentKey">
                        {{ showAgentKey ? '👁' : '👁‍🗨' }}
                      </button>
                    </div>
                  </div>

                  <div v-if="showCustomUrl" class="form-row">
                    <label class="form-label">自定义 API 地址</label>
                    <input
                      type="url"
                      class="form-input"
                      :value="webImportStore.settings.agent.customBaseUrl"
                      @input="webImportStore.setAgentBaseUrl(($event.target as HTMLInputElement).value)"
                      placeholder="https://api.example.com/v1"
                    />
                  </div>

                  <div class="form-row">
                    <label class="form-label">模型名称</label>
                    <input
                      type="text"
                      class="form-input"
                      :value="webImportStore.settings.agent.modelName"
                      @input="webImportStore.setAgentModelName(($event.target as HTMLInputElement).value)"
                      placeholder="gpt-4o-mini"
                    />
                  </div>

                  <div class="form-row inline">
                    <label class="checkbox-label">
                      <input
                        type="checkbox"
                        :checked="webImportStore.settings.agent.forceJsonOutput"
                        @change="webImportStore.setAgentForceJson(($event.target as HTMLInputElement).checked)"
                      />
                      强制 JSON 格式
                    </label>
                    <label class="checkbox-label">
                      <input
                        type="checkbox"
                        :checked="webImportStore.settings.agent.useStream"
                        @change="webImportStore.setAgentUseStream(($event.target as HTMLInputElement).checked)"
                      />
                      流式调用
                    </label>
                  </div>

                  <div class="form-row">
                    <button
                      class="test-btn full"
                      :disabled="testingAgent || !webImportStore.settings.agent.apiKey"
                      @click="handleTestAgent"
                    >
                      {{ testingAgent ? '测试中...' : '测试 Agent 连接' }}
                    </button>
                  </div>
                </div>

                <!-- 提取设置 -->
                <div class="settings-group">
                  <h4 class="group-title">
                    提取设置
                    <button class="reset-btn" @click="handleResetPrompt">重置为默认</button>
                  </h4>

                  <div class="form-row">
                    <label class="form-label">提取提示词</label>
                    <textarea
                      class="form-textarea"
                      :value="webImportStore.settings.extraction.prompt"
                      @input="webImportStore.setExtractionPrompt(($event.target as HTMLTextAreaElement).value)"
                      rows="6"
                      placeholder="输入提取提示词..."
                    ></textarea>
                  </div>

                  <div class="form-row">
                    <label class="form-label">最大迭代次数</label>
                    <input
                      type="number"
                      class="form-input small"
                      :value="webImportStore.settings.extraction.maxIterations"
                      @input="webImportStore.setExtractionMaxIterations(Number(($event.target as HTMLInputElement).value))"
                      min="1"
                      max="20"
                    />
                  </div>
                </div>

                <!-- 下载设置 -->
                <div class="settings-group">
                  <h4 class="group-title">下载设置</h4>

                  <div class="form-grid">
                    <div class="form-row">
                      <label class="form-label">并发数</label>
                      <input
                        type="number"
                        class="form-input small"
                        :value="webImportStore.settings.download.concurrency"
                        @input="webImportStore.setDownloadConcurrency(Number(($event.target as HTMLInputElement).value))"
                        min="1"
                        max="10"
                      />
                    </div>

                    <div class="form-row">
                      <label class="form-label">超时 (秒)</label>
                      <input
                        type="number"
                        class="form-input small"
                        :value="webImportStore.settings.download.timeout"
                        @input="webImportStore.setDownloadTimeout(Number(($event.target as HTMLInputElement).value))"
                        min="5"
                        max="120"
                      />
                    </div>

                    <div class="form-row">
                      <label class="form-label">重试次数</label>
                      <input
                        type="number"
                        class="form-input small"
                        :value="webImportStore.settings.download.retries"
                        @input="webImportStore.setDownloadRetries(Number(($event.target as HTMLInputElement).value))"
                        min="0"
                        max="5"
                      />
                    </div>

                    <div class="form-row">
                      <label class="form-label">下载间隔 (ms)</label>
                      <input
                        type="number"
                        class="form-input small"
                        :value="webImportStore.settings.download.delay"
                        @input="webImportStore.setDownloadDelay(Number(($event.target as HTMLInputElement).value))"
                        min="0"
                        max="2000"
                        step="100"
                      />
                    </div>
                  </div>

                  <div class="form-row">
                    <label class="checkbox-label">
                      <input
                        type="checkbox"
                        :checked="webImportStore.settings.download.useReferer"
                        @change="webImportStore.setDownloadUseReferer(($event.target as HTMLInputElement).checked)"
                      />
                      自动添加 Referer
                    </label>
                  </div>
                </div>

                <!-- 界面设置 -->
                <div class="settings-group">
                  <h4 class="group-title">界面设置</h4>
                  <div class="form-row inline">
                    <label class="checkbox-label">
                      <input
                        type="checkbox"
                        :checked="webImportStore.settings.ui.showAgentLogs"
                        @change="webImportStore.setShowAgentLogs(($event.target as HTMLInputElement).checked)"
                      />
                      显示 AI 工作日志
                    </label>
                    <label class="checkbox-label">
                      <input
                        type="checkbox"
                        :checked="webImportStore.settings.ui.autoImport"
                        @change="webImportStore.setAutoImport(($event.target as HTMLInputElement).checked)"
                      />
                      提取后自动导入
                    </label>
                  </div>
                </div>
              </div>

              <!-- 图片预处理 -->
              <div v-show="activeSettingsTab === 'preprocess'" class="settings-tab-content">
                <div class="settings-group">
                  <div class="form-row">
                    <label class="checkbox-label">
                      <input
                        type="checkbox"
                        :checked="webImportStore.settings.imagePreprocess.enabled"
                        @change="webImportStore.setImagePreprocessEnabled(($event.target as HTMLInputElement).checked)"
                      />
                      启用图片预处理
                    </label>
                  </div>

                  <template v-if="webImportStore.settings.imagePreprocess.enabled">
                    <div class="form-row">
                      <label class="checkbox-label">
                        <input
                          type="checkbox"
                          :checked="webImportStore.settings.imagePreprocess.autoRotate"
                          @change="webImportStore.setImageAutoRotate(($event.target as HTMLInputElement).checked)"
                        />
                        根据 EXIF 自动旋转
                      </label>
                    </div>

                    <h5 class="subsection-title">压缩设置</h5>
                    <div class="form-row">
                      <label class="checkbox-label">
                        <input
                          type="checkbox"
                          :checked="webImportStore.settings.imagePreprocess.compression.enabled"
                          @change="webImportStore.setImageCompressionEnabled(($event.target as HTMLInputElement).checked)"
                        />
                        启用压缩
                      </label>
                    </div>

                    <template v-if="webImportStore.settings.imagePreprocess.compression.enabled">
                      <div class="form-grid">
                        <div class="form-row">
                          <label class="form-label">质量 (0-100)</label>
                          <input
                            type="number"
                            class="form-input small"
                            :value="webImportStore.settings.imagePreprocess.compression.quality"
                            @input="webImportStore.setImageCompressionQuality(Number(($event.target as HTMLInputElement).value))"
                            min="1"
                            max="100"
                          />
                        </div>
                        <div class="form-row">
                          <label class="form-label">最大宽度 (0=不限)</label>
                          <input
                            type="number"
                            class="form-input small"
                            :value="webImportStore.settings.imagePreprocess.compression.maxWidth"
                            @input="webImportStore.setImageMaxWidth(Number(($event.target as HTMLInputElement).value))"
                            min="0"
                          />
                        </div>
                        <div class="form-row">
                          <label class="form-label">最大高度 (0=不限)</label>
                          <input
                            type="number"
                            class="form-input small"
                            :value="webImportStore.settings.imagePreprocess.compression.maxHeight"
                            @input="webImportStore.setImageMaxHeight(Number(($event.target as HTMLInputElement).value))"
                            min="0"
                          />
                        </div>
                      </div>
                    </template>

                    <h5 class="subsection-title">格式转换</h5>
                    <div class="form-row">
                      <label class="checkbox-label">
                        <input
                          type="checkbox"
                          :checked="webImportStore.settings.imagePreprocess.formatConvert.enabled"
                          @change="webImportStore.setImageFormatConvertEnabled(($event.target as HTMLInputElement).checked)"
                        />
                        启用格式转换
                      </label>
                    </div>

                    <div v-if="webImportStore.settings.imagePreprocess.formatConvert.enabled" class="form-row">
                      <label class="form-label">目标格式</label>
                      <select
                        class="form-select"
                        :value="webImportStore.settings.imagePreprocess.formatConvert.targetFormat"
                        @change="webImportStore.setImageTargetFormat(($event.target as HTMLSelectElement).value as 'jpeg' | 'png' | 'webp' | 'original')"
                      >
                        <option value="original">保持原格式</option>
                        <option value="jpeg">JPEG</option>
                        <option value="png">PNG</option>
                        <option value="webp">WebP</option>
                      </select>
                    </div>
                  </template>
                </div>
              </div>

              <!-- 高级设置 -->
              <div v-show="activeSettingsTab === 'advanced'" class="settings-tab-content">
                <div class="settings-group">
                  <h4 class="group-title">自定义请求头</h4>

                  <div class="form-row">
                    <label class="form-label">Cookie</label>
                    <input
                      type="text"
                      class="form-input"
                      :value="webImportStore.settings.advanced.customCookie"
                      @input="webImportStore.setCustomCookie(($event.target as HTMLInputElement).value)"
                      placeholder="name=value; name2=value2"
                    />
                  </div>

                  <div class="form-row">
                    <label class="form-label">Headers (JSON)</label>
                    <textarea
                      class="form-textarea"
                      :value="webImportStore.settings.advanced.customHeaders"
                      @input="webImportStore.setCustomHeaders(($event.target as HTMLTextAreaElement).value)"
                      rows="3"
                      placeholder='{"X-Custom-Header": "value"}'
                    ></textarea>
                  </div>

                  <div class="form-row">
                    <label class="checkbox-label">
                      <input
                        type="checkbox"
                        :checked="webImportStore.settings.advanced.bypassProxy"
                        @change="webImportStore.setBypassProxy(($event.target as HTMLInputElement).checked)"
                      />
                      绕过系统代理 (连接本地服务时使用)
                    </label>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- AI 工作日志 -->
          <div v-if="showAgentLogs && logs.length > 0" class="logs-section">
            <div class="logs-header" @click="logsExpanded = !logsExpanded">
              <span class="logs-toggle">{{ logsExpanded ? '▼' : '▶' }}</span>
              <span>AI 工作日志</span>
              <span v-if="status === 'extracting'" class="extracting-hint">(提取中...)</span>
            </div>
            <div v-if="logsExpanded" class="logs-content">
              <div
                v-for="(log, index) in logs"
                :key="index"
                class="log-item"
                :class="`log-${log.type}`"
              >
                <span class="log-time">[{{ log.timestamp }}]</span>
                <span class="log-message">{{ log.message }}</span>
              </div>
            </div>
          </div>

          <!-- 错误提示 -->
          <div v-if="error" class="error-section">
            <span class="error-icon">❌</span>
            <span class="error-message">{{ error }}</span>
          </div>

          <!-- 提取结果 -->
          <div v-if="extractResult?.success" class="result-section">
            <div class="result-header">
              <span class="result-title">
                📖 《{{ extractResult.comicTitle }}》- {{ extractResult.chapterTitle }}
              </span>
              <span class="result-meta">
                <span class="result-count">共 {{ extractResult.totalPages }} 张</span>
                <span v-if="engineDisplayName" class="result-engine">| 引擎: {{ engineDisplayName }}</span>
              </span>
            </div>

            <!-- 选择控制 -->
            <div class="select-control">
              <label class="select-all">
                <input
                  type="checkbox"
                  :checked="isAllSelected"
                  @change="toggleAll"
                />
                全选
              </label>
              <span class="selected-count">已选: {{ selectedCount }} 张</span>
            </div>

            <!-- 图片网格 -->
            <div class="image-grid">
              <div
                v-for="page in extractResult.pages"
                :key="page.pageNumber"
                class="image-item"
                :class="{ selected: selectedPages.has(page.pageNumber) }"
                @click="togglePage(page.pageNumber)"
              >
                <div class="image-checkbox">
                  <input
                    type="checkbox"
                    :checked="selectedPages.has(page.pageNumber)"
                    @click.stop
                    @change="togglePage(page.pageNumber)"
                  />
                </div>
                <div class="image-preview">
                  <img :src="getPreviewUrl(page.imageUrl)" :alt="`第${page.pageNumber}页`" loading="lazy" />
                </div>
                <div class="image-label">第 {{ page.pageNumber }} 页</div>
              </div>
            </div>
          </div>

          <!-- 下载进度 -->
          <div v-if="status === 'downloading'" class="progress-section">
            <div class="progress-label">
              下载进度: {{ downloadProgress.current }}/{{ downloadProgress.total }}
            </div>
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: `${downloadProgressPercent}%` }"></div>
            </div>
          </div>

        <!-- 底部 -->
        <template #footer>
          <button class="cancel-btn" @click="handleClose" :disabled="status === 'downloading'">
            取消
          </button>
          <button
            class="import-btn"
            :disabled="!extractResult?.success || selectedCount === 0 || isProcessing"
            @click="handleImport"
          >
            <span v-if="status === 'downloading'" class="loading-spinner"></span>
            <span v-else>📥</span>
            {{ status === 'downloading' ? '下载中...' : '导入' }}
          </button>
        </template>
  </BaseModal>
</template>

<style>
/* 不使用 scoped，因为 BaseModal 使用 Teleport 将内容传送到 body */

/* WebImportModal 的 BaseModal 定制 */
.web-import-modal .modal-container {
  max-width: 800px;
  box-shadow: 0 20px 60px rgb(0, 0, 0, 0.3);
}


.url-section {
  display: flex;
  gap: 12px;
  margin-bottom: 12px;
}

.url-input {
  flex: 1;
  padding: 10px 14px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 8px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
}

.url-input:focus {
  border-color: var(--color-primary, #4a90d9);
}

.engine-select {
  padding: 10px 12px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 8px;
  font-size: 14px;
  outline: none;
  background: var(--bg-primary, #fff);
  cursor: pointer;
  min-width: 120px;
}

.engine-select:focus {
  border-color: var(--color-primary, #4a90d9);
}

.engine-select:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.engine-hint {
  font-size: 12px;
  margin-bottom: 12px;
  padding: 0 2px;
}

.hint-checking {
  color: var(--text-secondary, #888);
}

.hint-supported {
  color: #28a745;
}

.hint-unsupported {
  color: var(--text-secondary, #888);
}

.extract-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 18px;
  background: var(--btn-primary-bg, #4a90d9);
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  white-space: nowrap;
  transition: background 0.2s;
}

.extract-btn:hover:not(:disabled) {
  background: var(--btn-primary-hover-bg, #3a7fc8);
}

.extract-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.notice {
  padding: 10px 14px;
  background: #fff8e6;
  border: 1px solid #ffe0a0;
  border-radius: 6px;
  font-size: 13px;
  color: #856404;
  margin-bottom: 16px;
}

/* 设置区域样式 */
.settings-section {
  margin-bottom: 16px;
  border: 1px solid var(--border-color, #eee);
  border-radius: 8px;
  overflow: hidden;
}

.settings-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 14px;
  background: var(--bg-secondary, #f9f9f9);
  cursor: pointer;
  user-select: none;
  transition: background 0.2s;
}

.settings-header:hover {
  background: var(--bg-hover, #efefef);
}

.settings-toggle {
  font-size: 10px;
  color: var(--text-secondary, #888);
}

.settings-title {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary, #333);
}

.settings-hint {
  margin-left: auto;
  font-size: 12px;
  color: var(--text-secondary, #999);
}

.settings-content {
  padding: 16px;
  background: var(--bg-primary, #fff);
}

.settings-tabs {
  display: flex;
  gap: 4px;
  margin-bottom: 16px;
  border-bottom: 1px solid var(--border-color, #eee);
  padding-bottom: 8px;
}

.settings-tab {
  padding: 8px 16px;
  background: transparent;
  border: none;
  border-radius: 6px 6px 0 0;
  cursor: pointer;
  font-size: 13px;
  color: var(--text-secondary, #666);
  transition: all 0.2s;
}

.settings-tab:hover {
  background: var(--bg-secondary, #f5f5f5);
}

.settings-tab.active {
  background: var(--bg-secondary, #f5f5f5);
  color: var(--text-primary, #333);
  font-weight: 500;
}

.settings-tab-content {
  max-height: 400px;
  overflow-y: auto;
}

.settings-group {
  margin-bottom: 16px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border-color, #eee);
}

.settings-group:last-child {
  border-bottom: none;
  margin-bottom: 0;
}

.group-title {
  margin: 0 0 12px;
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary, #333);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.subsection-title {
  margin: 12px 0 8px;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-secondary, #666);
}

.form-row {
  margin-bottom: 12px;
}

.form-row.inline {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 12px;
}

.form-label {
  display: block;
  margin-bottom: 4px;
  font-size: 13px;
  color: var(--text-secondary, #666);
}

.form-input,
.form-select,
.form-textarea {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 6px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
  background: var(--bg-primary, #fff);
  color: var(--text-primary, #333);
}

.form-input:focus,
.form-select:focus,
.form-textarea:focus {
  border-color: var(--color-primary, #4a90d9);
}

.form-input.small {
  width: 100px;
}

.form-textarea {
  resize: vertical;
  min-height: 80px;
}

.input-group {
  display: flex;
  gap: 8px;
}

.input-group .form-input {
  flex: 1;
}

.toggle-btn {
  padding: 8px 12px;
  background: var(--bg-secondary, #f5f5f5);
  border: 1px solid var(--border-color, #ddd);
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.2s;
}

.toggle-btn:hover {
  background: var(--bg-hover, #efefef);
}

.test-btn {
  padding: 8px 14px;
  background: var(--btn-secondary-bg, #f0f0f0);
  border: 1px solid var(--border-color, #ddd);
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  white-space: nowrap;
  transition: all 0.2s;
}

.test-btn:hover:not(:disabled) {
  background: var(--btn-secondary-hover-bg, #e5e5e5);
}

.test-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.test-btn.full {
  width: 100%;
}

.reset-btn {
  padding: 4px 10px;
  background: transparent;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  color: var(--text-secondary, #666);
  transition: background 0.2s;
}

.reset-btn:hover {
  background: var(--bg-secondary, #f5f5f5);
}

.checkbox-label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  cursor: pointer;
  color: var(--text-primary, #333);
}

.checkbox-label input[type='checkbox'] {
  width: 16px;
  height: 16px;
  cursor: pointer;
}

.logs-section {
  margin-bottom: 16px;
  border: 1px solid var(--border-color, #eee);
  border-radius: 8px;
  overflow: hidden;
}

.logs-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  background: var(--bg-secondary, #f9f9f9);
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  user-select: none;
}

.logs-toggle {
  font-size: 10px;
  color: var(--text-secondary, #888);
}

.extracting-hint {
  color: var(--color-primary, #4a90d9);
  font-weight: normal;
  font-size: 13px;
}

.logs-content {
  max-height: 200px;
  overflow-y: auto;
  padding: 12px;
  background: #1e1e1e;
  font-family: Consolas, Monaco, monospace;
  font-size: 12px;
}

.log-item {
  padding: 2px 0;
  color: #ccc;
}

.log-time {
  color: #888;
  margin-right: 8px;
}

.log-info .log-message { color: #9cdcfe; }
.log-tool_call .log-message { color: #dcdcaa; }
.log-tool_result .log-message { color: #6a9955; }
.log-thinking .log-message { color: #ce9178; }
.log-error .log-message { color: #f14c4c; }

.error-section {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 14px;
  background: #fff5f5;
  border: 1px solid #ffc0c0;
  border-radius: 6px;
  margin-bottom: 16px;
  color: #c00;
}

.result-section {
  margin-bottom: 16px;
}

.result-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.result-title {
  font-size: 15px;
  font-weight: 500;
  color: var(--text-primary, #333);
}

.result-meta {
  display: flex;
  align-items: center;
  gap: 8px;
}

.result-count {
  font-size: 13px;
  color: var(--text-secondary, #666);
}

.result-engine {
  font-size: 12px;
  color: var(--text-secondary, #888);
}

.select-control {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 12px;
}

.select-all {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  font-size: 14px;
}

.selected-count {
  font-size: 13px;
  color: var(--text-secondary, #666);
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: 12px;
  max-height: 300px;
  overflow-y: auto;
  padding: 4px;
}

.image-item {
  position: relative;
  border: 2px solid var(--border-color, #eee);
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.2s;
}

.image-item:hover {
  border-color: var(--color-primary, #4a90d9);
}

.image-item.selected {
  border-color: var(--color-primary, #4a90d9);
  box-shadow: 0 0 0 2px rgb(74, 144, 217, 0.2);
}

.image-checkbox {
  position: absolute;
  top: 6px;
  left: 6px;
  z-index: 1;
}

.image-preview {
  width: 100%;
  aspect-ratio: 3/4;
  background: var(--bg-secondary, #f5f5f5);
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.image-preview img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.image-label {
  padding: 6px;
  text-align: center;
  font-size: 12px;
  color: var(--text-secondary, #666);
  background: var(--bg-primary, #fff);
}

.progress-section {
  margin-bottom: 16px;
}

.progress-label {
  font-size: 13px;
  color: var(--text-secondary, #666);
  margin-bottom: 8px;
}

.progress-bar {
  height: 8px;
  background: var(--bg-secondary, #eee);
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--color-primary, #4a90d9);
  transition: width 0.3s ease;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  padding: 16px 20px;
  border-top: 1px solid var(--border-color, #eee);
}

.cancel-btn,
.import-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 20px;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.cancel-btn {
  background: var(--btn-secondary-bg, #f0f0f0);
  border: 1px solid var(--border-color, #ddd);
  color: var(--text-primary, #333);
}

.cancel-btn:hover:not(:disabled) {
  background: var(--btn-secondary-hover-bg, #e5e5e5);
}

.import-btn {
  background: var(--btn-primary-bg, #4a90d9);
  border: none;
  color: #fff;
}

.import-btn:hover:not(:disabled) {
  background: var(--btn-primary-hover-bg, #3a7fc8);
}

.import-btn:disabled,
.cancel-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.loading-spinner {
  width: 14px;
  height: 14px;
  border: 2px solid transparent;
  border-top-color: currentcolor;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

/* @keyframes spin 已迁移到 global.css */
</style>
