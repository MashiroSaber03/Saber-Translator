<script setup lang="ts">
import './WebImportModal.global.styles.css'
import UiPanel from '@/components/ui/UiPanel.vue'

import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiSelect from '@/components/ui/UiSelect.vue'

import UiButton from '@/components/ui/UiButton.vue'
import BaseModal from '@/components/common/BaseModal.vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import { unref } from 'vue'
import { useWebImportModal } from './useWebImportModal'
import WebImportPreprocessSettings from './WebImportPreprocessSettings.vue'

const {
  webImportStore,
  urlInput,
  logsExpanded,
  selectedEngine,
  galleryDLAvailable,
  galleryDLSupported,
  checkingSupport,
  settingsExpanded,
  activeSettingsTab,
  testingFirecrawl,
  testingAgent,
  isFetchingModels,
  showFirecrawlKey,
  showAgentKey,
  modelList,
  isVisible,
  status,
  logs,
  extractResult,
  selectedPages,
  selectedCount,
  downloadProgress,
  downloadProgressPercent,
  error,
  isProcessing,
  draftSettings,
  hasUnsavedSettings,
  isSavingSettings,
  showAgentLogs,
  agentProviderOptions,
  supportsFetchModels,
  modelListOptions,
  engineDisplayName,
  isAllSelected,
  getPreviewUrl,
  handleClose,
  handleSaveSettings,
  handleDiscardSettings,
  handleExtract,
  togglePage,
  toggleAll,
  handleImport,
  showCustomUrl,
  handleTestFirecrawl,
  handleTestAgent,
  handleFetchModels,
  handleResetPrompt,
  providerRequiresApiKey,
} = useWebImportModal()
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
      <UiInput
        v-model="urlInput"
        type="url"
        class="url-input"
        placeholder="输入漫画网页 URL，如 https://example.com/chapter-1"
        :disabled="unref(isProcessing)"
        @keyup.enter="handleExtract"
      />
      <UiSelect
        v-model="selectedEngine"
        class="engine-select"
        :disabled="unref(isProcessing)"
      >
        <option value="auto">自动选择</option>
        <option value="gallery-dl">Gallery-DL</option>
        <option value="ai-agent">AI Agent</option>
      </UiSelect>
      <UiButton
        variant="toolbar"
        class="extract-btn"
        :disabled="unref(isProcessing) || !unref(urlInput).trim()"
        @click="handleExtract"
      >
        <span v-if="status === 'extracting'" class="loading-spinner"></span>
        <span v-else>🔍</span>
        {{ status === 'extracting' ? '提取中...' : '开始提取' }}
      </UiButton>
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
    <div class="web-import-modal__settings-section">
      <div class="web-import-modal__settings-header" @click="settingsExpanded = !settingsExpanded">
        <span class="web-import-modal__settings-toggle">{{ settingsExpanded ? '▼' : '▶' }}</span>
        <span class="web-import-modal__settings-title">⚙️ 设置</span>
        <span class="web-import-modal__settings-hint">点击展开配置</span>
      </div>
             
      <div v-if="settingsExpanded" class="web-import-modal__settings-content">
        <!-- 选项卡 -->
        <div class="web-import-modal__settings-tabs">
          <UiButton
            variant="toolbar"
            class="web-import-modal__settings-tab"
            :class="{ active: activeSettingsTab === 'basic' }"
            @click="activeSettingsTab = 'basic'"
          >
            基本设置
          </UiButton>
          <UiButton
            variant="toolbar"
            class="web-import-modal__settings-tab"
            :class="{ active: activeSettingsTab === 'preprocess' }"
            @click="activeSettingsTab = 'preprocess'"
          >
            图片预处理
          </UiButton>
          <UiButton
            variant="toolbar"
            class="web-import-modal__settings-tab"
            :class="{ active: activeSettingsTab === 'advanced' }"
            @click="activeSettingsTab = 'advanced'"
          >
            高级设置
          </UiButton>
        </div>

        <div class="web-import-modal__settings-actions">
          <span v-if="hasUnsavedSettings" class="web-import-modal__settings-dirty">有未保存的修改</span>
          <span v-else class="web-import-modal__settings-clean">设置已同步</span>
          <div class="web-import-modal__settings-action-buttons">
            <UiButton
              variant="toolbar"
              class="web-import-modal__settings-action-secondary"
              :disabled="!unref(hasUnsavedSettings) || unref(isSavingSettings)"
              @click="handleDiscardSettings"
            >
              取消修改
            </UiButton>
            <UiButton
              variant="toolbar"
              class="web-import-modal__settings-action-primary"
              :disabled="!unref(hasUnsavedSettings) || unref(isSavingSettings)"
              @click="() => handleSaveSettings()"
            >
              {{ isSavingSettings ? '保存中...' : '保存设置' }}
            </UiButton>
          </div>
        </div>

        <!-- 基本设置 -->
        <div v-show="activeSettingsTab === 'basic'" class="web-import-modal__settings-tab-content">
          <!-- Firecrawl 配置 -->
          <UiPanel variant="settings">
            <h4 class="web-import-modal__group-title">Firecrawl 配置</h4>
            <div class="web-import-modal__form-row">
              <label class="web-import-modal__form-label">API Key</label>
              <div class="password-input-wrapper">
                <UiInput
                  :type="showFirecrawlKey ? 'text' : 'password'"
                  class="web-import-modal__form-input"
                  :value="draftSettings.firecrawl.apiKey"
                  @input="webImportStore.setFirecrawlApiKey(($event.target as HTMLInputElement).value)"
                  placeholder="fc-xxxxxxxxxxxxxxxx"
                />
                <UiButton variant="toolbar" type="button" class="password-toggle-btn" tabindex="-1" @click="showFirecrawlKey = !showFirecrawlKey">
                  {{ showFirecrawlKey ? '👁' : '👁‍🗨' }}
                </UiButton>
              </div>
              <div class="web-import-modal__form-row web-import-modal__test-action-row">
                <UiButton
                  variant="toolbar"
                  class="web-import-modal__settings-test-button"
                  :disabled="unref(testingFirecrawl) || !unref(draftSettings).firecrawl.apiKey"
                  @click="handleTestFirecrawl"
                >
                  {{ testingFirecrawl ? '测试中...' : '测试连接' }}
                </UiButton>
              </div>
            </div>
          </UiPanel>

          <!-- AI Agent 配置 -->
          <UiPanel variant="settings">
            <h4 class="web-import-modal__group-title">AI Agent 配置</h4>
                  
            <div class="web-import-modal__form-row">
              <label class="web-import-modal__form-label">服务商</label>
              <CustomSelect
                :model-value="draftSettings.agent.provider"
                :options="agentProviderOptions"
                @change="(value) => webImportStore.setAgentProvider(String(value))"
              />
            </div>

            <div v-if="providerRequiresApiKey(draftSettings.agent.provider)" class="web-import-modal__form-row">
              <label class="web-import-modal__form-label">API Key</label>
              <div class="password-input-wrapper">
                <UiInput
                  :type="showAgentKey ? 'text' : 'password'"
                  class="web-import-modal__form-input"
                  :value="draftSettings.agent.apiKey"
                  @input="webImportStore.setAgentApiKey(($event.target as HTMLInputElement).value)"
                  placeholder="sk-xxxxxxxxxxxxxxxx"
                />
                <UiButton variant="toolbar" type="button" class="password-toggle-btn" tabindex="-1" @click="showAgentKey = !showAgentKey">
                  {{ showAgentKey ? '👁' : '👁‍🗨' }}
                </UiButton>
              </div>
            </div>

            <div v-if="showCustomUrl" class="web-import-modal__form-row">
              <label class="web-import-modal__form-label">自定义 API 地址</label>
              <UiInput
                type="url"
                class="web-import-modal__form-input"
                :value="draftSettings.agent.customBaseUrl"
                @input="webImportStore.setAgentBaseUrl(($event.target as HTMLInputElement).value)"
                placeholder="https://api.example.com/v1"
              />
            </div>

            <div class="web-import-modal__form-row">
              <label class="web-import-modal__form-label">模型名称</label>
              <div class="model-input-with-fetch">
                <UiInput
                  type="text"
                  class="web-import-modal__form-input"
                  :value="draftSettings.agent.modelName"
                  @input="webImportStore.setAgentModelName(($event.target as HTMLInputElement).value)"
                  placeholder="gpt-4o-mini"
                />
                <UiButton
                  variant="toolbar"
                  v-if="supportsFetchModels"
                  type="button"
                  class="fetch-models-btn"
                  title="获取可用模型列表"
                  :disabled="unref(isFetchingModels)"
                  @click="handleFetchModels"
                >
                  <span class="fetch-icon">🔍</span>
                  <span class="fetch-text">{{ isFetchingModels ? '获取中...' : '获取模型' }}</span>
                </UiButton>
              </div>
              <div v-if="modelList.length > 0" class="model-select-container">
                <CustomSelect
                  :model-value="draftSettings.agent.modelName"
                  :options="modelListOptions"
                  @change="(value) => webImportStore.setAgentModelName(String(value))"
                />
                <span class="model-count">共 {{ modelList.length }} 个模型</span>
              </div>
            </div>

            <div class="web-import-modal__form-row web-import-modal__form-row--inline">
              <label class="ui-checkbox-label">
                <UiInput
                  type="checkbox"
                  :checked="draftSettings.agent.forceJsonOutput"
                  @change="webImportStore.setAgentForceJson(($event.target as HTMLInputElement).checked)"
                />
                强制 JSON 格式
              </label>
              <label class="ui-checkbox-label">
                <UiInput
                  type="checkbox"
                  :checked="draftSettings.agent.useStream"
                  @change="webImportStore.setAgentUseStream(($event.target as HTMLInputElement).checked)"
                />
                流式调用
              </label>
            </div>

            <div class="web-import-modal__form-row">
              <UiButton
                variant="toolbar"
                class="web-import-modal__settings-test-button web-import-modal__settings-test-button--full"
                :disabled="unref(testingAgent) || (providerRequiresApiKey(unref(draftSettings).agent.provider) && !unref(draftSettings).agent.apiKey)"
                @click="handleTestAgent"
              >
                {{ testingAgent ? '测试中...' : '测试 Agent 连接' }}
              </UiButton>
            </div>
          </UiPanel>

          <!-- 提取设置 -->
          <UiPanel variant="settings">
            <h4 class="web-import-modal__group-title">
              提取设置
              <UiButton variant="toolbar" class="reset-btn" @click="handleResetPrompt">重置为默认</UiButton>
            </h4>

            <div class="web-import-modal__form-row">
              <label class="web-import-modal__form-label">提取提示词</label>
              <UiTextarea
                class="web-import-modal__form-textarea"
                :value="draftSettings.extraction.prompt"
                @input="webImportStore.setExtractionPrompt(($event.target as HTMLTextAreaElement).value)"
                rows="6"
                placeholder="输入提取提示词..."
              />
            </div>

            <div class="web-import-modal__form-row">
              <label class="web-import-modal__form-label">最大迭代次数</label>
              <UiInput
                type="number"
                class="web-import-modal__form-input web-import-modal__form-input--small"
                :value="draftSettings.extraction.maxIterations"
                @input="webImportStore.setExtractionMaxIterations(Number(($event.target as HTMLInputElement).value))"
                min="1"
                max="20"
              />
            </div>
          </UiPanel>

          <!-- 下载设置 -->
          <UiPanel variant="settings">
            <h4 class="web-import-modal__group-title">下载设置</h4>

            <div class="web-import-modal__form-grid">
              <div class="web-import-modal__form-row">
                <label class="web-import-modal__form-label">并发数</label>
                <UiInput
                  type="number"
                  class="web-import-modal__form-input web-import-modal__form-input--small"
                  :value="draftSettings.download.concurrency"
                  @input="webImportStore.setDownloadConcurrency(Number(($event.target as HTMLInputElement).value))"
                  min="1"
                  max="10"
                />
              </div>

              <div class="web-import-modal__form-row">
                <label class="web-import-modal__form-label">超时 (秒)</label>
                <UiInput
                  type="number"
                  class="web-import-modal__form-input web-import-modal__form-input--small"
                  :value="draftSettings.download.timeout"
                  @input="webImportStore.setDownloadTimeout(Number(($event.target as HTMLInputElement).value))"
                  min="5"
                  max="120"
                />
              </div>

              <div class="web-import-modal__form-row">
                <label class="web-import-modal__form-label">重试次数</label>
                <UiInput
                  type="number"
                  class="web-import-modal__form-input web-import-modal__form-input--small"
                  :value="draftSettings.download.retries"
                  @input="webImportStore.setDownloadRetries(Number(($event.target as HTMLInputElement).value))"
                  min="0"
                  max="5"
                />
              </div>

              <div class="web-import-modal__form-row">
                <label class="web-import-modal__form-label">下载间隔 (ms)</label>
                <UiInput
                  type="number"
                  class="web-import-modal__form-input web-import-modal__form-input--small"
                  :value="draftSettings.download.delay"
                  @input="webImportStore.setDownloadDelay(Number(($event.target as HTMLInputElement).value))"
                  min="0"
                  max="2000"
                  step="100"
                />
              </div>
            </div>

            <div class="web-import-modal__form-row">
              <label class="ui-checkbox-label">
                <UiInput
                  type="checkbox"
                  :checked="draftSettings.download.useReferer"
                  @change="webImportStore.setDownloadUseReferer(($event.target as HTMLInputElement).checked)"
                />
                自动添加 Referer
              </label>
            </div>
          </UiPanel>

          <!-- 界面设置 -->
          <UiPanel variant="settings">
            <h4 class="web-import-modal__group-title">界面设置</h4>
            <div class="web-import-modal__form-row web-import-modal__form-row--inline">
              <label class="ui-checkbox-label">
                <UiInput
                  type="checkbox"
                  :checked="draftSettings.ui.showAgentLogs"
                  @change="webImportStore.setShowAgentLogs(($event.target as HTMLInputElement).checked)"
                />
                显示 AI 工作日志
              </label>
              <label class="ui-checkbox-label">
                <UiInput
                  type="checkbox"
                  :checked="draftSettings.ui.autoImport"
                  @change="webImportStore.setAutoImport(($event.target as HTMLInputElement).checked)"
                />
                提取后自动导入
              </label>
            </div>
          </UiPanel>
        </div>

        <!-- 图片预处理 -->
        <div v-show="activeSettingsTab === 'preprocess'" class="web-import-modal__settings-tab-content">
          <WebImportPreprocessSettings :draft-settings="draftSettings" />
        </div>

        <!-- 高级设置 -->
        <div v-show="activeSettingsTab === 'advanced'" class="web-import-modal__settings-tab-content">
          <UiPanel variant="settings" class="web-import-modal__settings-panel">
            <template #title>自定义请求头</template>

            <div class="web-import-modal__form-row">
              <label class="web-import-modal__form-label">Cookie</label>
              <UiInput
                type="text"
                class="web-import-modal__form-input"
                :value="draftSettings.advanced.customCookie"
                @input="webImportStore.setCustomCookie(($event.target as HTMLInputElement).value)"
                placeholder="name=value; name2=value2"
              />
            </div>

            <div class="web-import-modal__form-row">
              <label class="web-import-modal__form-label">Headers (JSON)</label>
              <UiTextarea
                class="web-import-modal__form-textarea"
                :value="draftSettings.advanced.customHeaders"
                @input="webImportStore.setCustomHeaders(($event.target as HTMLTextAreaElement).value)"
                rows="3"
                placeholder="{&quot;X-Custom-Header&quot;: &quot;value&quot;}"
              />
            </div>

            <div class="web-import-modal__form-row">
              <label class="ui-checkbox-label">
                <UiInput
                  type="checkbox"
                  :checked="draftSettings.advanced.bypassProxy"
                  @change="webImportStore.setBypassProxy(($event.target as HTMLInputElement).checked)"
                />
                绕过系统代理 (连接本地服务时使用)
              </label>
            </div>
          </UiPanel>
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
          :class="`log-${log.type.replaceAll('_', '-')}`"
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
          <UiInput
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
            <UiInput
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
      <UiButton variant="toolbar" class="cancel-btn" @click="handleClose" :disabled="unref(status) === 'downloading'">
        取消
      </UiButton>
      <UiButton
        variant="toolbar"
        class="import-btn"
        :disabled="!unref(extractResult)?.success || unref(selectedCount) === 0 || unref(isProcessing)"
        @click="handleImport"
      >
        <span v-if="status === 'downloading'" class="loading-spinner"></span>
        <span v-else>📥</span>
        {{ status === 'downloading' ? '下载中...' : '导入' }}
      </UiButton>
    </template>
  </BaseModal>
</template>

