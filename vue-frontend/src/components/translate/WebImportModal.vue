<script setup lang="ts">
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
    :custom-style="{
      maxWidth: '800px',
      boxShadow: '0 20px 60px var(--web-import-modal-extract-shadow-default)',
      '--ui-dialog-actions-gap': '12px',
      '--ui-dialog-actions-padding': '16px 20px',
      '--ui-dialog-actions-border': '1px solid var(--color-border-muted, var(--color-border-soft))',
      '--web-import-modal-extract-border-default': '#ffe0a0',
      '--web-import-modal-extract-border-strong': '#e6e6e6',
      '--web-import-modal-extract-shadow-default': 'rgba(0, 0, 0, .3)',
      '--web-import-modal-extract-surface-base': '#4a90d9',
      '--web-import-modal-extract-surface-raised': '#3a7fc8',
      '--web-import-modal-extract-surface-muted': '#f9f9f9',
      '--web-import-modal-extract-surface-subtle': '#efefef',
      '--web-import-modal-extract-surface-hover': '#fafafa',
      '--web-import-modal-extract-text-primary': '#28a745',
      '--web-import-modal-extract-text-secondary': '#856404',
      '--web-import-modal-extract-text-muted': '#b26a00',
      '--web-import-modal-extract-text-subtle': '#2f7d32',
      '--web-import-modal-settings-border-default': '#ffc0c0',
      '--web-import-modal-settings-shadow-default': 'rgba(74, 144, 217, .2)',
      '--web-import-modal-settings-surface-base': '#f9f9f9',
      '--web-import-modal-settings-surface-raised': '#1e1e1e',
      '--web-import-modal-settings-surface-muted': '#eee',
      '--web-import-modal-settings-surface-subtle': '#4a90d9',
      '--web-import-modal-settings-surface-hover': '#f0f0f0',
      '--web-import-modal-settings-surface-active': '#e5e5e5',
      '--web-import-modal-settings-surface-selected': '#3a7fc8',
      '--web-import-modal-settings-text-primary': '#4a90d9',
      '--web-import-modal-settings-text-secondary': '#ccc',
      '--web-import-modal-settings-text-muted': '#ce9178',
      '--web-import-modal-settings-text-subtle': '#ec4899',
      '--web-import-modal-settings-text-supporting': '#818cf8',
      '--web-import-modal-settings-text-disabled': '#dcdcaa',
      '--web-import-modal-settings-text-inverse': '#f1f5f9',
      '--web-import-modal-settings-text-brand': '#c00'
    }"
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
                  @change="webImportStore.setAgentForceJsonOutput(($event.target as HTMLInputElement).checked)"
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

<style scoped>
.url-section {
  display: flex;
  gap: 12px;
  margin-bottom: 12px;
}

.url-input {
  flex: 1;
  padding: 10px 14px;
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 8px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
}

.url-section > .url-input {
  flex: 1 1 auto;
  min-width: 0;
}

.url-input:focus {
  border-color: var(--color-action-primary, var(--color-border-info));
}

.engine-select {
  padding: 10px 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 8px;
  font-size: 14px;
  outline: none;
  background: var(--color-surface-base);
  cursor: pointer;
  min-width: 120px;
}

.url-section > .engine-select {
  flex: 0 0 120px;
  width: 120px;
}

.engine-select:focus {
  border-color: var(--color-action-primary, var(--color-border-info));
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
  color: var(--color-text-supporting, var(--color-text-subtle));
}

.hint-supported {
  color: var(--web-import-modal-extract-text-primary);
}

.hint-unsupported {
  color: var(--color-text-supporting, var(--color-text-subtle));
}

.extract-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 18px;
  background: var(--web-import-modal-extract-surface-base);
  color: var(--color-text-inverse);
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  white-space: nowrap;
  transition: background 0.2s;
}

.extract-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.extract-btn:hover:not(:disabled) {
  background: var(--web-import-modal-extract-surface-raised);
}

.notice {
  padding: 10px 14px;
  background: var(--color-surface-warning-subtle);
  border: 1px solid var(--web-import-modal-extract-border-default);
  border-radius: 6px;
  font-size: 13px;
  color: var(--web-import-modal-extract-text-secondary);
  margin-bottom: 16px;
}

/* 设置区域样式 */
.web-import-modal__settings-section {
  margin-bottom: 16px;
  border: 1px solid var(--color-border-muted, var(--color-border-soft));
  border-radius: 8px;
  overflow: hidden;
}

.web-import-modal__settings-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 14px;
  background: var(--web-import-modal-extract-surface-muted);
  cursor: pointer;
  user-select: none;
  transition: background 0.2s;
}

.web-import-modal__settings-header:hover {
  background: var(--web-import-modal-extract-surface-subtle);
}

.web-import-modal__settings-toggle {
  font-size: 10px;
  color: var(--color-text-supporting, var(--color-text-subtle));
}

.web-import-modal__settings-title {
  font-size: 14px;
  font-weight: 500;
  color: var(--color-text-default, var(--color-text-default));
}

.web-import-modal__settings-hint {
  margin-left: auto;
  font-size: 12px;
  color: var(--color-text-supporting, var(--color-text-muted));
}

.web-import-modal__settings-content {
  padding: 16px;
  background: var(--color-surface-base);
}

.web-import-modal__settings-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 16px;
  padding: 12px 14px;
  border: 1px solid var(--color-border-muted, var(--web-import-modal-extract-border-strong));
  border-radius: 10px;
  background: var(--web-import-modal-extract-surface-hover);
}

.web-import-modal__settings-dirty,
.web-import-modal__settings-clean {
  font-size: 13px;
  font-weight: 500;
}

.web-import-modal__settings-dirty {
  color: var(--web-import-modal-extract-text-muted);
}

.web-import-modal__settings-clean {
  color: var(--web-import-modal-extract-text-subtle);
}

.web-import-modal__settings-action-buttons {
  display: flex;
  gap: 10px;
}

.web-import-modal__settings-action-primary,
.web-import-modal__settings-action-secondary {
  padding: 8px 14px;
  border-radius: 8px;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s;
}

.web-import-modal__settings-action-primary {
  border: none;
  background: var(--web-import-modal-extract-surface-base);
  color: var(--color-text-inverse);
}

.web-import-modal__settings-action-primary:hover:not(:disabled) {
  background: var(--web-import-modal-extract-surface-raised);
}

.web-import-modal__settings-action-secondary {
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  background: var(--color-surface-base);
  color: var(--color-text-default, var(--color-text-default));
}

.web-import-modal__settings-action-secondary:hover:not(:disabled) {
  background: var(--web-import-modal-extract-surface-subtle);
}

.web-import-modal__settings-action-primary:disabled,
.web-import-modal__settings-action-secondary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.web-import-modal__settings-tabs {
  display: flex;
  gap: 4px;
  margin-bottom: 16px;
  border-bottom: 1px solid var(--color-border-muted, var(--color-border-soft));
  padding-bottom: 8px;
}

.web-import-modal__settings-tab {
  padding: 8px 16px;
  background: transparent;
  border: none;
  border-radius: 6px 6px 0 0;
  cursor: pointer;
  font-size: 13px;
  color: var(--color-text-supporting, var(--color-text-secondary));
  transition: all 0.2s;
}

.web-import-modal__settings-tab:hover {
  background: var(--color-surface-subtle);
}

.web-import-modal__settings-tab.active {
  background: var(--color-surface-subtle);
  color: var(--color-text-default, var(--color-text-default));
  font-weight: 500;
}

.web-import-modal__settings-tab-content {
  max-height: 400px;
  overflow-y: auto;
}

.web-import-modal__group-title {
  margin: 0 0 12px;
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text-default, var(--color-text-default));
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.web-import-modal__subsection-title {
  margin: 12px 0 8px;
  font-size: 13px;
  font-weight: 500;
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.web-import-modal__form-row {
  margin-bottom: 12px;
}

.web-import-modal__form-row--inline {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}

.web-import-modal__form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 12px;
}

.web-import-modal__form-label {
  display: block;
  margin-bottom: 4px;
  font-size: 13px;
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.web-import-modal__form-input,
.web-import-modal__form-textarea {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 6px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
  background: var(--color-surface-base);
  color: var(--color-text-default, var(--color-text-default));
}

.web-import-modal__form-input:focus,
.web-import-modal__form-textarea:focus {
  border-color: var(--color-action-primary, var(--color-border-info));
}

.web-import-modal__form-input--small {
  width: 100px;
}

.web-import-modal__form-textarea {
  resize: vertical;
  min-height: 80px;
}

.test-action-row {
  margin-top: 10px;
}

.web-import-modal__settings-test-button--full {
  width: 100%;
}

.reset-btn {
  padding: 4px 10px;
  background: transparent;
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  color: var(--color-text-supporting, var(--color-text-secondary));
  transition: background 0.2s;
}

.reset-btn:hover {
  background: var(--color-surface-subtle);
}

.ui-checkbox-label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  cursor: pointer;
  color: var(--color-text-default, var(--color-text-default));
}

.ui-checkbox-label input[type='checkbox'] {
  width: 16px;
  height: 16px;
  cursor: pointer;
}

.logs-section {
  margin-bottom: 16px;
  border: 1px solid var(--color-border-muted, var(--color-border-soft));
  border-radius: 8px;
  overflow: hidden;
}

.logs-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  background: var(--web-import-modal-settings-surface-base);
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  user-select: none;
}

.logs-toggle {
  font-size: 10px;
  color: var(--color-text-supporting, var(--color-text-subtle));
}

.extracting-hint {
  color: var(--color-action-primary, var(--web-import-modal-settings-text-primary));
  font-weight: normal;
  font-size: 13px;
}

.logs-content {
  max-height: 200px;
  overflow-y: auto;
  padding: 12px;
  background: var(--web-import-modal-settings-surface-raised);
  font-family: Consolas, Monaco, monospace;
  font-size: 12px;
}

.log-item {
  padding: 2px 0;
  color: var(--web-import-modal-settings-text-secondary);
}

.log-time {
  color: var(--color-text-subtle);
  margin-right: 8px;
}

.log-info .log-message { color: var(--web-import-modal-settings-text-muted); }

.log-tool-call .log-message { color: var(--web-import-modal-settings-text-subtle); }

.log-tool-result .log-message { color: var(--web-import-modal-settings-text-supporting); }

.log-thinking .log-message { color: var(--web-import-modal-settings-text-disabled); }

.log-error .log-message { color: var(--web-import-modal-settings-text-inverse); }

.error-section {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 14px;
  background: var(--color-surface-neutral-soft);
  border: 1px solid var(--web-import-modal-settings-border-default);
  border-radius: 6px;
  margin-bottom: 16px;
  color: var(--web-import-modal-settings-text-brand);
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
  color: var(--color-text-default, var(--color-text-default));
}

.result-meta {
  display: flex;
  align-items: center;
  gap: 8px;
}

.result-count {
  font-size: 13px;
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.result-engine {
  font-size: 12px;
  color: var(--color-text-supporting, var(--color-text-subtle));
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
  color: var(--color-text-supporting, var(--color-text-secondary));
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
  border: 2px solid var(--color-border-muted, var(--color-border-soft));
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.2s;
}

.image-item:hover {
  border-color: var(--color-action-primary, var(--color-border-info));
}

.image-item.selected {
  border-color: var(--color-action-primary, var(--color-border-info));
  box-shadow: 0 0 0 2px var(--web-import-modal-settings-shadow-default);
}

.image-checkbox {
  position: absolute;
  top: 6px;
  left: 6px;
  z-index: var(--z-local);
}

.image-preview {
  width: 100%;
  aspect-ratio: 3/4;
  background: var(--color-surface-subtle);
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
  color: var(--color-text-supporting, var(--color-text-secondary));
  background: var(--color-surface-base);
}

.progress-section {
  margin-bottom: 16px;
}

.progress-label {
  font-size: 13px;
  color: var(--color-text-supporting, var(--color-text-secondary));
  margin-bottom: 8px;
}

.progress-bar {
  height: 8px;
  background: var(--web-import-modal-settings-surface-muted);
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--color-action-primary, var(--web-import-modal-settings-surface-subtle));
  transition: width 0.3s ease;
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
  background: var(--web-import-modal-settings-surface-hover);
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  color: var(--color-text-default, var(--color-text-default));
}

.cancel-btn:hover:not(:disabled) {
  background: var(--web-import-modal-settings-surface-active);
}

.import-btn {
  background: var(--web-import-modal-settings-surface-subtle);
  border: none;
  color: var(--color-text-inverse);
}

.import-btn:hover:not(:disabled) {
  background: var(--web-import-modal-settings-surface-selected);
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
</style>
