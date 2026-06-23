<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiPanel from '@/components/ui/UiPanel.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import WebImportPreprocessSettings from '../WebImportPreprocessSettings.vue'
import type { useWebImportStore } from '@/stores/webImportStore'
import type { WebImportSettings } from '@/types/webImport'

type SettingsTab = 'basic' | 'preprocess' | 'advanced'
type SelectOption = { label: string; value: string | number }

defineProps<{
  activeSettingsTab: SettingsTab
  agentProviderOptions: SelectOption[]
  draftSettings: WebImportSettings
  hasUnsavedSettings: boolean
  isFetchingModels: boolean
  isSavingSettings: boolean
  modelList: string[]
  modelListOptions: SelectOption[]
  providerRequiresApiKey: (provider: string) => boolean
  settingsExpanded: boolean
  showAgentKey: boolean
  showCustomUrl: boolean
  showFirecrawlKey: boolean
  supportsFetchModels: boolean
  testingAgent: boolean
  testingFirecrawl: boolean
  webImportStore: ReturnType<typeof useWebImportStore>
}>()

defineEmits<{
  (event: 'discard-settings'): void
  (event: 'fetch-models'): void
  (event: 'reset-prompt'): void
  (event: 'save-settings'): void
  (event: 'test-agent'): void
  (event: 'test-firecrawl'): void
  (event: 'update:activeSettingsTab', value: SettingsTab): void
  (event: 'update:settingsExpanded', value: boolean): void
  (event: 'update:showAgentKey', value: boolean): void
  (event: 'update:showFirecrawlKey', value: boolean): void
}>()
</script>

<template>
  <div class="web-import-modal__settings-section">
    <UiButton
      variant="toolbar"
      type="button"
      class="web-import-modal__settings-header"
      :aria-expanded="settingsExpanded ? 'true' : 'false'"
      @click="$emit('update:settingsExpanded', !settingsExpanded)"
    >
      <span class="web-import-modal__settings-toggle">{{ settingsExpanded ? '▼' : '▶' }}</span>
      <span class="web-import-modal__settings-title">⚙️ 设置</span>
      <span class="web-import-modal__settings-hint">点击展开配置</span>
    </UiButton>

    <div v-if="settingsExpanded" class="web-import-modal__settings-content">
      <div class="web-import-modal__settings-tabs">
        <UiButton
          variant="toolbar"
          class="web-import-modal__settings-tab"
          :class="{ active: activeSettingsTab === 'basic' }"
          @click="$emit('update:activeSettingsTab', 'basic')"
        >
          基本设置
        </UiButton>
        <UiButton
          variant="toolbar"
          class="web-import-modal__settings-tab"
          :class="{ active: activeSettingsTab === 'preprocess' }"
          @click="$emit('update:activeSettingsTab', 'preprocess')"
        >
          图片预处理
        </UiButton>
        <UiButton
          variant="toolbar"
          class="web-import-modal__settings-tab"
          :class="{ active: activeSettingsTab === 'advanced' }"
          @click="$emit('update:activeSettingsTab', 'advanced')"
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
            :disabled="!hasUnsavedSettings || isSavingSettings"
            @click="$emit('discard-settings')"
          >
            取消修改
          </UiButton>
          <UiButton
            variant="toolbar"
            class="web-import-modal__settings-action-primary"
            :disabled="!hasUnsavedSettings || isSavingSettings"
            @click="$emit('save-settings')"
          >
            {{ isSavingSettings ? '保存中...' : '保存设置' }}
          </UiButton>
        </div>
      </div>

      <div v-show="activeSettingsTab === 'basic'" class="web-import-modal__settings-tab-content">
        <UiPanel variant="settings">
          <h4 class="web-import-modal__group-title">Firecrawl 配置</h4>
          <div class="web-import-modal__form-row">
            <label class="web-import-modal__form-label">API Key</label>
            <div class="password-input-wrapper">
              <UiInput
                :type="showFirecrawlKey ? 'text' : 'password'"
                class="web-import-modal__form-input"
                :value="draftSettings.firecrawl.apiKey"
                placeholder="fc-xxxxxxxxxxxxxxxx"
                @input="webImportStore.setFirecrawlApiKey(($event.target as HTMLInputElement).value)"
              />
              <UiButton variant="toolbar" type="button" class="password-toggle-btn" tabindex="-1" @click="$emit('update:showFirecrawlKey', !showFirecrawlKey)">
                {{ showFirecrawlKey ? '👁' : '👁‍🗨' }}
              </UiButton>
            </div>
            <div class="web-import-modal__form-row web-import-modal__test-action-row">
              <UiButton
                variant="toolbar"
                class="web-import-modal__settings-test-button"
                :disabled="testingFirecrawl || !draftSettings.firecrawl.apiKey"
                @click="$emit('test-firecrawl')"
              >
                {{ testingFirecrawl ? '测试中...' : '测试连接' }}
              </UiButton>
            </div>
          </div>
        </UiPanel>

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
                placeholder="sk-xxxxxxxxxxxxxxxx"
                @input="webImportStore.setAgentApiKey(($event.target as HTMLInputElement).value)"
              />
              <UiButton variant="toolbar" type="button" class="password-toggle-btn" tabindex="-1" @click="$emit('update:showAgentKey', !showAgentKey)">
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
              placeholder="https://api.example.com/v1"
              @input="webImportStore.setAgentBaseUrl(($event.target as HTMLInputElement).value)"
            />
          </div>

          <div class="web-import-modal__form-row">
            <label class="web-import-modal__form-label">模型名称</label>
            <div class="model-input-with-fetch">
              <UiInput
                type="text"
                class="web-import-modal__form-input"
                :value="draftSettings.agent.modelName"
                placeholder="gpt-4o-mini"
                @input="webImportStore.setAgentModelName(($event.target as HTMLInputElement).value)"
              />
              <UiButton
                v-if="supportsFetchModels"
                variant="toolbar"
                type="button"
                class="fetch-models-btn"
                title="获取可用模型列表"
                :disabled="isFetchingModels"
                @click="$emit('fetch-models')"
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
              :disabled="testingAgent || (providerRequiresApiKey(draftSettings.agent.provider) && !draftSettings.agent.apiKey)"
              @click="$emit('test-agent')"
            >
              {{ testingAgent ? '测试中...' : '测试 Agent 连接' }}
            </UiButton>
          </div>
        </UiPanel>

        <UiPanel variant="settings">
          <h4 class="web-import-modal__group-title">
            提取设置
            <UiButton variant="toolbar" class="reset-btn" @click="$emit('reset-prompt')">重置为默认</UiButton>
          </h4>

          <div class="web-import-modal__form-row">
            <label class="web-import-modal__form-label">提取提示词</label>
            <UiTextarea
              class="web-import-modal__form-textarea"
              :value="draftSettings.extraction.prompt"
              rows="6"
              placeholder="输入提取提示词..."
              @input="webImportStore.setExtractionPrompt(($event.target as HTMLTextAreaElement).value)"
            />
          </div>

          <div class="web-import-modal__form-row">
            <label class="web-import-modal__form-label">最大迭代次数</label>
            <UiInput
              type="number"
              class="web-import-modal__form-input web-import-modal__form-input--small"
              :value="draftSettings.extraction.maxIterations"
              min="1"
              max="20"
              @input="webImportStore.setExtractionMaxIterations(Number(($event.target as HTMLInputElement).value))"
            />
          </div>
        </UiPanel>

        <UiPanel variant="settings">
          <h4 class="web-import-modal__group-title">下载设置</h4>

          <div class="web-import-modal__form-grid">
            <div class="web-import-modal__form-row">
              <label class="web-import-modal__form-label">并发数</label>
              <UiInput
                type="number"
                class="web-import-modal__form-input web-import-modal__form-input--small"
                :value="draftSettings.download.concurrency"
                min="1"
                max="10"
                @input="webImportStore.setDownloadConcurrency(Number(($event.target as HTMLInputElement).value))"
              />
            </div>

            <div class="web-import-modal__form-row">
              <label class="web-import-modal__form-label">超时 (秒)</label>
              <UiInput
                type="number"
                class="web-import-modal__form-input web-import-modal__form-input--small"
                :value="draftSettings.download.timeout"
                min="5"
                max="120"
                @input="webImportStore.setDownloadTimeout(Number(($event.target as HTMLInputElement).value))"
              />
            </div>

            <div class="web-import-modal__form-row">
              <label class="web-import-modal__form-label">重试次数</label>
              <UiInput
                type="number"
                class="web-import-modal__form-input web-import-modal__form-input--small"
                :value="draftSettings.download.retries"
                min="0"
                max="5"
                @input="webImportStore.setDownloadRetries(Number(($event.target as HTMLInputElement).value))"
              />
            </div>

            <div class="web-import-modal__form-row">
              <label class="web-import-modal__form-label">下载间隔 (ms)</label>
              <UiInput
                type="number"
                class="web-import-modal__form-input web-import-modal__form-input--small"
                :value="draftSettings.download.delay"
                min="0"
                max="2000"
                step="100"
                @input="webImportStore.setDownloadDelay(Number(($event.target as HTMLInputElement).value))"
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

      <div v-show="activeSettingsTab === 'preprocess'" class="web-import-modal__settings-tab-content">
        <WebImportPreprocessSettings :draft-settings="draftSettings" />
      </div>

      <div v-show="activeSettingsTab === 'advanced'" class="web-import-modal__settings-tab-content">
        <UiPanel variant="settings">
          <h4 class="web-import-modal__group-title">自定义请求头</h4>

          <div class="web-import-modal__form-row">
            <label class="web-import-modal__form-label">Cookie</label>
            <UiInput
              type="text"
              class="web-import-modal__form-input"
              :value="draftSettings.advanced.customCookie"
              placeholder="name=value; name2=value2"
              @input="webImportStore.setCustomCookie(($event.target as HTMLInputElement).value)"
            />
          </div>

          <div class="web-import-modal__form-row">
            <label class="web-import-modal__form-label">Headers (JSON)</label>
            <UiTextarea
              class="web-import-modal__form-textarea"
              :value="draftSettings.advanced.customHeaders"
              rows="3"
              placeholder="{&quot;X-Custom-Header&quot;: &quot;value&quot;}"
              @input="webImportStore.setCustomHeaders(($event.target as HTMLTextAreaElement).value)"
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
</template>

<style scoped>
.web-import-modal__settings-section {
  margin-bottom: 16px;
  overflow: hidden;
  border: 1px solid var(--color-border-muted, var(--color-border-soft));
  border-radius: 8px;
}

.web-import-modal__settings-header {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 12px 14px;
  border: 0;
  background: var(--web-import-modal-extract-surface-muted);
  cursor: pointer;
  font: inherit;
  text-align: left;
  user-select: none;
  transition: background 0.2s;
}

.web-import-modal__settings-header:hover {
  background: var(--web-import-modal-extract-surface-subtle);
}

.web-import-modal__settings-toggle {
  color: var(--color-text-supporting, var(--color-text-subtle));
  font-size: 10px;
}

.web-import-modal__settings-title {
  color: var(--color-text-default, var(--color-text-default));
  font-weight: 500;
  font-size: 14px;
}

.web-import-modal__settings-hint {
  margin-left: auto;
  color: var(--color-text-supporting, var(--color-text-muted));
  font-size: 12px;
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
  font-weight: 500;
  font-size: 13px;
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
  cursor: not-allowed;
  opacity: 0.6;
}

.web-import-modal__settings-tabs {
  display: flex;
  gap: 4px;
  margin-bottom: 16px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--color-border-muted, var(--color-border-soft));
}

.web-import-modal__settings-tab {
  padding: 8px 16px;
  border: none;
  border-radius: 6px 6px 0 0;
  background: transparent;
  color: var(--color-text-supporting, var(--color-text-secondary));
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s;
}

.web-import-modal__settings-tab:hover,
.web-import-modal__settings-tab.active {
  background: var(--color-surface-subtle);
}

.web-import-modal__settings-tab.active {
  color: var(--color-text-default, var(--color-text-default));
  font-weight: 500;
}

.web-import-modal__settings-tab-content {
  max-height: 400px;
  overflow-y: auto;
}

.web-import-modal__group-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 0 0 12px;
  color: var(--color-text-default, var(--color-text-default));
  font-weight: 600;
  font-size: 14px;
}

.web-import-modal__form-row {
  margin-bottom: 12px;
}

.web-import-modal__form-row--inline {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
}

.web-import-modal__form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 12px;
}

.web-import-modal__form-label {
  display: block;
  margin-bottom: 4px;
  color: var(--color-text-supporting, var(--color-text-secondary));
  font-size: 13px;
}

.web-import-modal__form-input,
.web-import-modal__form-textarea {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 6px;
  outline: none;
  background: var(--color-surface-base);
  color: var(--color-text-default, var(--color-text-default));
  font-size: 14px;
  transition: border-color 0.2s;
}

.web-import-modal__form-input:focus,
.web-import-modal__form-textarea:focus {
  border-color: var(--color-action-primary, var(--color-border-info));
}

.web-import-modal__form-input--small {
  width: 100px;
}

.web-import-modal__form-textarea {
  min-height: 80px;
  resize: vertical;
}

.web-import-modal__test-action-row {
  margin-top: 10px;
}

.web-import-modal__settings-test-button--full {
  width: 100%;
}

.reset-btn {
  padding: 4px 10px;
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 4px;
  background: transparent;
  color: var(--color-text-supporting, var(--color-text-secondary));
  font-size: 12px;
  cursor: pointer;
  transition: background 0.2s;
}

.reset-btn:hover {
  background: var(--color-surface-subtle);
}

.ui-checkbox-label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--color-text-default, var(--color-text-default));
  font-size: 14px;
  cursor: pointer;
}

.ui-checkbox-label input[type='checkbox'] {
  width: 16px;
  height: 16px;
  cursor: pointer;
}
</style>
