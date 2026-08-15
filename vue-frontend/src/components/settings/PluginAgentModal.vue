<template>
  <BaseModal
    v-model="isOpen"
    title="自动生成插件"
    size="full"
    custom-class="plugin-agent-modal"
    width="95vw"
    height="90vh"
    max-height="90vh"
    body-display="flex"
    body-min-height="0"
    @close="handleClose"
  >
    <div class="plugin-agent-layout-shell">
      <div class="plugin-agent-layout">
        <section class="plugin-agent-column plugin-agent-column-left plugin-agent-scroll-column">
          <div class="plugin-agent-block">
            <h3 class="plugin-agent-modal__block-title">任务模式</h3>
            <ProductSegmentedTabs
              class="plugin-agent-mode-switch"
              :tabs="modeTabs"
              :active-tab="mode"
              aria-label="插件 Agent 任务模式"
              @select="handleModeTabSelect"
            />
            <UiField
              v-if="mode === 'modify'"
              class="plugin-agent-field"
              label="目标插件"
              control-id="pluginAgentTargetPlugin"
              variant="settings"
            >
              <UiCombobox
                input-id="pluginAgentTargetPlugin"
                aria-label="目标插件"
                :model-value="selectedPluginId"
                :options="pluginOptions"
                :disabled="unref(isConversationPending)"
                @change="handleSelectedPluginChange"
              />
            </UiField>
          </div>

          <div class="plugin-agent-block">
            <h3 class="plugin-agent-modal__block-title">Agent 设置</h3>
            <AiProviderSelectField
              :model-value="agentSettings.provider"
              input-id="pluginAgentProvider"
              :options="providerOptions"
              :disabled="unref(isRunning)"
              field-class="plugin-agent-field"
              @change="handleProviderChange"
            />
            <AiProviderCredentialFields
              :api-key="agentSettings.apiKey"
              api-key-input-id="pluginAgentApiKey"
              :base-url="agentSettings.customBaseUrl"
              base-url-input-id="pluginAgentBaseUrl"
              :disabled="unref(isRunning)"
              :show-base-url="true"
              :has-stored-credential="unref(hasStoredAgentCredential)"
              field-class="plugin-agent-field"
              api-key-placeholder="请输入 API Key"
              api-key-show-label="显示插件 Agent API Key"
              api-key-hide-label="隐藏插件 Agent API Key"
              base-url-placeholder="可选，自定义服务填写"
              @update:api-key="updateAgentString('apiKey', $event)"
              @update:base-url="updateAgentString('customBaseUrl', $event)"
            />
            <UiField
              class="plugin-agent-field"
              label="模型名称"
              control-id="pluginAgentModelName"
              variant="settings"
            >
              <UiModelPicker
                input-id="pluginAgentModelName"
                :model-value="agentSettings.modelName"
                placeholder="请输入模型名称"
                :disabled="unref(isRunning)"
                :fetching="unref(isFetchingModels)"
                :fetch-disabled="unref(isFetchingModels) || unref(isRunning)"
                :options="modelListOptions"
                :model-count="modelListOptions.length - 1"
                @change="handleModelSelected"
                @update:model-value="handleModelSelected"
                @fetch="fetchModels"
              />
            </UiField>
            <div class="plugin-agent-grid-two">
              <UiField label="RPM" control-id="pluginAgentRpmLimit" variant="settings">
                <UiNumberField
                  input-id="pluginAgentRpmLimit"
                  :model-value="agentSettings.openaiOptions.execution.rpmLimit"
                  aria-label="插件 Agent RPM"
                  :disabled="unref(isRunning)"
                  :min="0"
                  :max="100000"
                  :step="1"
                  @update:model-value="updateAgentNumber('rpmLimit', $event)"
                />
              </UiField>
              <UiField label="业务重试" control-id="pluginAgentBusinessRetries" variant="settings">
                <UiNumberField
                  input-id="pluginAgentBusinessRetries"
                  :model-value="agentSettings.openaiOptions.execution.businessRetries"
                  aria-label="插件 Agent 业务重试"
                  :disabled="unref(isRunning)"
                  :min="0"
                  :max="100"
                  :step="1"
                  @update:model-value="updateAgentNumber('businessRetries', $event)"
                />
              </UiField>
            </div>
            <div class="plugin-agent-grid-two">
              <UiField label="传输重试" control-id="pluginAgentTransportRetries" variant="settings">
                <UiNumberField
                  input-id="pluginAgentTransportRetries"
                  :model-value="agentSettings.openaiOptions.execution.transportRetries"
                  aria-label="插件 Agent 传输重试"
                  :disabled="unref(isRunning)"
                  :min="0"
                  :max="100"
                  :step="1"
                  @update:model-value="updateAgentNumber('transportRetries', $event)"
                />
              </UiField>
              <UiField label="输出选项" variant="settings" control="checkbox">
                <div class="plugin-agent-checkboxes">
                  <UiCheckbox :model-value="agentSettings.openaiOptions.request.forceJsonOutput" :disabled="unref(isRunning)" label="强制 JSON 输出" @update:model-value="updateAgentBoolean('forceJsonOutput', $event)" />
                  <UiCheckbox :model-value="agentSettings.openaiOptions.execution.useStream" :disabled="unref(isRunning)" label="流式调用" @update:model-value="updateAgentBoolean('useStream', $event)" />
                </div>
              </UiField>
            </div>
            <div class="plugin-agent-field">
              <OpenAIExtraBodyEditor :model-value="agentSettings.openaiOptions.request.extraBody" :disabled="unref(isRunning)" @update:model-value="updateAgentExtraBody" />
            </div>
            <ProductActionRow class="plugin-agent-actions" aria-label="Agent 设置操作" justify="start">
              <UiButton variant="secondary" type="button" @click="testConnection" :disabled="unref(isTestingConnection) || unref(isRunning)" size="sm">
                {{ isTestingConnection ? '测试中...' : '测试连接' }}
              </UiButton>
              <UiButton variant="primary" type="button" class="plugin-agent-save-settings-action" @click="saveAgentSettings" :disabled="unref(isSavingAgentSettings) || unref(isRunning)" size="sm">
                {{ isSavingAgentSettings ? '保存中...' : '保存设置' }}
              </UiButton>
            </ProductActionRow>
          </div>

          <div class="plugin-agent-block">
            <h3 class="plugin-agent-modal__block-title">插件开发提示</h3>
            <div v-if="overviewSections.length" class="plugin-agent-overview-sections">
              <section
                v-for="section in overviewSections"
                :key="section.title"
                class="plugin-agent-overview-section"
              >
                <h4 class="plugin-agent-overview-section__title">{{ section.title }}</h4>
                <ul class="plugin-agent-list plugin-agent-overview-section__list">
                  <li
                    v-for="item in section.items"
                    :key="`${section.title}-${item}`"
                    class="plugin-agent-overview-section__list-item"
                  >
                    <div class="plugin-agent-overview-item" v-html="renderMarkdown(item)" />
                  </li>
                </ul>
              </section>
            </div>
            <ul v-else class="plugin-agent-list">
              <li v-for="item in overview" :key="item">{{ item }}</li>
            </ul>
            <h4 class="plugin-agent-modal__examples-title">示例描述</h4>
            <ProductChipList
              v-if="promptExampleItems.length"
              aria-label="插件 Agent 示例描述"
              :items="promptExampleItems"
              @select="handlePromptExampleSelect"
            />
          </div>
        </section>

        <section class="plugin-agent-column plugin-agent-column-center">
          <div class="plugin-agent-block plugin-agent-history-panel">
            <div class="plugin-agent-chat-header">
              <h3 class="plugin-agent-modal__block-title">对话与过程</h3>
              <div class="plugin-agent-inline">
                <UiButton
                  variant="secondary"
                  v-if="isRunning"
                  type="button"
                  class="plugin-agent-cancel-action"
                  :disabled="isSessionCommandPending"
                  @click="cancelExecution" size="sm"
                >
                  {{ isSessionCommandPending ? '请求中...' : '取消执行' }}
                </UiButton>
                <UiButton
                  variant="secondary"
                  v-else-if="session"
                  type="button"
                  class="plugin-agent-clear-session-action"
                  :disabled="isSessionCommandPending"
                  @click="clearSession" size="sm"
                >
                  结束会话
                </UiButton>
                <UiButton
                  variant="secondary"
                  type="button"
                  class="plugin-agent-lock-target-action"
                  v-if="canLockTarget"
                  @click="lockTarget" size="sm"
                >
                  锁定目标插件
                </UiButton>
                <UiButton
                  variant="primary"
                  type="button"
                  class="plugin-agent-start-execution-action"
                  :disabled="!unref(canStartExecution)"
                  @click="startExecution" size="sm"
                >
                  开始执行
                </UiButton>
              </div>
            </div>

            <ProductScrollStack
              ref="messagesContainer"
              class="plugin-agent-messages"
              role="log"
              aria-label="插件 Agent 对话和过程"
              aria-live="polite"
              :empty="messages.length === 0 && timelineItems.length === 0"
              gap="md"
              padding="none"
            >
              <template #empty>
                <ProductStatusBanner title="插件 Agent" tone="neutral" role="note">
                  描述你想创建或修改的插件需求，agent 会先给出方案，再在你确认后执行。
                </ProductStatusBanner>
              </template>

              <ProductMessageBubble
                v-for="message in messages"
                :key="message.id"
                class="plugin-agent-message-bubble"
                :role="message.role"
                :avatar-label="message.role === 'user' ? '你' : 'Agent'"
                :aria-label="message.role === 'user' ? '用户消息' : 'Agent 消息'"
              >
                <template #avatar>
                  <span class="plugin-agent-message-avatar">{{ message.role === 'user' ? '你' : 'AI' }}</span>
                </template>
                <template #meta>
                  <span class="plugin-agent-message-role">{{ message.role === 'user' ? '你' : 'Agent' }}</span>
                </template>
                <div v-if="message.isLoading" class="plugin-agent-message-loading">
                  {{ message.content }}<span class="plugin-agent-loading-dots"></span>
                </div>
                <div
                  v-else
                  class="plugin-agent-message-markdown"
                  v-html="message.role === 'assistant'
                    ? renderMarkdown(getAssistantMessageContent(message.id, message.content))
                    : escapeHtml(message.content)"
                />
              </ProductMessageBubble>

              <div
                v-for="item in timelineItems"
                :key="item.id"
                class="plugin-agent-step-card"
                :class="[
                  `plugin-agent-step-card--${item.kind}`,
                  `plugin-agent-step-card--status-${item.status}`,
                  { 'plugin-agent-step-card--streaming': item.status === 'streaming' },
                ]"
              >
                <div class="plugin-agent-step-card-header">
                  <div class="plugin-agent-step-badge">{{ item.badge }}</div>
                  <div class="plugin-agent-step-meta">
                    <div class="plugin-agent-step-title">{{ item.title }}</div>
                    <div v-if="item.timestampLabel" class="plugin-agent-step-time">{{ item.timestampLabel }}</div>
                  </div>
                </div>
                <div v-if="item.summary" class="plugin-agent-step-summary">{{ item.summary }}</div>
                <div
                  v-if="item.content"
                  class="plugin-agent-step-content"
                  v-html="item.markdown ? renderMarkdown(item.content) : escapeHtml(item.content)"
                />
                <details v-if="item.details.length" class="plugin-agent-step-details">
                  <summary class="plugin-agent-step-details__summary">查看细节</summary>
                  <div v-for="detail in item.details" :key="detail.label" class="plugin-agent-step-detail">
                    <div class="plugin-agent-step-detail-label">{{ detail.label }}</div>
                    <pre class="plugin-agent-step-detail-content">{{ detail.content }}</pre>
                  </div>
                </details>
              </div>

              <ProductLogPanel
                v-if="debugLogItems.length > 0"
                class="plugin-agent-debug-log"
                :expanded="isDebugExpanded"
                :items="debugLogItems"
                title="调试事件"
                aria-label="插件 Agent 调试事件"
                :active-hint="`${debugLogItems.length} 条`"
                @toggle="isDebugExpanded = !isDebugExpanded"
              />
            </ProductScrollStack>
          </div>

          <div class="plugin-agent-block plugin-agent-composer-panel">
            <h3 class="plugin-agent-modal__block-title">输入</h3>
            <div class="plugin-agent-composer">
              <UiTextarea
                v-model="messageInput"
                class="plugin-agent-input"
                variant="panel"
                :disabled="unref(isConversationPending)"
                :rows="4"
                placeholder="例如：做一个 after_translate 插件，把译文里的敏感词替换成更自然的说法。"
              />
              <UiButton
                variant="primary"
                type="button"
                class="plugin-agent-submit-message-action"
                :disabled="!unref(canBeginConversation) || unref(isConversationPending)"
                @click="beginConversation"
              >
                {{ isAwaitingPlanningReply ? '等待回复...' : (session ? '继续对话' : '开始会话') }}
              </UiButton>
            </div>
          </div>
        </section>

        <section class="plugin-agent-column plugin-agent-column-right plugin-agent-scroll-column">
          <div class="plugin-agent-block">
            <h3 class="plugin-agent-modal__block-title">本轮任务工件</h3>
            <div class="plugin-agent-meta-row">
              <span class="plugin-agent-meta-row__label">状态</span>
              <strong class="plugin-agent-meta-row__value">{{ currentRunStateLabel }}</strong>
            </div>
            <div class="plugin-agent-meta-row">
              <span class="plugin-agent-meta-row__label">锁定目标</span>
              <strong class="plugin-agent-meta-row__value">{{ lockedTargetLabel }}</strong>
            </div>
            <div v-if="session?.pending_target" class="plugin-agent-pending-target">
              <h4 class="plugin-agent-modal__pending-target-title">待锁定目标</h4>
              <div>{{ session.pending_target.display_name }} / {{ session.pending_target.plugin_id }}</div>
            </div>
            <div v-if="session?.last_validation" class="plugin-agent-validation">
              <h4 class="plugin-agent-modal__validation-title">最后校验</h4>
              <pre class="plugin-agent-modal__validation-payload">{{ formatEventPayload(session.last_validation) }}</pre>
            </div>
          </div>

          <div class="plugin-agent-block">
            <h3 class="plugin-agent-modal__block-title">触达文件</h3>
            <ProductStatusBanner
              v-if="!session?.touched_files?.length"
              title="暂无文件变更"
              tone="neutral"
              role="note"
            >
              执行后会在这里显示本轮写入或修改的文件
            </ProductStatusBanner>
            <ProductRecordCard
              v-for="filePath in session?.touched_files || []"
              :key="filePath"
              :aria-label="`触达文件：${filePath}`"
            >
              <template #meta>
                <span class="plugin-agent-file-name">{{ filePath }}</span>
              </template>
              <pre class="plugin-agent-file-preview">{{ session?.file_previews?.[filePath] || '' }}</pre>
            </ProductRecordCard>
          </div>
        </section>
      </div>
    </div>
  </BaseModal>
</template>

<script setup lang="ts">
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'

import UiButton from '@/components/ui/UiButton.vue'
import BaseModal from '@/components/common/BaseModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList, { type ProductChipItem } from '@/components/product/ProductChipList.vue'
import ProductLogPanel, { type ProductLogItem } from '@/components/product/ProductLogPanel.vue'
import ProductMessageBubble from '@/components/product/ProductMessageBubble.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductScrollStack from '@/components/product/ProductScrollStack.vue'
import ProductSegmentedTabs, { type ProductSegmentedTab } from '@/components/product/ProductSegmentedTabs.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiCombobox from '@/components/ui/UiCombobox.vue'
import UiField from '@/components/ui/UiField.vue'
import UiModelPicker from '@/components/ui/UiModelPicker.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import OpenAIExtraBodyEditor from '@/components/common/OpenAIExtraBodyEditor.vue'
import AiProviderCredentialFields from '@/components/settings/AiProviderCredentialFields.vue'
import AiProviderSelectField from '@/components/settings/AiProviderSelectField.vue'
import { computed, unref } from 'vue'
import { usePluginAgentModal, type PluginAgentModalEmit, type PluginAgentModalProps } from './usePluginAgentModal'

const props = defineProps<PluginAgentModalProps>()
const emit = defineEmits<PluginAgentModalEmit>()

const {
  isOpen,
  mode,
  selectedPluginId,
  overview,
  overviewSections,
  promptExamples,
  providerOptions,
  pluginOptions,
  session,
  messageInput,
  eventFeed,
  messagesContainer,
  isDebugExpanded,
  isFetchingModels,
  isTestingConnection,
  isSavingAgentSettings,
  isAwaitingPlanningReply,
  isSessionCommandPending,
  agentSettings,
  hasStoredAgentCredential,
  messages,
  modelListOptions,
  timelineItems,
  canBeginConversation,
  canLockTarget,
  canStartExecution,
  isRunning,
  isConversationPending,
  currentRunStateLabel,
  lockedTargetLabel,
  handleModeChange,
  handleSelectedPluginChange,
  handleProviderChange,
  handleModelSelected,
  updateAgentString,
  updateAgentNumber,
  updateAgentBoolean,
  updateAgentExtraBody,
  fetchModels,
  testConnection,
  saveAgentSettings,
  applyExamplePrompt,
  beginConversation,
  lockTarget,
  startExecution,
  cancelExecution,
  clearSession,
  handleClose,
  renderMarkdown,
  getAssistantMessageContent,
  escapeHtml,
  formatEventPayload,
} = usePluginAgentModal(props, emit)

type PluginAgentMode = 'create' | 'modify'

function buildModeTab(id: PluginAgentMode, label: string): ProductSegmentedTab {
  return unref(isConversationPending) ? { id, label, disabled: true } : { id, label }
}

const modeTabs = computed<ProductSegmentedTab[]>(() => [
  buildModeTab('create', '新建插件'),
  buildModeTab('modify', '修改现有插件'),
])

const promptExampleItems = computed<ProductChipItem[]>(() =>
  unref(promptExamples).map(example => ({
    id: example,
    label: example,
    interactive: true,
    tone: 'neutral',
  })),
)

const debugLogItems = computed<ProductLogItem[]>(() =>
  unref(eventFeed).map(event => ({
    id: event.id,
    message: event.type,
    detail: formatEventPayload(event.payload),
    tone: event.type.includes('error') ? 'danger' : event.type.includes('result') ? 'success' : 'accent',
  })),
)

function handleModeTabSelect(tabId: string): void {
  if (tabId !== 'create' && tabId !== 'modify') return
  void handleModeChange(tabId)
}

function handlePromptExampleSelect(exampleId: string | number): void {
  if (typeof exampleId !== 'string') return
  applyExamplePrompt(exampleId)
}
</script>

<style scoped>
.plugin-agent-layout-shell {
  width: 100%;
  height: 100%;
  min-height: 0;
  container: plugin-agent-modal / inline-size;
}

.plugin-agent-layout {
  --plugin-agent-timeline-card-border: color-mix(in srgb, var(--color-border-muted) 70%, transparent);
  --plugin-agent-timeline-card-sheen-start: var(--color-surface-base);
  --plugin-agent-timeline-card-sheen-end: color-mix(in srgb, var(--color-surface-base) 2%, transparent);
  --plugin-agent-timeline-card-shadow: var(--shadow-soft);
  --plugin-agent-timeline-rail-neutral: color-mix(in srgb, var(--color-text-muted) 50%, transparent);
  --plugin-agent-timeline-rail-streaming-start: var(--color-action-brand);
  --plugin-agent-timeline-rail-streaming-end: var(--color-status-info);
  --plugin-agent-timeline-rail-success-start: var(--color-status-success);
  --plugin-agent-timeline-rail-success-end: var(--color-action-success-strong);
  --plugin-agent-timeline-rail-error-start: var(--color-status-error);
  --plugin-agent-timeline-rail-error-end: var(--color-status-error-bright);
  --plugin-agent-timeline-rail-waiting-start: var(--color-status-warning-hover);
  --plugin-agent-timeline-rail-waiting-end: var(--color-status-warning-bright);
  --plugin-agent-timeline-badge-background: color-mix(in srgb, var(--color-action-brand) 10%, transparent);
  --plugin-agent-timeline-badge-text: var(--color-text-brand);

  display: grid;
  grid-template-columns: minmax(260px, 300px) minmax(0, 1fr) minmax(280px, 320px);
  gap: 16px;
  width: 100%;
  height: 100%;
  min-height: 0;
  overflow: hidden;
}

.plugin-agent-column {
  min-width: 0;
  min-height: 0;
}

.plugin-agent-scroll-column {
  display: flex;
  flex-direction: column;
  gap: 16px;
  overflow-y: auto;
  padding-right: 4px;
}

.plugin-agent-column-center {
  display: grid;
  grid-template-rows: minmax(0, 1fr) auto;
  gap: 16px;
  overflow: hidden;
}

@container plugin-agent-modal (max-width: 980px) {
  .plugin-agent-layout {
    grid-template-columns: 1fr;
    height: auto;
    overflow: visible;
  }

  .plugin-agent-scroll-column,
  .plugin-agent-column-center {
    overflow: visible;
    padding-right: 0;
  }
}

.plugin-agent-block {
  border: 1px solid var(--color-border-muted);
  border-radius: 12px;
  background: var(--color-surface-base);
  padding: 16px;
}

.plugin-agent-history-panel {
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}

.plugin-agent-chat-header,
.plugin-agent-inline,
.plugin-agent-meta-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.plugin-agent-actions {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 12px;
}

.plugin-agent-mode-switch {
  --product-segmented-tabs-padding: 0;
  --product-segmented-tabs-border: transparent;
  --product-segmented-tabs-radius: 0;
  --product-segmented-tabs-background: transparent;
  --product-segmented-tabs-tab-padding: 10px 12px;
  --product-segmented-tabs-tab-radius: 10px;
  --product-segmented-tabs-tab-border: 1px solid var(--color-border-muted);
  --product-segmented-tabs-tab-background: var(--color-surface-muted);
  --product-segmented-tabs-active-border: 1px solid var(--color-action-primary);
  --product-segmented-tabs-active-background: var(--color-action-primary);
  --product-segmented-tabs-active-text: var(--color-text-inverse);
  --product-segmented-tabs-active-shadow: none;

  gap: 8px;
}

.plugin-agent-start-execution-action {
  width: auto;
}

.plugin-agent-submit-message-action {
  flex: 0 0 auto;
  height: 80px;
  min-height: 80px;
}

.plugin-agent-input {
  height: 80px;
  min-height: 80px;
  line-height: normal;
}

.plugin-agent-field {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 12px;
}

.plugin-agent-grid-two {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-top: 12px;
}

.plugin-agent-checkboxes {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 8px;
}

.plugin-agent-list {
  margin: 0;
  padding-left: 18px;
}

.plugin-agent-overview-sections {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.plugin-agent-overview-section {
  border: 1px solid var(--color-border-muted);
  border-radius: 10px;
  padding: 12px;
  background: var(--color-surface-subtle);
}

.plugin-agent-overview-section__title {
  margin: 0 0 10px;
  font-size: 14px;
  font-weight: 700;
}

.plugin-agent-overview-section__list {
  margin: 0;
}

.plugin-agent-overview-section__list-item + .plugin-agent-overview-section__list-item {
  margin-top: 8px;
}

.plugin-agent-overview-section__list-item {
  font-size: 13px;
  line-height: 1.6;
}

.plugin-agent-overview-item p {
  margin: 0;
}

.plugin-agent-overview-item strong {
  font-weight: 700;
}

.plugin-agent-messages {
  --product-scroll-stack-empty-justify-content: flex-start;

  margin-top: 16px;
}

.plugin-agent-message-role,
.plugin-agent-file-name {
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 8px;
  color: var(--color-text-supporting);
}

.plugin-agent-message-avatar {
  font-size: 12px;
  font-weight: 700;
}

.plugin-agent-message-loading {
  color: var(--color-text-supporting);
}

.plugin-agent-loading-dots::after {
  content: '';
  animation: plugin-agent-dots 1.2s steps(4, end) infinite;
}

@keyframes plugin-agent-dots {
  0%, 20% { content: ''; }
  40% { content: '.'; }
  60% { content: '..'; }
  80%, 100% { content: '...'; }
}

.plugin-agent-step-card {
  position: relative;
  border: 1px solid var(--plugin-agent-timeline-card-border);
  border-radius: 14px;
  padding: 14px 16px;
  background:
    linear-gradient(180deg, var(--plugin-agent-timeline-card-sheen-start), var(--plugin-agent-timeline-card-sheen-end)),
    var(--color-surface-subtle);
  box-shadow: 0 10px 24px var(--plugin-agent-timeline-card-shadow);
}

.plugin-agent-step-card::before {
  content: '';
  position: absolute;
  left: 0;
  top: 14px;
  bottom: 14px;
  width: 4px;
  border-radius: 999px;
  background: var(--plugin-agent-timeline-rail-neutral);
}

.plugin-agent-step-card--status-streaming::before {
  background: linear-gradient(180deg, var(--plugin-agent-timeline-rail-streaming-start), var(--plugin-agent-timeline-rail-streaming-end));
}

.plugin-agent-step-card--status-success::before {
  background: linear-gradient(180deg, var(--plugin-agent-timeline-rail-success-start), var(--plugin-agent-timeline-rail-success-end));
}

.plugin-agent-step-card--status-error::before {
  background: linear-gradient(180deg, var(--plugin-agent-timeline-rail-error-start), var(--plugin-agent-timeline-rail-error-end));
}

.plugin-agent-step-card--status-waiting::before {
  background: linear-gradient(180deg, var(--plugin-agent-timeline-rail-waiting-start), var(--plugin-agent-timeline-rail-waiting-end));
}

.plugin-agent-step-card-header {
  display: flex;
  gap: 12px;
  align-items: flex-start;
}

.plugin-agent-step-badge {
  flex-shrink: 0;
  min-width: 46px;
  border-radius: 999px;
  padding: 5px 10px;
  background: var(--plugin-agent-timeline-badge-background);
  color: var(--plugin-agent-timeline-badge-text);
  font-size: 12px;
  font-weight: 700;
  text-align: center;
}

.plugin-agent-step-meta {
  flex: 1;
  min-width: 0;
}

.plugin-agent-step-title {
  font-size: 14px;
  font-weight: 700;
  color: var(--color-text-default);
}

.plugin-agent-step-time {
  margin-top: 4px;
  font-size: 12px;
  color: var(--color-text-supporting);
}

.plugin-agent-step-summary {
  margin-top: 10px;
  color: var(--color-text-default);
  line-height: 1.6;
}

.plugin-agent-step-content {
  margin-top: 10px;
  color: var(--color-text-default);
}

.plugin-agent-step-content p {
  margin: 0 0 8px;
}

.plugin-agent-step-content p:last-child {
  margin-bottom: 0;
}

.plugin-agent-step-card--assistant.plugin-agent-step-card--streaming .plugin-agent-step-title::after {
  content: ' ...';
  color: var(--plugin-agent-timeline-badge-text);
}

.plugin-agent-step-details {
  margin-top: 12px;
  border-top: 1px dashed var(--color-border-muted);
  padding-top: 12px;
}

.plugin-agent-step-details__summary {
  cursor: pointer;
  font-size: 12px;
  font-weight: 700;
  color: var(--color-text-supporting);
}

.plugin-agent-step-detail {
  margin-top: 10px;
}

.plugin-agent-step-detail-label {
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 8px;
  color: var(--color-text-supporting);
}

.plugin-agent-step-detail-content,
.plugin-agent-file-preview,
.plugin-agent-modal__validation-payload {
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
  font-size: 12px;
}

.plugin-agent-debug-log {
  margin-top: 4px;
}

.plugin-agent-composer {
  display: flex;
  gap: 12px;
}

.plugin-agent-input {
  flex: 1;
}

.plugin-agent-composer-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.plugin-agent-pending-target {
  margin-top: 12px;
}
</style>
