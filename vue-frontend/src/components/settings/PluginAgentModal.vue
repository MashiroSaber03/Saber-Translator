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
    <div class="plugin-agent-layout">
      <section class="plugin-agent-column plugin-agent-column-left plugin-agent-scroll-column">
        <div class="plugin-agent-block">
          <h3>任务模式</h3>
          <div class="plugin-agent-mode-switch">
            <UiButton
              variant="toolbar"
              type="button"
              class="plugin-agent-mode-btn plugin-agent-mode-create"
              :class="{ active: mode === 'create' }"
              :disabled="unref(isRunning)"
              @click="handleModeChange('create')"
            >
              新建插件
            </UiButton>
            <UiButton
              variant="toolbar"
              type="button"
              class="plugin-agent-mode-btn plugin-agent-mode-modify"
              :class="{ active: mode === 'modify' }"
              :disabled="unref(isRunning)"
              @click="handleModeChange('modify')"
            >
              修改现有插件
            </UiButton>
          </div>
          <div v-if="mode === 'modify'" class="plugin-agent-field">
            <label>目标插件</label>
            <CustomSelect
              :model-value="selectedPluginId"
              :options="pluginOptions"
              :disabled="unref(isRunning)"
              @change="handleSelectedPluginChange"
            />
          </div>
        </div>

        <div class="plugin-agent-block">
          <h3>Agent 设置</h3>
          <div class="plugin-agent-field">
            <label>服务商</label>
            <CustomSelect
              :model-value="localAgentSettings.provider"
              :options="providerOptions"
              :disabled="unref(isRunning)"
              @change="handleProviderChange"
            />
          </div>
          <div class="plugin-agent-field">
            <label>API Key</label>
            <UiInput v-model="localAgentSettings.apiKey" :disabled="unref(isRunning)" type="password" placeholder="请输入 API Key" />
          </div>
          <div class="plugin-agent-field">
            <label>Base URL</label>
            <UiInput v-model="localAgentSettings.customBaseUrl" :disabled="unref(isRunning)" type="text" placeholder="可选，自定义服务填写" />
          </div>
          <div class="plugin-agent-field">
            <label>模型名称</label>
            <div class="plugin-agent-inline">
              <UiInput v-model="localAgentSettings.modelName" :disabled="unref(isRunning)" type="text" placeholder="请输入模型名称" />
              <UiButton variant="secondary" type="button" @click="fetchModels" :disabled="unref(isFetchingModels) || unref(isRunning)" size="sm">
                {{ isFetchingModels ? '获取中...' : '获取模型' }}
              </UiButton>
            </div>
            <div v-if="modelListOptions.length > 1" class="plugin-agent-model-select">
              <CustomSelect
                :model-value="localAgentSettings.modelName"
                :options="modelListOptions"
                :disabled="unref(isRunning)"
                @change="handleModelSelected"
              />
              <span class="plugin-agent-model-count">共 {{ modelListOptions.length - 1 }} 个模型</span>
            </div>
          </div>
          <div class="plugin-agent-field plugin-agent-grid-two">
            <div>
              <label>RPM</label>
              <UiInput v-model.number="localAgentSettings.rpmLimit" :disabled="unref(isRunning)" type="number" min="0" step="1" />
            </div>
            <div>
              <label>业务重试</label>
              <UiInput v-model.number="localAgentSettings.businessRetries" :disabled="unref(isRunning)" type="number" min="0" max="10" step="1" />
            </div>
          </div>
          <div class="plugin-agent-field plugin-agent-grid-two">
            <div>
              <label>传输重试</label>
              <UiInput v-model.number="localAgentSettings.transportRetries" :disabled="unref(isRunning)" type="number" min="0" max="10" step="1" />
            </div>
            <div class="plugin-agent-checkboxes">
              <label class="ui-checkbox-label">
                <UiInput v-model="localAgentSettings.forceJsonOutput" :disabled="unref(isRunning)" type="checkbox" />
                强制 JSON 输出
              </label>
              <label class="ui-checkbox-label">
                <UiInput v-model="localAgentSettings.useStream" :disabled="unref(isRunning)" type="checkbox" />
                流式调用
              </label>
            </div>
          </div>
          <div class="plugin-agent-field">
            <OpenAIExtraBodyEditor v-model="localAgentSettings.extraBody" :disabled="unref(isRunning)" />
          </div>
          <div class="plugin-agent-actions">
            <UiButton variant="secondary" type="button" @click="testConnection" :disabled="unref(isTestingConnection) || unref(isRunning)" size="sm">
              {{ isTestingConnection ? '测试中...' : '测试连接' }}
            </UiButton>
            <UiButton variant="primary" type="button" class="plugin-agent-save-settings-btn" @click="saveAgentSettings" :disabled="unref(isSavingAgentSettings) || unref(isRunning)" size="sm">
              {{ isSavingAgentSettings ? '保存中...' : '保存设置' }}
            </UiButton>
          </div>
        </div>

        <div class="plugin-agent-block">
          <h3>插件开发提示</h3>
          <div v-if="overviewSections.length" class="plugin-agent-overview-sections">
            <section
              v-for="section in overviewSections"
              :key="section.title"
              class="plugin-agent-overview-section"
            >
              <h4>{{ section.title }}</h4>
              <ul class="plugin-agent-list">
                <li v-for="item in section.items" :key="`${section.title}-${item}`">
                  <div class="plugin-agent-overview-item" v-html="renderMarkdown(item)" />
                </li>
              </ul>
            </section>
          </div>
          <ul v-else class="plugin-agent-list">
            <li v-for="item in overview" :key="item">{{ item }}</li>
          </ul>
          <h4>示例描述</h4>
          <UiButton
            variant="toolbar"
            v-for="example in promptExamples"
            :key="example"
            type="button"
            class="plugin-agent-example"
            @click="applyExamplePrompt(example)"
          >
            {{ example }}
          </UiButton>
        </div>
      </section>

      <section class="plugin-agent-column plugin-agent-column-center">
        <div class="plugin-agent-block plugin-agent-history-panel">
          <div class="plugin-agent-chat-header">
            <h3>对话与过程</h3>
            <div class="plugin-agent-inline">
              <UiButton
                variant="secondary"
                v-if="isRunning"
                type="button"
                class="plugin-agent-cancel-btn"
                @click="cancelExecution" size="sm"
              >
                取消执行
              </UiButton>
              <UiButton
                variant="secondary"
                v-else-if="session"
                type="button"
                class="plugin-agent-clear-btn"
                @click="clearSession" size="sm"
              >
                结束会话
              </UiButton>
              <UiButton
                variant="secondary"
                type="button"
                class="plugin-agent-lock-btn"
                v-if="canLockTarget"
                @click="lockTarget" size="sm"
              >
                锁定目标插件
              </UiButton>
              <UiButton
                variant="primary"
                type="button"
                class="plugin-agent-start-btn"
                :disabled="!unref(canStartExecution)"
                @click="startExecution" size="sm"
              >
                开始执行
              </UiButton>
            </div>
          </div>

          <div ref="messagesContainer" class="plugin-agent-messages">
            <div v-if="messages.length === 0 && timelineItems.length === 0" class="plugin-agent-empty">
              描述你想创建或修改的插件需求，agent 会先给出方案，再在你确认后执行。
            </div>
            <div
              v-for="message in messages"
              :key="message.id"
              class="plugin-agent-message"
              :class="[`role-${message.role}`, { 'is-loading': message.isLoading }]"
            >
              <div class="plugin-agent-message-role">{{ message.role === 'user' ? '你' : 'Agent' }}</div>
              <div v-if="message.isLoading" class="plugin-agent-message-content plugin-agent-message-loading">
                {{ message.content }}<span class="plugin-agent-loading-dots"></span>
              </div>
              <div
                v-else
                class="plugin-agent-message-content"
                v-html="message.role === 'assistant'
                  ? renderMarkdown(getAssistantMessageContent(message.id, message.content))
                  : escapeHtml(message.content)"
              />
            </div>

            <div
              v-for="item in timelineItems"
              :key="item.id"
              class="plugin-agent-step-card"
              :class="[
                `plugin-agent-step-card-${item.kind}`,
                `status-${item.status}`,
                { streaming: item.status === 'streaming' },
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
                <summary>查看细节</summary>
                <div v-for="detail in item.details" :key="detail.label" class="plugin-agent-step-detail">
                  <div class="plugin-agent-step-detail-label">{{ detail.label }}</div>
                  <pre class="plugin-agent-step-detail-content">{{ detail.content }}</pre>
                </div>
              </details>
            </div>

            <div v-if="eventFeed.length > 0" class="plugin-agent-debug-shell">
              <UiButton
                variant="secondary"
                type="button"
                class="plugin-agent-debug-toggle"
                @click="isDebugExpanded = !isDebugExpanded" size="sm"
              >
                {{ isDebugExpanded ? '隐藏调试事件' : '查看调试事件' }}
              </UiButton>
              <div v-if="isDebugExpanded" class="plugin-agent-debug-panel">
                <div v-for="event in eventFeed" :key="`event-${event.id}`" class="plugin-agent-event">
                  <div class="plugin-agent-event-type">{{ event.type }}</div>
                  <pre class="plugin-agent-event-payload">{{ formatEventPayload(event.payload) }}</pre>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="plugin-agent-block plugin-agent-composer-panel">
          <h3>输入</h3>
          <div class="plugin-agent-composer">
            <UiTextarea
              v-model="messageInput"
              class="plugin-agent-input"
              :disabled="unref(isConversationPending)"
              :rows="4"
              placeholder="例如：做一个 after_translate 插件，把译文里的敏感词替换成更自然的说法。"
            />
            <UiButton
              variant="primary"
              type="button"
              class="plugin-agent-begin-btn"
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
          <h3>本轮任务工件</h3>
          <div class="plugin-agent-meta-row">
            <span>状态</span>
            <strong>{{ currentRunStateLabel }}</strong>
          </div>
          <div class="plugin-agent-meta-row">
            <span>锁定目标</span>
            <strong>{{ lockedTargetLabel }}</strong>
          </div>
          <div v-if="session?.pending_target" class="plugin-agent-pending-target">
            <h4>待锁定目标</h4>
            <div>{{ session.pending_target.display_name }} / {{ session.pending_target.plugin_id }}</div>
          </div>
          <div v-if="session?.last_validation" class="plugin-agent-validation">
            <h4>最后校验</h4>
            <pre>{{ formatEventPayload(session.last_validation) }}</pre>
          </div>
        </div>

        <div class="plugin-agent-block">
          <h3>触达文件</h3>
          <div v-if="!session?.touched_files?.length" class="plugin-agent-empty">暂无文件变更</div>
          <div v-for="filePath in session?.touched_files || []" :key="filePath" class="plugin-agent-file-card">
            <div class="plugin-agent-file-name">{{ filePath }}</div>
            <pre class="plugin-agent-file-preview">{{ session?.file_previews?.[filePath] || '' }}</pre>
          </div>
        </div>
      </section>
    </div>
  </BaseModal>
</template>

<script setup lang="ts">
import UiInput from '@/components/ui/UiInput.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'

import UiButton from '@/components/ui/UiButton.vue'
import BaseModal from '@/components/common/BaseModal.vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import OpenAIExtraBodyEditor from '@/components/common/OpenAIExtraBodyEditor.vue'
import { unref } from 'vue'
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
  localAgentSettings,
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
</script>

<style scoped>
.plugin-agent-layout {
  --plugin-agent-modal-accent-primary: rgba(255, 255, 255, .94);
  --plugin-agent-modal-accent-secondary: rgba(255, 255, 255, .02);
  --plugin-agent-modal-border-default: #d0d0d0;
  --plugin-agent-modal-border-strong: rgba(148, 163, 184, .28);
  --plugin-agent-modal-shadow-default: rgba(15, 23, 42, .05);
  --plugin-agent-modal-surface-base: #eee;
  --plugin-agent-modal-surface-raised: rgba(37, 99, 235, .08);
  --plugin-agent-modal-surface-muted: rgba(148, 163, 184, .5);
  --plugin-agent-modal-surface-subtle: #2563eb;
  --plugin-agent-modal-surface-hover: #38bdf8;
  --plugin-agent-modal-surface-active: #16a34a;
  --plugin-agent-modal-surface-selected: #4ade80;
  --plugin-agent-modal-surface-overlay: #dc2626;
  --plugin-agent-modal-surface-inverse: #fb7185;
  --plugin-agent-modal-surface-contrast: #d97706;
  --plugin-agent-modal-surface-tint: #fbbf24;
  --plugin-agent-modal-surface-soft: rgba(37, 99, 235, .1);
  --plugin-agent-modal-surface-strong: rgba(15, 23, 42, .03);
  --plugin-agent-modal-text-primary: #757575;
  --plugin-agent-modal-text-secondary: #1d4ed8;

  display: grid;
  grid-template-columns: 300px minmax(0, 1fr) 320px;
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

@media (--breakpoint-modal-wide-down) {
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
  display: flex;
  gap: 8px;
}

.plugin-agent-mode-btn {
  flex: 1;
  border: 1px solid var(--color-border-muted);
  border-radius: 10px;
  background: var(--color-surface-subtle);
  padding: 10px 12px;
  line-height: 16px;
}

.plugin-agent-mode-btn.active {
  background: var(--color-action-primary);
  color: white;
}

.plugin-agent-start-btn.ui-button:disabled,
.plugin-agent-begin-btn.ui-button:disabled {
  border: 1px solid var(--plugin-agent-modal-border-default);
  background: var(--plugin-agent-modal-surface-base);
  color: var(--plugin-agent-modal-text-primary);
  box-shadow: none;
  opacity: 1;
  transform: none;
}

.plugin-agent-start-btn.ui-button,
.plugin-agent-begin-btn.ui-button {
  width: 69px;
  padding-right: 0;
  padding-left: 0;
}

.plugin-agent-start-btn.ui-button {
  height: 24px;
  min-height: 0;
  padding-top: 3px;
  padding-bottom: 3px;
  font-size: 12px;
}

.plugin-agent-begin-btn.ui-button {
  flex: 0 0 69px;
  height: 80px;
  min-height: 80px;
  padding-top: 0;
  padding-bottom: 0;
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
}

.plugin-agent-checkboxes {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 8px;
}

.plugin-agent-checkboxes .ui-checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.plugin-agent-field input,
.plugin-agent-input {
  width: 100%;
  border: 1px solid var(--color-border-muted);
  border-radius: 10px;
  padding: 10px 12px;
  background: var(--color-surface-subtle);
  color: var(--color-text-default);
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

.plugin-agent-overview-section h4 {
  margin: 0 0 10px;
  font-size: 14px;
  font-weight: 700;
}

.plugin-agent-overview-section .plugin-agent-list {
  margin: 0;
}

.plugin-agent-overview-section .plugin-agent-list li + li {
  margin-top: 8px;
}

.plugin-agent-overview-section .plugin-agent-list li {
  font-size: 13px;
  line-height: 1.6;
}

.plugin-agent-overview-item p {
  margin: 0;
}

.plugin-agent-overview-item strong {
  font-weight: 700;
}

.plugin-agent-example {
  display: block;
  width: 100%;
  margin-top: 8px;
  text-align: left;
  border: 1px dashed var(--color-border-muted);
  border-radius: 10px;
  padding: 10px 12px;
  background: var(--color-surface-subtle);
}

.plugin-agent-messages {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  margin-top: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.plugin-agent-message,
.plugin-agent-file-card {
  border: 1px solid var(--color-border-muted);
  border-radius: 10px;
  padding: 12px;
  background: var(--color-surface-subtle);
}

.plugin-agent-message.role-user {
  background: var(--plugin-agent-modal-surface-raised);
}

.plugin-agent-message.is-loading {
  border-style: dashed;
}

.plugin-agent-message-role,
.plugin-agent-file-name {
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 8px;
  color: var(--color-text-supporting);
}

.plugin-agent-message-content p {
  margin: 0 0 8px;
}

.plugin-agent-message-content p:last-child {
  margin-bottom: 0;
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
  border: 1px solid var(--plugin-agent-modal-border-strong);
  border-radius: 14px;
  padding: 14px 16px;
  background:
    linear-gradient(180deg, var(--plugin-agent-modal-accent-primary), var(--plugin-agent-modal-accent-secondary)),
    var(--color-surface-subtle);
  box-shadow: 0 10px 24px var(--plugin-agent-modal-shadow-default);
}

.plugin-agent-step-card::before {
  content: '';
  position: absolute;
  left: 0;
  top: 14px;
  bottom: 14px;
  width: 4px;
  border-radius: 999px;
  background: var(--plugin-agent-modal-surface-muted);
}

.plugin-agent-step-card.status-streaming::before {
  background: linear-gradient(180deg, var(--plugin-agent-modal-surface-subtle), var(--plugin-agent-modal-surface-hover));
}

.plugin-agent-step-card.status-success::before {
  background: linear-gradient(180deg, var(--plugin-agent-modal-surface-active), var(--plugin-agent-modal-surface-selected));
}

.plugin-agent-step-card.status-error::before {
  background: linear-gradient(180deg, var(--plugin-agent-modal-surface-overlay), var(--plugin-agent-modal-surface-inverse));
}

.plugin-agent-step-card.status-waiting::before {
  background: linear-gradient(180deg, var(--plugin-agent-modal-surface-contrast), var(--plugin-agent-modal-surface-tint));
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
  background: var(--plugin-agent-modal-surface-soft);
  color: var(--plugin-agent-modal-text-secondary);
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

.plugin-agent-step-card-assistant.streaming .plugin-agent-step-title::after {
  content: ' ...';
  color: var(--plugin-agent-modal-text-secondary);
}

.plugin-agent-step-details {
  margin-top: 12px;
  border-top: 1px dashed var(--color-border-muted);
  padding-top: 12px;
}

.plugin-agent-step-details summary {
  cursor: pointer;
  font-size: 12px;
  font-weight: 700;
  color: var(--color-text-supporting);
}

.plugin-agent-step-detail {
  margin-top: 10px;
}

.plugin-agent-step-detail-label,
.plugin-agent-event-type {
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 8px;
  color: var(--color-text-supporting);
}

.plugin-agent-step-detail-content,
.plugin-agent-event-payload,
.plugin-agent-file-preview,
.plugin-agent-validation pre {
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
  font-size: 12px;
}

.plugin-agent-debug-shell {
  margin-top: 4px;
}

.plugin-agent-debug-panel {
  margin-top: 12px;
  border: 1px dashed var(--color-border-muted);
  border-radius: 12px;
  padding: 12px;
  background: var(--plugin-agent-modal-surface-strong);
}

.plugin-agent-event {
  border: 1px solid var(--color-border-muted);
  border-radius: 10px;
  padding: 12px;
  background: var(--color-surface-subtle);
}

.plugin-agent-event + .plugin-agent-event {
  margin-top: 10px;
}

.plugin-agent-composer {
  display: flex;
  gap: 12px;
}

.plugin-agent-composer textarea {
  flex: 1;
}

.plugin-agent-composer-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.plugin-agent-empty {
  color: var(--color-text-supporting);
  font-size: 13px;
}

.plugin-agent-pending-target {
  margin-top: 12px;
}
</style>
