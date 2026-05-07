<template>
  <BaseModal
    v-model="isOpen"
    title="自动生成插件"
    size="full"
    custom-class="plugin-agent-modal"
    @close="handleClose"
  >
    <div class="plugin-agent-layout">
      <section class="plugin-agent-column plugin-agent-column-left plugin-agent-scroll-column">
        <div class="plugin-agent-block">
          <h3>任务模式</h3>
          <div class="plugin-agent-mode-switch">
            <button
              type="button"
              class="plugin-agent-mode-btn plugin-agent-mode-create"
              :class="{ active: mode === 'create' }"
              :disabled="isRunning"
              @click="handleModeChange('create')"
            >
              新建插件
            </button>
            <button
              type="button"
              class="plugin-agent-mode-btn plugin-agent-mode-modify"
              :class="{ active: mode === 'modify' }"
              :disabled="isRunning"
              @click="handleModeChange('modify')"
            >
              修改现有插件
            </button>
          </div>
          <div v-if="mode === 'modify'" class="plugin-agent-field">
            <label>目标插件</label>
            <CustomSelect
              :model-value="selectedPluginId"
              :options="pluginOptions"
              :disabled="isRunning"
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
              :disabled="isRunning"
              @change="handleProviderChange"
            />
          </div>
          <div class="plugin-agent-field">
            <label>API Key</label>
            <input v-model="localAgentSettings.apiKey" :disabled="isRunning" type="password" placeholder="请输入 API Key" />
          </div>
          <div class="plugin-agent-field">
            <label>Base URL</label>
            <input v-model="localAgentSettings.customBaseUrl" :disabled="isRunning" type="text" placeholder="可选，自定义服务填写" />
          </div>
          <div class="plugin-agent-field">
            <label>模型名称</label>
            <div class="plugin-agent-inline">
              <input v-model="localAgentSettings.modelName" :disabled="isRunning" type="text" placeholder="请输入模型名称" />
              <button type="button" class="btn btn-secondary btn-sm" @click="fetchModels" :disabled="isFetchingModels || isRunning">
                {{ isFetchingModels ? '获取中...' : '获取模型' }}
              </button>
            </div>
          </div>
          <div class="plugin-agent-field plugin-agent-grid-two">
            <div>
              <label>RPM</label>
              <input v-model.number="localAgentSettings.rpmLimit" :disabled="isRunning" type="number" min="0" step="1" />
            </div>
            <div>
              <label>业务重试</label>
              <input v-model.number="localAgentSettings.businessRetries" :disabled="isRunning" type="number" min="0" max="10" step="1" />
            </div>
          </div>
          <div class="plugin-agent-field plugin-agent-grid-two">
            <div>
              <label>传输重试</label>
              <input v-model.number="localAgentSettings.transportRetries" :disabled="isRunning" type="number" min="0" max="10" step="1" />
            </div>
            <div class="plugin-agent-checkboxes">
              <label class="checkbox-label">
                <input v-model="localAgentSettings.forceJsonOutput" :disabled="isRunning" type="checkbox" />
                强制 JSON 输出
              </label>
              <label class="checkbox-label">
                <input v-model="localAgentSettings.useStream" :disabled="isRunning" type="checkbox" />
                流式调用
              </label>
            </div>
          </div>
          <div class="plugin-agent-field">
            <OpenAIExtraBodyEditor v-model="localAgentSettings.extraBody" :disabled="isRunning" />
          </div>
          <div class="plugin-agent-actions">
            <button type="button" class="btn btn-secondary btn-sm" @click="testConnection" :disabled="isTestingConnection || isRunning">
              {{ isTestingConnection ? '测试中...' : '测试连接' }}
            </button>
            <button type="button" class="btn btn-primary btn-sm plugin-agent-save-settings-btn" @click="saveAgentSettings" :disabled="isSavingAgentSettings || isRunning">
              {{ isSavingAgentSettings ? '保存中...' : '保存设置' }}
            </button>
          </div>
        </div>

        <div class="plugin-agent-block">
          <h3>插件开发提示</h3>
          <ul class="plugin-agent-list">
            <li v-for="item in overview" :key="item">{{ item }}</li>
          </ul>
          <h4>示例描述</h4>
          <button
            v-for="example in promptExamples"
            :key="example"
            type="button"
            class="plugin-agent-example"
            @click="applyExamplePrompt(example)"
          >
            {{ example }}
          </button>
        </div>
      </section>

      <section class="plugin-agent-column plugin-agent-column-center">
        <div class="plugin-agent-block plugin-agent-history-panel">
          <div class="plugin-agent-chat-header">
            <h3>对话与过程</h3>
            <div class="plugin-agent-inline">
              <button
                v-if="isRunning"
                type="button"
                class="btn btn-secondary btn-sm plugin-agent-cancel-btn"
                @click="cancelExecution"
              >
                取消执行
              </button>
              <button
                v-else-if="session"
                type="button"
                class="btn btn-secondary btn-sm plugin-agent-clear-btn"
                @click="clearSession"
              >
                结束会话
              </button>
              <button
                type="button"
                class="btn btn-secondary btn-sm plugin-agent-lock-btn"
                v-if="canLockTarget"
                @click="lockTarget"
              >
                锁定目标插件
              </button>
              <button
                type="button"
                class="btn btn-primary btn-sm plugin-agent-start-btn"
                :disabled="!canStartExecution"
                @click="startExecution"
              >
                开始执行
              </button>
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
              :class="`role-${message.role}`"
            >
              <div class="plugin-agent-message-role">{{ message.role === 'user' ? '你' : 'Agent' }}</div>
              <div
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
              <button
                type="button"
                class="btn btn-secondary btn-sm plugin-agent-debug-toggle"
                @click="isDebugExpanded = !isDebugExpanded"
              >
                {{ isDebugExpanded ? '隐藏调试事件' : '查看调试事件' }}
              </button>
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
            <textarea
              v-model="messageInput"
              class="plugin-agent-input"
              :disabled="isRunning"
              rows="4"
              placeholder="例如：做一个 after_translate 插件，把译文里的敏感词替换成更自然的说法。"
            />
            <button
              type="button"
              class="btn btn-primary plugin-agent-begin-btn"
              :disabled="!canBeginConversation || isRunning"
              @click="beginConversation"
            >
              {{ session ? '继续对话' : '开始会话' }}
            </button>
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
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { marked } from 'marked'

import BaseModal from '@/components/common/BaseModal.vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import OpenAIExtraBodyEditor from '@/components/common/OpenAIExtraBodyEditor.vue'
import { configApi } from '@/api/config'
import {
  cancelPluginAgentExecution,
  createPluginAgentSession,
  deletePluginAgentSession,
  getPluginAgentSettings,
  lockPluginAgentTarget,
  sendPluginAgentMessage,
  startPluginAgentExecution,
  subscribePluginAgentEvents,
  type PluginAgentAssistantDeltaPayload,
  type PluginAgentAssistantPayload,
  type PluginAgentDonePayload,
  type PluginAgentErrorPayload,
  type PluginAgentEvent,
  type PluginAgentAgentConfig,
  type PluginAgentLogPayload,
  type PluginAgentSession,
  type PluginAgentStatePayload,
  type PluginAgentToolCallPayload,
  type PluginAgentToolResultPayload,
  type PluginAgentValidationPayload,
} from '@/api/pluginAgent'
import { useSettingsStore } from '@/stores/settingsStore'
import type { PluginAgentProvider } from '@/types/settings'
import { useToast } from '@/utils/toast'

const props = defineProps<{
  modelValue: boolean
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'pluginsChanged'): void
}>()

const settingsStore = useSettingsStore()
const toast = useToast()

interface PluginAgentStepDetail {
  label: string
  content: string
}

interface PluginAgentTimelineItem {
  id: string
  kind: 'assistant' | 'done' | 'error' | 'log' | 'state' | 'tool' | 'validation'
  badge: string
  title: string
  summary: string
  content: string
  markdown: boolean
  status: 'error' | 'info' | 'streaming' | 'success' | 'waiting'
  timestampLabel: string
  details: PluginAgentStepDetail[]
}

const stateLabelMap: Record<string, string> = {
  drafting: '等待需求描述',
  awaiting_target_lock: '等待锁定目标插件',
  ready: '已就绪',
  running: '执行中',
  completed: '已完成',
  failed: '执行失败',
  cancelled: '已取消',
}
const shouldAnimateAssistantStream = typeof navigator !== 'undefined'
  ? !navigator.userAgent.toLowerCase().includes('jsdom')
  : true

const isOpen = ref(props.modelValue)
const mode = ref<'create' | 'modify'>('create')
const selectedPluginId = ref('')
const overview = ref<string[]>([])
const promptExamples = ref<string[]>([])
const providerOptions = ref<Array<{ value: string; label: string }>>([])
const pluginOptions = ref<Array<{ value: string; label: string }>>([])
const session = ref<PluginAgentSession | null>(null)
const messageInput = ref('')
const eventFeed = ref<PluginAgentEvent[]>([])
const messagesContainer = ref<HTMLElement | null>(null)
const isDebugExpanded = ref(false)
const isFetchingModels = ref(false)
const isTestingConnection = ref(false)
const isSavingAgentSettings = ref(false)
const assistantMessageDisplayContent = ref<Record<string, string>>({})
const assistantMessageDisplayTargets = ref<Record<string, string>>({})
const assistantDisplayContent = ref<Record<string, string>>({})
const assistantDisplayTargets = ref<Record<string, string>>({})
const assistantMessageDisplayTimers = new Map<string, ReturnType<typeof setInterval>>()
const assistantDisplayTimers = new Map<string, ReturnType<typeof setInterval>>()
let streamAbortController: AbortController | null = null

const localAgentSettings = ref({
  provider: settingsStore.settings.pluginAgent.provider,
  apiKey: settingsStore.settings.pluginAgent.apiKey,
  modelName: settingsStore.settings.pluginAgent.modelName,
  customBaseUrl: settingsStore.settings.pluginAgent.customBaseUrl,
  rpmLimit: settingsStore.settings.pluginAgent.openaiOptions.execution.rpmLimit,
  transportRetries: settingsStore.settings.pluginAgent.openaiOptions.execution.transportRetries,
  businessRetries: settingsStore.settings.pluginAgent.openaiOptions.execution.businessRetries,
  forceJsonOutput: settingsStore.settings.pluginAgent.openaiOptions.request.forceJsonOutput,
  useStream: settingsStore.settings.pluginAgent.openaiOptions.execution.useStream,
  extraBody: settingsStore.settings.pluginAgent.openaiOptions.request.extraBody,
})

const messages = computed(() => session.value?.messages || [])
const timelineItems = computed<PluginAgentTimelineItem[]>(() => (
  buildTimelineItems(eventFeed.value, assistantDisplayContent.value, assistantDisplayTargets.value)
))
const canBeginConversation = computed(() => {
  if (mode.value === 'modify') {
    return Boolean(selectedPluginId.value && messageInput.value.trim())
  }
  return Boolean(messageInput.value.trim())
})
const canLockTarget = computed(() => mode.value === 'create' && Boolean(session.value?.pending_target))
const canStartExecution = computed(() => (
  Boolean(session.value?.locked_target && session.value?.run_state === 'ready')
  && messages.value.some(message => message.role === 'user')
))
const isRunning = computed(() => session.value?.run_state === 'running')
const currentRunStateLabel = computed(() => {
  return stateLabelMap[session.value?.run_state || 'drafting'] || '等待需求描述'
})
const lockedTargetLabel = computed(() => {
  if (session.value?.locked_target) {
    return `${session.value.locked_target.display_name} (${session.value.locked_target.plugin_id})`
  }
  return '未锁定'
})

function applySession(nextSession: PluginAgentSession | null): void {
  const previousSession = session.value
  session.value = nextSession
  if (!nextSession) {
    eventFeed.value = []
    isDebugExpanded.value = false
    clearAssistantDisplayAnimation()
    return
  }
  if ((!nextSession.messages || nextSession.messages.length === 0) && previousSession?.messages?.length) {
    session.value = {
      ...nextSession,
      messages: [...previousSession.messages],
    }
  }
  selectedPluginId.value = nextSession.selected_plugin_id || nextSession.locked_target?.plugin_id || selectedPluginId.value
  if (previousSession?.session_id === nextSession.session_id && eventFeed.value.length > 0) {
    const merged = [...eventFeed.value]
    for (const event of nextSession.events || []) {
      if (!merged.some(existing => existing.id === event.id)) {
        merged.push(event)
      }
    }
    eventFeed.value = merged.sort((left, right) => left.id - right.id)
  } else {
    eventFeed.value = [...(nextSession.events || [])]
  }

  const previousMessageIds = new Set((previousSession?.messages || []).map(message => message.id))
  const shouldAnimatePlanningMessages = previousSession?.session_id === nextSession.session_id
  for (const message of session.value.messages || []) {
    if (message.role !== 'assistant') {
      continue
    }
    if (shouldAnimatePlanningMessages && !previousMessageIds.has(message.id)) {
      setAssistantMessageDisplayTarget(message.id, message.content, { animate: true })
    } else {
      setAssistantMessageDisplayTarget(message.id, message.content, { animate: false })
    }
  }
}

function getLastEventId(): number {
  const events = eventFeed.value
  if (events.length === 0) return 0
  return events[events.length - 1]?.id || 0
}

function scrollHistoryToBottom(): void {
  const element = messagesContainer.value
  if (!element) return
  element.scrollTop = element.scrollHeight
}

async function syncHistoryScrollToBottom(): Promise<void> {
  await nextTick()
  scrollHistoryToBottom()
}

function clearAssistantDisplayAnimation(): void {
  for (const timer of assistantMessageDisplayTimers.values()) {
    clearInterval(timer)
  }
  assistantMessageDisplayTimers.clear()
  assistantMessageDisplayContent.value = {}
  assistantMessageDisplayTargets.value = {}

  for (const timer of assistantDisplayTimers.values()) {
    clearInterval(timer)
  }
  assistantDisplayTimers.clear()
  assistantDisplayContent.value = {}
  assistantDisplayTargets.value = {}
}

function setAssistantMessageDisplayTarget(
  messageId: string,
  targetContent: string,
  options: { animate: boolean },
): void {
  assistantMessageDisplayTargets.value = {
    ...assistantMessageDisplayTargets.value,
    [messageId]: targetContent,
  }

  if (!options.animate || !shouldAnimateAssistantStream) {
    assistantMessageDisplayContent.value = {
      ...assistantMessageDisplayContent.value,
      [messageId]: targetContent,
    }
    const existingTimer = assistantMessageDisplayTimers.get(messageId)
    if (existingTimer) {
      clearInterval(existingTimer)
      assistantMessageDisplayTimers.delete(messageId)
    }
    return
  }

  if (!Object.prototype.hasOwnProperty.call(assistantMessageDisplayContent.value, messageId)) {
    assistantMessageDisplayContent.value = {
      ...assistantMessageDisplayContent.value,
      [messageId]: '',
    }
  }

  if (assistantMessageDisplayTimers.has(messageId)) {
    return
  }

  const tick = () => {
    const current = assistantMessageDisplayContent.value[messageId] || ''
    const target = assistantMessageDisplayTargets.value[messageId] || ''
    if (current === target) {
      const timer = assistantMessageDisplayTimers.get(messageId)
      if (timer) {
        clearInterval(timer)
      }
      assistantMessageDisplayTimers.delete(messageId)
      return
    }

    const step = Math.max(1, Math.ceil((target.length - current.length) / 6))
    assistantMessageDisplayContent.value = {
      ...assistantMessageDisplayContent.value,
      [messageId]: target.slice(0, current.length + step),
    }
    void syncHistoryScrollToBottom()
  }

  tick()
  const timer = setInterval(tick, 16)
  assistantMessageDisplayTimers.set(messageId, timer)
}

function setAssistantDisplayTarget(streamId: string, targetContent: string): void {
  assistantDisplayTargets.value = {
    ...assistantDisplayTargets.value,
    [streamId]: targetContent,
  }

  if (!shouldAnimateAssistantStream) {
    assistantDisplayContent.value = {
      ...assistantDisplayContent.value,
      [streamId]: targetContent,
    }
    return
  }

  if (!Object.prototype.hasOwnProperty.call(assistantDisplayContent.value, streamId)) {
    assistantDisplayContent.value = {
      ...assistantDisplayContent.value,
      [streamId]: '',
    }
  }

  if (assistantDisplayTimers.has(streamId)) {
    return
  }

  const tick = () => {
    const current = assistantDisplayContent.value[streamId] || ''
    const target = assistantDisplayTargets.value[streamId] || ''
    if (current === target) {
      const timer = assistantDisplayTimers.get(streamId)
      if (timer) {
        clearInterval(timer)
      }
      assistantDisplayTimers.delete(streamId)
      return
    }

    const step = Math.max(1, Math.ceil((target.length - current.length) / 6))
    assistantDisplayContent.value = {
      ...assistantDisplayContent.value,
      [streamId]: target.slice(0, current.length + step),
    }
    void syncHistoryScrollToBottom()
  }

  tick()
  const timer = setInterval(tick, 16)
  assistantDisplayTimers.set(streamId, timer)
}

watch(
  () => props.modelValue,
  async (value) => {
    isOpen.value = value
    if (value) {
      await initializeModal()
    } else {
      stopStreaming()
    }
  },
  { immediate: true },
)

watch(isOpen, (value) => {
  if (!value) {
    emit('update:modelValue', false)
  }
})

watch(() => localAgentSettings.value.apiKey, (value) => {
  settingsStore.updatePluginAgent({ apiKey: value })
})
watch(() => localAgentSettings.value.modelName, (value) => {
  settingsStore.updatePluginAgent({ modelName: value })
})
watch(() => localAgentSettings.value.customBaseUrl, (value) => {
  settingsStore.updatePluginAgent({ customBaseUrl: value })
})
watch(() => localAgentSettings.value.rpmLimit, (value) => {
  settingsStore.updatePluginAgent({ rpmLimit: value })
})
watch(() => localAgentSettings.value.transportRetries, (value) => {
  settingsStore.updatePluginAgent({ transportRetries: value })
})
watch(() => localAgentSettings.value.businessRetries, (value) => {
  settingsStore.updatePluginAgent({ businessRetries: value })
})
watch(() => localAgentSettings.value.forceJsonOutput, (value) => {
  settingsStore.updatePluginAgent({ forceJsonOutput: value })
})
watch(() => localAgentSettings.value.useStream, (value) => {
  settingsStore.updatePluginAgent({ useStream: value })
})
watch(() => localAgentSettings.value.extraBody, (value) => {
  settingsStore.updatePluginAgent({ extraBody: value })
})

watch(selectedPluginId, async (value, previousValue) => {
  if (
    mode.value === 'modify'
    && session.value?.session_id
    && value !== previousValue
  ) {
    try {
      await deletePluginAgentSession(session.value.session_id)
    } catch {
      // 忽略切换目标插件时的清理失败
    }
    applySession(null)
    messageInput.value = ''
    stopStreaming()
  }
})

onBeforeUnmount(() => {
  stopStreaming()
  clearAssistantDisplayAnimation()
})

function buildAgentConfig(): PluginAgentAgentConfig {
  return {
    provider: localAgentSettings.value.provider,
    apiKey: localAgentSettings.value.apiKey,
    modelName: localAgentSettings.value.modelName,
    customBaseUrl: localAgentSettings.value.customBaseUrl,
    openaiOptions: settingsStore.settings.pluginAgent.openaiOptions,
  }
}

function syncLocalAgentSettingsFromStore(): void {
  localAgentSettings.value.provider = settingsStore.settings.pluginAgent.provider
  localAgentSettings.value.apiKey = settingsStore.settings.pluginAgent.apiKey
  localAgentSettings.value.modelName = settingsStore.settings.pluginAgent.modelName
  localAgentSettings.value.customBaseUrl = settingsStore.settings.pluginAgent.customBaseUrl
  localAgentSettings.value.rpmLimit = settingsStore.settings.pluginAgent.openaiOptions.execution.rpmLimit
  localAgentSettings.value.transportRetries = settingsStore.settings.pluginAgent.openaiOptions.execution.transportRetries
  localAgentSettings.value.businessRetries = settingsStore.settings.pluginAgent.openaiOptions.execution.businessRetries
  localAgentSettings.value.forceJsonOutput = settingsStore.settings.pluginAgent.openaiOptions.request.forceJsonOutput
  localAgentSettings.value.useStream = settingsStore.settings.pluginAgent.openaiOptions.execution.useStream
  localAgentSettings.value.extraBody = settingsStore.settings.pluginAgent.openaiOptions.request.extraBody
}

async function initializeModal(): Promise<void> {
  try {
    syncLocalAgentSettingsFromStore()
    const result = await getPluginAgentSettings()
    if (!result.success) {
      toast.error(result.error || '加载插件 Agent 设置失败')
      return
    }

    overview.value = result.overview || []
    promptExamples.value = result.prompt_examples || []
    providerOptions.value = result.providers || []
    pluginOptions.value = [{ value: '', label: '-- 选择插件 --' }, ...(result.plugins || []).map(plugin => ({
      value: plugin.id,
      label: plugin.display_name,
    }))]
    if (result.session) {
      applySession(result.session)
    }
    if (session.value) {
      selectedPluginId.value = session.value.selected_plugin_id || session.value.locked_target?.plugin_id || selectedPluginId.value
      if (session.value.run_state === 'running') {
        void startStreaming()
      }
    }
  } catch (error) {
    toast.error(error instanceof Error ? error.message : '加载插件 Agent 设置失败')
  }
}

async function handleModeChange(nextMode: 'create' | 'modify'): Promise<void> {
  if (nextMode === mode.value && !session.value) {
    return
  }
  if (session.value?.session_id) {
    try {
      await deletePluginAgentSession(session.value.session_id)
    } catch {
      // 忽略切换模式时的清理失败，避免阻断 UI
    }
  }
  mode.value = nextMode
  applySession(null)
  selectedPluginId.value = ''
  messageInput.value = ''
  stopStreaming()
}

function handleProviderChange(value: string | number): void {
  const provider = String(value || '') as PluginAgentProvider
  localAgentSettings.value.provider = provider
  settingsStore.setPluginAgentProvider(provider)
  localAgentSettings.value.apiKey = settingsStore.settings.pluginAgent.apiKey
  localAgentSettings.value.modelName = settingsStore.settings.pluginAgent.modelName
  localAgentSettings.value.customBaseUrl = settingsStore.settings.pluginAgent.customBaseUrl
  localAgentSettings.value.rpmLimit = settingsStore.settings.pluginAgent.openaiOptions.execution.rpmLimit
  localAgentSettings.value.transportRetries = settingsStore.settings.pluginAgent.openaiOptions.execution.transportRetries
  localAgentSettings.value.businessRetries = settingsStore.settings.pluginAgent.openaiOptions.execution.businessRetries
  localAgentSettings.value.forceJsonOutput = settingsStore.settings.pluginAgent.openaiOptions.request.forceJsonOutput
  localAgentSettings.value.useStream = settingsStore.settings.pluginAgent.openaiOptions.execution.useStream
  localAgentSettings.value.extraBody = settingsStore.settings.pluginAgent.openaiOptions.request.extraBody
}

function handleSelectedPluginChange(value: string | number): void {
  selectedPluginId.value = String(value || '')
}

function applyExamplePrompt(example: string): void {
  messageInput.value = example
}

async function beginConversation(): Promise<void> {
  try {
    if (!canBeginConversation.value) return

    if (!session.value) {
      const createResult = await createPluginAgentSession({
        mode: mode.value,
        ...(mode.value === 'modify' ? { plugin_id: selectedPluginId.value } : {}),
      })
      if (!createResult.success) {
        toast.error(createResult.error || '创建会话失败')
        return
      }
      applySession(createResult.session)
    }

    if (!messageInput.value.trim()) {
      return
    }

    const activeSession = session.value
    if (!activeSession) {
      toast.error('会话初始化失败')
      return
    }

    const result = await sendPluginAgentMessage(activeSession.session_id, {
      content: messageInput.value.trim(),
      agentConfig: buildAgentConfig(),
    })

    if (!result.success) {
      toast.error(result.error || '发送消息失败')
      return
    }

    applySession(result.session)
    messageInput.value = ''
    await syncHistoryScrollToBottom()
  } catch (error) {
    toast.error(error instanceof Error ? error.message : '发送消息失败')
  }
}

async function lockTarget(): Promise<void> {
  try {
    if (!session.value?.pending_target) return

    const result = await lockPluginAgentTarget(session.value.session_id, session.value.pending_target)
    if (!result.success) {
      toast.error(result.error || '锁定目标失败')
      return
    }
    applySession(result.session)
  } catch (error) {
    toast.error(error instanceof Error ? error.message : '锁定目标失败')
  }
}

async function startExecution(): Promise<void> {
  try {
    if (!session.value || !canStartExecution.value) return

    const result = await startPluginAgentExecution(session.value.session_id, buildAgentConfig())
    if (!result.success) {
      toast.error(result.error || '启动执行失败')
      return
    }

    applySession(result.session)
    await syncHistoryScrollToBottom()
    await startStreaming()
  } catch (error) {
    toast.error(error instanceof Error ? error.message : '启动执行失败')
  }
}

function appendEvent(event: PluginAgentEvent): void {
  eventFeed.value = [...eventFeed.value, event]
  applyEventToSession(event)
}

function applyEventToSession(event: PluginAgentEvent): void {
  if (!session.value) return

  if (event.type === 'assistant_delta') {
    const payload = event.payload as PluginAgentAssistantDeltaPayload
    setAssistantDisplayTarget(payload.stream_id, payload.content || payload.delta || '')
    return
  }

  if (event.type === 'assistant') {
    const payload = event.payload as PluginAgentAssistantPayload
    if (payload.stream_id) {
      setAssistantDisplayTarget(payload.stream_id, payload.message)
    }
  }

  if (event.type === 'state') {
    const payload = event.payload as PluginAgentStatePayload
    if (payload.run_state) {
      session.value.run_state = payload.run_state
    }
    if (Object.prototype.hasOwnProperty.call(payload, 'locked_target')) {
      session.value.locked_target = payload.locked_target ?? null
    }
    if (Object.prototype.hasOwnProperty.call(payload, 'pending_target')) {
      session.value.pending_target = payload.pending_target ?? null
    }
    return
  }

  if (event.type === 'tool_result') {
    const payload = event.payload as PluginAgentToolResultPayload
    for (const filePath of payload.changed_files || []) {
      if (!session.value.touched_files.includes(filePath)) {
        session.value.touched_files.push(filePath)
      }
    }
    const previews = payload.file_previews || {}
    for (const [filePath, preview] of Object.entries(previews)) {
      session.value.file_previews[filePath] = preview
    }
    return
  }

  if (event.type === 'validation') {
    const payload = event.payload as PluginAgentValidationPayload
    session.value.last_validation = payload.details
    return
  }

  if (event.type === 'done') {
    const payload = event.payload as PluginAgentDonePayload
    session.value.run_state = payload.run_state
    session.value.last_validation = payload.validation || session.value.last_validation || null
    session.value.last_error = null
    return
  }

  if (event.type === 'error') {
    const payload = event.payload as PluginAgentErrorPayload
    session.value.run_state = payload.run_state
    session.value.last_error = payload.message
  }
}

async function startStreaming(): Promise<void> {
  if (!session.value) return
  stopStreaming()

  streamAbortController = new AbortController()
  while (session.value?.run_state === 'running' && !streamAbortController.signal.aborted) {
    const activeSession = session.value
    if (!activeSession) break
    await subscribePluginAgentEvents(activeSession.session_id, {
      afterId: getLastEventId(),
      signal: streamAbortController.signal,
      onEvent: async (event) => {
        appendEvent(event)
        await syncHistoryScrollToBottom()
        if (event.type === 'done') {
          emit('pluginsChanged')
          await initializeModal()
        }
      },
      onError: (error) => {
        if (!streamAbortController?.signal.aborted) {
          toast.error(error)
        }
      },
    })
  }
}

async function cancelExecution(): Promise<void> {
  try {
    if (!session.value?.session_id) return
    await cancelPluginAgentExecution(session.value.session_id)
    toast.info('已请求取消执行')
  } catch (error) {
    toast.error(error instanceof Error ? error.message : '取消执行失败')
  }
}

async function clearSession(): Promise<void> {
  try {
    if (!session.value?.session_id) return
    await deletePluginAgentSession(session.value.session_id)
    applySession(null)
    messageInput.value = ''
    stopStreaming()
    await initializeModal()
  } catch (error) {
    toast.error(error instanceof Error ? error.message : '清理会话失败')
  }
}

function stopStreaming(): void {
  streamAbortController?.abort()
  streamAbortController = null
}

async function fetchModels(): Promise<void> {
  isFetchingModels.value = true
  try {
    const result = await configApi.fetchModels(
      localAgentSettings.value.provider,
      localAgentSettings.value.apiKey,
      localAgentSettings.value.customBaseUrl,
    )
    if (result.success && result.models?.length) {
      toast.success(`获取到 ${result.models.length} 个模型`)
    } else {
      toast.warning(result.message || '未获取到可用模型')
    }
  } catch (error) {
    toast.error(error instanceof Error ? error.message : '获取模型失败')
  } finally {
    isFetchingModels.value = false
  }
}

async function testConnection(): Promise<void> {
  isTestingConnection.value = true
  try {
    const result = await configApi.testAiTranslateConnection({
      provider: localAgentSettings.value.provider,
      apiKey: localAgentSettings.value.apiKey,
      modelName: localAgentSettings.value.modelName,
      baseUrl: localAgentSettings.value.customBaseUrl,
    })
    if (result.success) {
      toast.success(result.message || '连接成功')
    } else {
      toast.error(result.message || result.error || '连接失败')
    }
  } catch (error) {
    toast.error(error instanceof Error ? error.message : '连接测试失败')
  } finally {
    isTestingConnection.value = false
  }
}

async function saveAgentSettings(): Promise<void> {
  isSavingAgentSettings.value = true
  try {
    const success = await settingsStore.savePluginAgentSettings()
    if (success) {
      toast.success('Agent 设置已保存')
    } else {
      toast.error('保存 Agent 设置失败')
    }
  } catch (error) {
    toast.error(error instanceof Error ? error.message : '保存 Agent 设置失败')
  } finally {
    isSavingAgentSettings.value = false
  }
}

function buildTimelineItems(
  events: PluginAgentEvent[],
  displayContentMap: Record<string, string>,
  displayTargetMap: Record<string, string>,
): PluginAgentTimelineItem[] {
  const items: PluginAgentTimelineItem[] = []
  const assistantItems = new Map<string, PluginAgentTimelineItem>()
  const toolItems = new Map<string, PluginAgentTimelineItem>()

  for (const event of events) {
    if (event.type === 'assistant_delta') {
      const payload = event.payload as PluginAgentAssistantDeltaPayload
      let item = assistantItems.get(payload.stream_id)
      const displayContent = displayContentMap[payload.stream_id] || payload.content || payload.delta
      if (!item) {
        item = {
          id: `assistant-${payload.stream_id}`,
          kind: 'assistant',
          badge: 'Agent',
          title: '正在编写插件',
          summary: 'Agent 正在输出当前开发说明',
          content: displayContent,
          markdown: true,
          status: 'streaming',
          timestampLabel: formatTimestamp(event.timestamp),
          details: [],
        }
        assistantItems.set(payload.stream_id, item)
        items.push(item)
      } else {
        item.content = displayContent
        item.timestampLabel = formatTimestamp(event.timestamp)
      }
      continue
    }

    if (event.type === 'assistant') {
      const payload = event.payload as PluginAgentAssistantPayload
      if (payload.phase === 'planning') {
        continue
      }
      const streamId = payload.stream_id || `assistant-${event.id}`
      const displayContent = displayContentMap[streamId] || payload.message
      const targetContent = displayTargetMap[streamId] || payload.message
      let item = assistantItems.get(streamId)
      if (!item) {
        item = {
          id: `assistant-${streamId}`,
          kind: 'assistant',
          badge: 'Agent',
          title: '开发说明',
          summary: 'Agent 给出了当前执行说明',
          content: displayContent,
          markdown: true,
          status: displayContent === targetContent ? 'success' : 'streaming',
          timestampLabel: formatTimestamp(event.timestamp),
          details: [],
        }
        assistantItems.set(streamId, item)
        items.push(item)
      } else {
        item.content = displayContent
        item.status = displayContent === targetContent ? 'success' : 'streaming'
        item.timestampLabel = formatTimestamp(event.timestamp)
      }
      continue
    }

    if (event.type === 'tool_call') {
      const payload = event.payload as PluginAgentToolCallPayload
      const details: PluginAgentStepDetail[] = []
      if (payload.args_preview && Object.keys(payload.args_preview).length > 0) {
        details.push({
          label: '参数摘要',
          content: formatEventPayload(payload.args_preview),
        })
      }
      const item: PluginAgentTimelineItem = {
        id: `tool-${payload.group_id}`,
        kind: 'tool',
        badge: '工具',
        title: payload.summary || payload.tool,
        summary: payload.summary || payload.tool,
        content: '',
        markdown: false,
        status: 'streaming',
        timestampLabel: formatTimestamp(event.timestamp),
        details,
      }
      toolItems.set(payload.group_id, item)
      items.push(item)
      continue
    }

    if (event.type === 'tool_result') {
      const payload = event.payload as PluginAgentToolResultPayload
      const item = toolItems.get(payload.group_id)
      const details: PluginAgentStepDetail[] = []
      if (payload.changed_files && payload.changed_files.length > 0) {
        details.push({
          label: '触达文件',
          content: payload.changed_files.join('\n'),
        })
      }
      if (item) {
        item.summary = payload.summary
        item.status = payload.success ? 'success' : 'error'
        item.timestampLabel = formatTimestamp(event.timestamp)
        item.details = [...item.details, ...details]
      } else {
        items.push({
          id: `tool-result-${payload.group_id}`,
          kind: 'tool',
          badge: '工具',
          title: payload.summary || payload.tool,
          summary: payload.summary || payload.tool,
          content: '',
          markdown: false,
          status: payload.success ? 'success' : 'error',
          timestampLabel: formatTimestamp(event.timestamp),
          details,
        })
      }
      continue
    }

    if (event.type === 'validation') {
      const payload = event.payload as PluginAgentValidationPayload
      items.push({
        id: `validation-${event.id}`,
        kind: 'validation',
        badge: '校验',
        title: payload.success ? '插件校验通过' : '插件校验失败',
        summary: payload.summary,
        content: '',
        markdown: false,
        status: payload.success ? 'success' : 'error',
        timestampLabel: formatTimestamp(event.timestamp),
        details: [],
      })
      continue
    }

    if (event.type === 'done') {
      const payload = event.payload as PluginAgentDonePayload
      items.push({
        id: `done-${event.id}`,
        kind: 'done',
        badge: '完成',
        title: payload.summary || '插件开发任务已完成',
        summary: '插件已通过校验并完成刷新。',
        content: payload.message,
        markdown: true,
        status: 'success',
        timestampLabel: formatTimestamp(event.timestamp),
        details: [],
      })
      continue
    }

    if (event.type === 'error') {
      const payload = event.payload as PluginAgentErrorPayload
      items.push({
        id: `error-${event.id}`,
        kind: 'error',
        badge: '错误',
        title: payload.summary || '插件开发任务失败',
        summary: payload.message,
        content: '',
        markdown: false,
        status: 'error',
        timestampLabel: formatTimestamp(event.timestamp),
        details: [],
      })
      continue
    }

    if (event.type === 'log') {
      const payload = event.payload as PluginAgentLogPayload
      items.push({
        id: `log-${event.id}`,
        kind: 'log',
        badge: '日志',
        title: '运行日志',
        summary: payload.message,
        content: '',
        markdown: false,
        status: 'info',
        timestampLabel: formatTimestamp(event.timestamp),
        details: [],
      })
      continue
    }

    if (event.type === 'state') {
      const payload = event.payload as PluginAgentStatePayload
      if (payload.run_state === 'drafting') {
        continue
      }
      items.push({
        id: `state-${event.id}`,
        kind: 'state',
        badge: '状态',
        title: payload.label || payload.run_state,
        summary: payload.message || '',
        content: '',
        markdown: false,
        status: mapRunStateToCardStatus(payload.run_state),
        timestampLabel: formatTimestamp(event.timestamp),
        details: [],
      })
    }
  }

  return items
}

function mapRunStateToCardStatus(runState: string): PluginAgentTimelineItem['status'] {
  if (runState === 'failed' || runState === 'cancelled') {
    return 'error'
  }
  if (runState === 'completed') {
    return 'success'
  }
  if (runState === 'running') {
    return 'streaming'
  }
  if (runState === 'awaiting_target_lock' || runState === 'ready') {
    return 'waiting'
  }
  return 'info'
}

function formatTimestamp(timestamp: string): string {
  const match = timestamp.match(/T(\d{2}:\d{2}:\d{2})/)
  return match?.[1] || timestamp
}

function handleClose(): void {
  isOpen.value = false
}

function renderMarkdown(content: string): string {
  return marked.parse(content) as string
}

function getAssistantMessageContent(messageId: string, fallback: string): string {
  return assistantMessageDisplayContent.value[messageId] ?? fallback
}

function escapeHtml(content: string): string {
  return content
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function formatEventPayload(payload: unknown): string {
  return JSON.stringify(payload, null, 2)
}
</script>

<style scoped>
:deep(.plugin-agent-modal) {
  width: 95vw;
  height: 90vh;
  max-height: 90vh;
}

:deep(.plugin-agent-modal .modal-body) {
  display: flex;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.plugin-agent-layout {
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

.plugin-agent-block {
  border: 1px solid var(--border-color);
  border-radius: 12px;
  background: var(--bg-primary);
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
  border: 1px solid var(--border-color);
  border-radius: 10px;
  background: var(--bg-secondary);
  padding: 10px 12px;
}

.plugin-agent-mode-btn.active {
  background: var(--color-primary);
  color: white;
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

.plugin-agent-field input,
.plugin-agent-input {
  width: 100%;
  border: 1px solid var(--border-color);
  border-radius: 10px;
  padding: 10px 12px;
  background: var(--bg-secondary);
  color: var(--text-primary);
}

.plugin-agent-list {
  margin: 0;
  padding-left: 18px;
}

.plugin-agent-example {
  display: block;
  width: 100%;
  margin-top: 8px;
  text-align: left;
  border: 1px dashed var(--border-color);
  border-radius: 10px;
  padding: 10px 12px;
  background: var(--bg-secondary);
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
  border: 1px solid var(--border-color);
  border-radius: 10px;
  padding: 12px;
  background: var(--bg-secondary);
}

.plugin-agent-message.role-user {
  background: rgba(37, 99, 235, 0.08);
}

.plugin-agent-message-role,
.plugin-agent-file-name {
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 8px;
  color: var(--text-secondary);
}

.plugin-agent-message-content :deep(p) {
  margin: 0 0 8px;
}

.plugin-agent-message-content :deep(p:last-child) {
  margin-bottom: 0;
}

.plugin-agent-step-card {
  position: relative;
  border: 1px solid rgba(148, 163, 184, 0.28);
  border-radius: 14px;
  padding: 14px 16px;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.02), rgba(255, 255, 255, 0)),
    var(--bg-secondary);
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
}

.plugin-agent-step-card::before {
  content: '';
  position: absolute;
  left: 0;
  top: 14px;
  bottom: 14px;
  width: 4px;
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.5);
}

.plugin-agent-step-card.status-streaming::before {
  background: linear-gradient(180deg, #2563eb, #38bdf8);
}

.plugin-agent-step-card.status-success::before {
  background: linear-gradient(180deg, #16a34a, #4ade80);
}

.plugin-agent-step-card.status-error::before {
  background: linear-gradient(180deg, #dc2626, #fb7185);
}

.plugin-agent-step-card.status-waiting::before {
  background: linear-gradient(180deg, #d97706, #fbbf24);
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
  background: rgba(37, 99, 235, 0.1);
  color: #1d4ed8;
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
  color: var(--text-primary);
}

.plugin-agent-step-time {
  margin-top: 4px;
  font-size: 12px;
  color: var(--text-secondary);
}

.plugin-agent-step-summary {
  margin-top: 10px;
  color: var(--text-primary);
  line-height: 1.6;
}

.plugin-agent-step-content {
  margin-top: 10px;
  color: var(--text-primary);
}

.plugin-agent-step-content :deep(p) {
  margin: 0 0 8px;
}

.plugin-agent-step-content :deep(p:last-child) {
  margin-bottom: 0;
}

.plugin-agent-step-card-assistant.streaming .plugin-agent-step-title::after {
  content: ' ...';
  color: #1d4ed8;
}

.plugin-agent-step-details {
  margin-top: 12px;
  border-top: 1px dashed var(--border-color);
  padding-top: 12px;
}

.plugin-agent-step-details summary {
  cursor: pointer;
  font-size: 12px;
  font-weight: 700;
  color: var(--text-secondary);
}

.plugin-agent-step-detail {
  margin-top: 10px;
}

.plugin-agent-step-detail-label,
.plugin-agent-event-type {
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 8px;
  color: var(--text-secondary);
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
  border: 1px dashed var(--border-color);
  border-radius: 12px;
  padding: 12px;
  background: rgba(15, 23, 42, 0.03);
}

.plugin-agent-event {
  border: 1px solid var(--border-color);
  border-radius: 10px;
  padding: 12px;
  background: var(--bg-secondary);
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
  color: var(--text-secondary);
  font-size: 13px;
}

.plugin-agent-pending-target {
  margin-top: 12px;
}

@media (max-width: 1180px) {
  :deep(.plugin-agent-modal .modal-body) {
    overflow-y: auto;
  }

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
</style>
