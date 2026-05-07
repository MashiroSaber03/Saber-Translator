<template>
  <BaseModal
    v-model="isOpen"
    title="自动生成插件"
    size="full"
    custom-class="plugin-agent-modal"
    @close="handleClose"
  >
    <div class="plugin-agent-layout">
      <section class="plugin-agent-column plugin-agent-column-left">
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
          <div class="plugin-agent-inline">
            <button type="button" class="btn btn-secondary btn-sm" @click="testConnection" :disabled="isTestingConnection || isRunning">
              {{ isTestingConnection ? '测试中...' : '测试连接' }}
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
        <div class="plugin-agent-block plugin-agent-chat-block">
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

          <div class="plugin-agent-messages">
            <div v-if="messages.length === 0" class="plugin-agent-empty">
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
                v-html="message.role === 'assistant' ? renderMarkdown(message.content) : escapeHtml(message.content)"
              />
            </div>

            <div v-for="event in eventFeed" :key="`event-${event.id}`" class="plugin-agent-event">
              <div class="plugin-agent-event-type">{{ event.type }}</div>
              <pre class="plugin-agent-event-payload">{{ formatEventPayload(event.payload) }}</pre>
            </div>
          </div>

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

      <section class="plugin-agent-column plugin-agent-column-right">
        <div class="plugin-agent-block">
          <h3>本轮任务工件</h3>
          <div class="plugin-agent-meta-row">
            <span>状态</span>
            <strong>{{ session?.run_state || 'drafting' }}</strong>
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
import { computed, onBeforeUnmount, ref, watch } from 'vue'
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
  getPluginAgentSession,
  lockPluginAgentTarget,
  sendPluginAgentMessage,
  startPluginAgentExecution,
  subscribePluginAgentEvents,
  type PluginAgentEvent,
  type PluginAgentAgentConfig,
  type PluginAgentSession,
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
const isFetchingModels = ref(false)
const isTestingConnection = ref(false)
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
const lockedTargetLabel = computed(() => {
  if (session.value?.locked_target) {
    return `${session.value.locked_target.display_name} (${session.value.locked_target.plugin_id})`
  }
  return '未锁定'
})

function applySession(nextSession: PluginAgentSession | null): void {
  session.value = nextSession
  if (!nextSession) {
    eventFeed.value = []
    return
  }
  selectedPluginId.value = nextSession.selected_plugin_id || nextSession.locked_target?.plugin_id || selectedPluginId.value
  eventFeed.value = nextSession.events || []
}

function getLastEventId(activeSession: PluginAgentSession): number {
  const events = activeSession.events || []
  if (events.length === 0) return 0
  return events[events.length - 1]?.id || 0
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
    session.value = null
    messageInput.value = ''
    stopStreaming()
  }
})

onBeforeUnmount(() => {
  stopStreaming()
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
    if (session.value) {
      selectedPluginId.value = session.value.selected_plugin_id || session.value.locked_target?.plugin_id || selectedPluginId.value
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
  session.value = null
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
    await startStreaming()
  } catch (error) {
    toast.error(error instanceof Error ? error.message : '启动执行失败')
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
      afterId: getLastEventId(activeSession),
      signal: streamAbortController.signal,
      onEvent: async (event) => {
        await refreshSession()
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
    await refreshSession()
  }
}

async function refreshSession(): Promise<void> {
  if (!session.value) return
  const result = await getPluginAgentSession(session.value.session_id)
  if (result.success) {
    applySession(result.session)
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

function handleClose(): void {
  isOpen.value = false
}

function renderMarkdown(content: string): string {
  return marked.parse(content) as string
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
.plugin-agent-layout {
  display: grid;
  grid-template-columns: 300px minmax(0, 1fr) 320px;
  gap: 16px;
  min-height: 70vh;
}

.plugin-agent-column {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.plugin-agent-block {
  border: 1px solid var(--border-color);
  border-radius: 12px;
  background: var(--bg-primary);
  padding: 16px;
}

.plugin-agent-chat-block {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}

.plugin-agent-chat-header,
.plugin-agent-inline,
.plugin-agent-meta-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
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
.plugin-agent-event,
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
.plugin-agent-event-type,
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

.plugin-agent-event-payload,
.plugin-agent-file-preview,
.plugin-agent-validation pre {
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
  font-size: 12px;
}

.plugin-agent-composer {
  display: flex;
  gap: 12px;
  margin-top: 16px;
}

.plugin-agent-composer textarea {
  flex: 1;
}

.plugin-agent-empty {
  color: var(--text-secondary);
  font-size: 13px;
}

.plugin-agent-pending-target {
  margin-top: 12px;
}

@media (max-width: 1180px) {
  .plugin-agent-layout {
    grid-template-columns: 1fr;
  }
}
</style>
