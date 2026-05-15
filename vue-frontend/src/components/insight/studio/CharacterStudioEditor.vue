<template>
  <main class="studio-editor">
    <div v-if="!localDocument" class="empty-card">
      <div class="empty-mark">角色工坊</div>
      <h2>选择或创建角色文档</h2>
      <p>先从左侧分析候选创建角色，或直接新建空白文档。选择角色后，这里会切换成完整的编辑工作区。</p>
      <div class="empty-grid">
        <article class="empty-tip">
          <strong>从候选开始</strong>
          <p>如果当前书已有增强时间线，优先从分析候选创建角色，能自动带入来源证据和初始设定。</p>
        </article>
        <article class="empty-tip">
          <strong>空白新建或导入</strong>
          <p>如果你想手工构建角色，或者直接导入外部角色卡，也可以从左侧工具区快速开始。</p>
        </article>
      </div>
    </div>

    <template v-else>
      <section class="overview-hero">
        <div class="hero-main">
          <div class="avatar-shell">
            <img v-if="avatarUrl" :src="avatarUrl" alt="角色头像">
            <div v-else class="avatar-placeholder">{{ localDocument.identity.name.slice(0, 1) || '角' }}</div>
          </div>
          <div class="hero-copy">
            <div class="hero-kicker">当前角色</div>
            <h2>{{ localDocument.meta.title || localDocument.identity.name }}</h2>
            <p>{{ localDocument.identity.description || '当前角色还没有完善简介，可以先在“角色设定”里补充角色基底。' }}</p>
            <div class="hero-meta">
              <span class="meta-pill">{{ formatOrigin(localDocument.origin.type) }}</span>
              <span class="meta-pill" v-if="localDocument.origin.source_character">来源: {{ localDocument.origin.source_character }}</span>
              <span class="meta-pill" v-if="localDocument.meta.tags.length > 0">{{ localDocument.meta.tags.length }} 个标签</span>
              <span class="meta-pill" v-if="localDocument.status.frozen_sections.length > 0">已钉住 {{ localDocument.status.frozen_sections.length }} 区块</span>
            </div>
          </div>
        </div>
        <div class="hero-actions">
          <button class="ghost-btn" :disabled="isGenerationLocked" @click="$emit('generate', 'identity')">
            {{ isGenerating('identity') ? '补全中...' : 'AI 补全角色设定' }}
          </button>
          <button class="ghost-btn" :disabled="isGenerationLocked" @click="$emit('generate', 'review')">
            {{ isGenerating('review') ? '审查中...' : 'AI 审查当前角色' }}
          </button>
          <button class="danger-btn" :disabled="pendingState.deleting" @click="$emit('delete')">
            {{ pendingState.deleting ? '删除中...' : '删除文档' }}
          </button>
        </div>
      </section>

      <section class="editor-shell">
        <StudioSectionTabs
          :model-value="activeTab"
          :items="tabItems"
          @update:model-value="$emit('update:activeTab', normalizeTab($event))"
        />

        <section v-if="activeTab === 'overview'" class="panel-stack">
          <div class="summary-grid">
            <article class="summary-card">
              <span class="summary-label">来源摘要</span>
              <strong>{{ formatOrigin(localDocument.origin.type) }}</strong>
              <p v-if="localDocument.origin.source_character">基于 {{ localDocument.origin.source_character }} 的分析结果创建</p>
              <p v-else>当前文档为手工或外部导入角色。</p>
            </article>
            <article class="summary-card">
              <span class="summary-label">运行时资源</span>
              <strong>{{ localDocument.regexScripts.length + localDocument.stateTasks.length }}</strong>
              <p>{{ localDocument.regexScripts.length }} 个脚本 · {{ localDocument.stateTasks.length }} 个任务</p>
            </article>
            <article class="summary-card">
              <span class="summary-label">问候语库存</span>
              <strong>{{ localDocument.coreMessages.alternate_greetings.length + (localDocument.coreMessages.first_message ? 1 : 0) }}</strong>
              <p>1 条主问候 + {{ localDocument.coreMessages.alternate_greetings.length }} 条备用问候</p>
            </article>
            <article class="summary-card">
              <span class="summary-label">知识量</span>
              <strong>{{ flattenedLorebookCount }}</strong>
              <p>世界书树当前共有 {{ flattenedLorebookCount }} 个节点。</p>
            </article>
          </div>

          <div class="workspace-row">
            <section class="workspace-card">
              <div class="card-head">
                <div>
                  <h3>快速入口</h3>
                  <p>直接跳到你现在最可能继续编辑的模块。</p>
                </div>
              </div>
              <div class="quick-grid">
                <button class="quick-card" @click="$emit('update:activeTab', 'character')">
                  <span class="quick-icon">🧬</span>
                  <strong>角色设定</strong>
                  <p>完善简介、性格、场景、标签。</p>
                </button>
                <button class="quick-card" @click="$emit('update:activeTab', 'greetings')">
                  <span class="quick-icon">💬</span>
                  <strong>问候语</strong>
                  <p>打磨主问候和备用开场。</p>
                </button>
                <button class="quick-card" @click="$emit('update:activeTab', 'lorebook')">
                  <span class="quick-icon">📚</span>
                  <strong>世界书</strong>
                  <p>维护角色知识树和触发条目。</p>
                </button>
                <button class="quick-card" @click="$emit('update:activeTab', 'scripts')">
                  <span class="quick-icon">⚙️</span>
                  <strong>脚本任务</strong>
                  <p>配置正则脚本和状态任务。</p>
                </button>
              </div>
            </section>

            <section class="workspace-card">
              <div class="card-head">
                <div>
                  <h3>保护设置</h3>
                  <p>被钉住的区块不会被 AI 再生成或 Agent patch 覆盖。</p>
                </div>
              </div>
              <div class="freeze-grid">
                <label v-for="item in freezeItems" :key="item.key" class="freeze-item">
                  <span class="freeze-item-label">{{ item.label }}</span>
                  <span class="freeze-item-control">
                    <input :checked="isFrozen(item.key)" type="checkbox" @change="toggleFrozen(item.key, $event)">
                  </span>
                </label>
              </div>
            </section>
          </div>

          <div class="workspace-row single">
            <section class="workspace-card">
              <div class="card-head">
                <div>
                  <h3>最近诊断摘要</h3>
                  <p>导出前先看这里，能快速判断当前角色是否存在结构性问题。</p>
                </div>
                <button class="ghost-btn small" :disabled="pendingState.validating" @click="$emit('validate')">
                  {{ pendingState.validating ? '诊断中...' : '重新诊断' }}
                </button>
              </div>
              <DiagnosticsPanel :diagnostics="diagnostics" />
            </section>
          </div>

          <div v-if="latestReview" class="workspace-row single">
            <section class="workspace-card">
              <div class="card-head">
                <div>
                  <h3>最近审查</h3>
                  <p>这里展示最近一次“AI 审查当前角色”的结果，方便你直接据此继续补卡。</p>
                </div>
              </div>
              <div class="review-summary">
                <strong>{{ latestReview.summary }}</strong>
                <ul v-if="latestReview.issues.length > 0" class="review-list">
                  <li v-for="(item, index) in latestReview.issues" :key="`review-issue-${index}`">{{ item }}</li>
                </ul>
                <ul v-if="latestReview.suggestions.length > 0" class="review-list suggestions">
                  <li v-for="(item, index) in latestReview.suggestions" :key="`review-suggestion-${index}`">{{ item }}</li>
                </ul>
              </div>
            </section>
          </div>
        </section>

        <section v-else-if="activeTab === 'character'" class="panel-stack">
          <section class="workspace-card">
            <div class="card-head">
              <div>
                <h3>角色设定</h3>
                <p>聚合角色身份与世界观上下文，优先把角色基底写清楚，再去扩展运行时能力。</p>
              </div>
              <div class="head-actions">
                <button class="ghost-btn small" :disabled="isGenerationLocked" @click="$emit('generate', 'identity')">
                  {{ isGenerating('identity') ? '重写中...' : 'AI 重写本区' }}
                </button>
                <button class="ghost-btn small" :disabled="isGenerationLocked" @click="$emit('generate', 'translate')">
                  {{ isGenerating('translate') ? '翻译中...' : '整卡翻译' }}
                </button>
              </div>
            </div>

            <div class="form-grid">
              <label>
                角色名称
                <input v-model="localDocument.identity.name" type="text">
              </label>
              <label>
                别名（逗号分隔）
                <input :value="localDocument.identity.aliases.join(', ')" type="text" @input="updateAliases($event)">
              </label>
              <label class="full">
                角色简介
                <textarea v-model="localDocument.identity.description" rows="6"></textarea>
              </label>
              <label class="full">
                性格 / 人设
                <textarea v-model="localDocument.identity.personality" rows="5"></textarea>
              </label>
              <label class="full">
                当前场景
                <textarea v-model="localDocument.identity.scenario" rows="5"></textarea>
              </label>
              <label class="full">
                标签（逗号分隔）
                <input :value="localDocument.meta.tags.join(', ')" type="text" @input="updateTags($event)">
              </label>
              <div class="full option-row">
                <label class="toggle-chip">
                  <input v-model="localDocument.status.is_favorite" type="checkbox">
                  <span>收藏当前角色</span>
                </label>
                <label class="toggle-chip">
                  <input :checked="isFrozen('identity')" type="checkbox" @change="toggleFrozen('identity', $event)">
                  <span>钉住角色设定</span>
                </label>
              </div>
            </div>
          </section>
        </section>

        <section v-else-if="activeTab === 'greetings'" class="panel-stack">
          <GreetingWorkbench
            :first-message="localDocument.coreMessages.first_message"
            :alternates="localDocument.coreMessages.alternate_greetings"
            :generating="isGenerating('greetings')"
            @update:first-message="localDocument.coreMessages.first_message = $event"
            @update:item="updateGreeting"
            @add="addGreeting"
            @remove="removeGreeting"
            @move="moveGreeting"
            @promote="useAsPrimary"
            @generate="$emit('generate', 'greetings')"
          />

          <section class="workspace-card">
            <div class="card-head">
              <div>
                <h3>对话元信息</h3>
                <p>这里的系统提示词和示例对话会影响预览聊天和导出卡片的整体语气。</p>
              </div>
            </div>
            <div class="form-grid">
              <label class="full">
                示例对话
                <textarea v-model="localDocument.coreMessages.message_example" rows="5"></textarea>
              </label>
              <label class="full">
                系统提示词（System Prompt）
                <textarea v-model="localDocument.coreMessages.system_prompt" rows="4"></textarea>
              </label>
              <label class="full">
                历史后置说明（Post History）
                <textarea v-model="localDocument.coreMessages.post_history_instructions" rows="3"></textarea>
              </label>
              <label class="full">
                备注
                <textarea v-model="localDocument.coreMessages.creator_notes" rows="3"></textarea>
              </label>
              <label>
                角色版本
                <input v-model="localDocument.coreMessages.character_version" type="text">
              </label>
              <div class="option-row">
                <label class="toggle-chip">
                  <input :checked="isFrozen('greetings')" type="checkbox" @change="toggleFrozen('greetings', $event)">
                  <span>钉住问候语区</span>
                </label>
              </div>
            </div>
          </section>
        </section>

        <section v-else-if="activeTab === 'lorebook'" class="panel-stack">
          <section class="workspace-card">
            <div class="card-head">
              <div>
                <h3>世界书</h3>
                <p>把角色设定、关系、场景、专有名词沉淀成可命中的知识树。条目设计越清晰，预览聊天越稳定。</p>
              </div>
              <label class="toggle-chip">
                <input :checked="isFrozen('lorebook')" type="checkbox" @change="toggleFrozen('lorebook', $event)">
                <span>钉住世界书区</span>
              </label>
            </div>
            <LorebookTreeEditor
              :entries="localDocument.lorebook.entries"
              :importing="pendingState.importingWorldbook"
              @update:entries="localDocument.lorebook.entries = $event"
              @import-worldbook="$emit('import-worldbook', $event)"
            />
          </section>
        </section>

        <section v-else-if="activeTab === 'scripts'" class="panel-stack">
          <section class="workspace-card">
            <div class="card-head">
              <div>
                <h3>脚本与任务</h3>
                <p>把运行时逻辑拆成两个子区：正则脚本负责输入输出变换，状态任务负责变量初始化与节奏控制。</p>
              </div>
              <label class="toggle-chip">
                <input :checked="isFrozen('regex') || isFrozen('state-tasks')" type="checkbox" @change="toggleScriptFreeze($event)">
                <span>统一钉住脚本区</span>
              </label>
            </div>

            <StudioSectionTabs
              :model-value="activeScriptTab"
              :items="scriptTabItems"
              @update:model-value="$emit('update:activeScriptTab', normalizeScriptTab($event))"
            />

            <div class="script-panel">
              <RegexWorkbench
                v-if="activeScriptTab === 'regex'"
                :scripts="localDocument.regexScripts"
                :generating="isGenerating('regex')"
                @generate="$emit('generate', 'regex')"
                @add="addRegexScript"
                @remove="removeRegexScript"
                @update:field="updateRegexField"
                @update:placement="updatePlacement"
                @toggle:field="toggleRegexField"
              />

              <TaskWorkbench
                v-else
                :tasks="localDocument.stateTasks"
                :generating="isGenerating('state-tasks')"
                @generate="$emit('generate', 'state-tasks')"
                @add="addStateTask"
                @remove="removeStateTask"
                @update:field="updateTaskField"
                @update:number="updateTaskNumber"
                @toggle:field="toggleTaskField"
              />
            </div>
          </section>
        </section>

        <section v-else class="panel-stack">
          <section class="workspace-card">
            <div class="card-head">
              <div>
                <h3>导出与诊断</h3>
                <p>在这里完成结构诊断、兼容裁剪确认和最终导出。导出前建议先跑一遍诊断。</p>
              </div>
              <div class="head-actions">
                <button class="ghost-btn small" :disabled="pendingState.validating" @click="$emit('validate')">
                  {{ pendingState.validating ? '诊断中...' : '重新诊断' }}
                </button>
                <button class="primary-btn small" :disabled="pendingState.saving" @click="$emit('save')">
                  {{ pendingState.saving ? '保存中...' : '保存文档' }}
                </button>
              </div>
            </div>

            <DiagnosticsPanel :diagnostics="diagnostics" />

            <div class="export-grid">
              <button class="export-card" :disabled="isDownloading('v3')" @click="$emit('download', 'v3')">
                <span class="export-icon">🧾</span>
                <strong>{{ isDownloading('v3') ? '导出中...' : '导出 V3 JSON' }}</strong>
                <p>当前工作台的主导出格式。</p>
              </button>
              <button class="export-card" :disabled="isDownloading('v2')" @click="$emit('download', 'v2')">
                <span class="export-icon">📦</span>
                <strong>{{ isDownloading('v2') ? '导出中...' : '导出 V2 JSON' }}</strong>
                <p>用于兼容旧生态，可能存在裁剪。</p>
              </button>
              <button class="export-card" :disabled="isDownloading('png')" @click="$emit('download', 'png')">
                <span class="export-icon">🖼️</span>
                <strong>{{ isDownloading('png') ? '导出中...' : '导出 PNG' }}</strong>
                <p>便于分享和回流导入。</p>
              </button>
              <button class="export-card" :disabled="isDownloading('worldbook')" @click="$emit('download', 'worldbook')">
                <span class="export-icon">📚</span>
                <strong>{{ isDownloading('worldbook') ? '导出中...' : '导出世界书' }}</strong>
                <p>单独导出当前角色知识树。</p>
              </button>
            </div>
          </section>
        </section>
      </section>
    </template>
  </main>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import type {
  CharacterStudioDocument,
  CharacterStudioEditorPendingState,
  ExportDiagnostic,
  RegexScript,
  StateTask,
} from '@/types/characterStudio'
import LorebookTreeEditor from './LorebookTreeEditor.vue'
import StudioSectionTabs from './StudioSectionTabs.vue'
import GreetingWorkbench from './GreetingWorkbench.vue'
import RegexWorkbench from './RegexWorkbench.vue'
import TaskWorkbench from './TaskWorkbench.vue'
import DiagnosticsPanel from './DiagnosticsPanel.vue'

const props = defineProps<{
  document: CharacterStudioDocument | null
  avatarUrl: string
  diagnostics: ExportDiagnostic | null
  pendingState: CharacterStudioEditorPendingState
  activeTab: 'overview' | 'character' | 'greetings' | 'lorebook' | 'scripts' | 'export'
  activeScriptTab: 'regex' | 'tasks'
}>()

const emit = defineEmits<{
  (e: 'update:document', value: CharacterStudioDocument | null): void
  (e: 'update:activeTab', value: 'overview' | 'character' | 'greetings' | 'lorebook' | 'scripts' | 'export'): void
  (e: 'update:activeScriptTab', value: 'regex' | 'tasks'): void
  (e: 'save'): void
  (e: 'generate', section: string): void
  (e: 'validate'): void
  (e: 'delete'): void
  (e: 'import-worldbook', file: File): void
  (e: 'download', format: string): void
}>()

const localDocument = ref<CharacterStudioDocument | null>(null)
let syncing = false

const tabItems = [
  { value: 'overview', label: '概览', icon: '◈' },
  { value: 'character', label: '角色设定', icon: '🧬' },
  { value: 'greetings', label: '问候语', icon: '💬' },
  { value: 'lorebook', label: '世界书', icon: '📚' },
  { value: 'scripts', label: '脚本任务', icon: '⚙️' },
  { value: 'export', label: '导出诊断', icon: '⬇' },
] as const

const scriptTabItems = [
  { value: 'regex', label: '正则脚本', icon: '∑' },
  { value: 'tasks', label: '状态任务', icon: '∞' },
] as const

const freezeItems = [
  { key: 'identity', label: '角色设定' },
  { key: 'greetings', label: '问候语' },
  { key: 'lorebook', label: '世界书' },
  { key: 'regex', label: '正则脚本' },
  { key: 'state-tasks', label: '状态任务' },
] as const

const flattenedLorebookCount = computed(() => {
  if (!localDocument.value) return 0
  const walk = (entries: CharacterStudioDocument['lorebook']['entries']): number =>
    entries.reduce((total, entry) => total + 1 + walk(entry.children || []), 0)
  return walk(localDocument.value.lorebook.entries)
})

const latestReview = computed(() => {
  const review = localDocument.value?.exportArtifacts?.last_review as
    | { summary?: string; issues?: string[]; suggestions?: string[] }
    | undefined
  if (!review || !review.summary) return null
  return {
    summary: review.summary,
    issues: Array.isArray(review.issues) ? review.issues : [],
    suggestions: Array.isArray(review.suggestions) ? review.suggestions : [],
  }
})

watch(() => props.document, value => {
  syncing = true
  localDocument.value = value ? JSON.parse(JSON.stringify(value)) as CharacterStudioDocument : null
  void nextTick(() => {
    syncing = false
  })
}, { immediate: true, deep: true })

watch(localDocument, value => {
  if (syncing) return
  emit('update:document', value ? JSON.parse(JSON.stringify(value)) as CharacterStudioDocument : null)
}, { deep: true })

function normalizeTab(value: string): 'overview' | 'character' | 'greetings' | 'lorebook' | 'scripts' | 'export' {
  return tabItems.some(item => item.value === value) ? value as typeof props.activeTab : 'overview'
}

function normalizeScriptTab(value: string): 'regex' | 'tasks' {
  return value === 'tasks' ? 'tasks' : 'regex'
}

function isGenerating(section: string) {
  return props.pendingState.generatingSection === section
}

const isGenerationLocked = computed(() => props.pendingState.generatingSection !== null)

function isDownloading(format: string) {
  return props.pendingState.downloadingFormat === format
}

function formatOrigin(origin: CharacterStudioDocument['origin']['type']) {
  if (origin === 'analysis') return '分析生成'
  if (origin === 'imported') return '外部导入'
  return '手工创建'
}

function updateAliases(event: Event) {
  if (!localDocument.value) return
  const target = event.target as HTMLInputElement
  localDocument.value.identity.aliases = target.value.split(/[,，]/).map(item => item.trim()).filter(Boolean)
}

function updateTags(event: Event) {
  if (!localDocument.value) return
  const target = event.target as HTMLInputElement
  localDocument.value.meta.tags = target.value.split(/[,，]/).map(item => item.trim()).filter(Boolean)
}

function addGreeting() {
  localDocument.value?.coreMessages.alternate_greetings.push('')
}

function updateGreeting(index: number, value: string) {
  if (!localDocument.value) return
  localDocument.value.coreMessages.alternate_greetings[index] = value
}

function removeGreeting(index: number) {
  localDocument.value?.coreMessages.alternate_greetings.splice(index, 1)
}

function moveGreeting(index: number, direction: -1 | 1) {
  if (!localDocument.value) return
  const target = index + direction
  const list = localDocument.value.coreMessages.alternate_greetings
  if (target < 0 || target >= list.length) return
  const [item] = list.splice(index, 1)
  list.splice(target, 0, item!)
}

function useAsPrimary(greeting: string) {
  if (!localDocument.value) return
  localDocument.value.coreMessages.first_message = greeting
}

function addRegexScript() {
  localDocument.value?.regexScripts.push({
    id: `regex_${Date.now()}`,
    scriptName: '新脚本',
    findRegex: '',
    replaceString: '',
    placement: [2],
    markdownOnly: false,
    promptOnly: false,
    runOnEdit: true,
    disabled: false,
  })
}

function removeRegexScript(index: number) {
  localDocument.value?.regexScripts.splice(index, 1)
}

function updateRegexField(index: number, field: keyof RegexScript, value: string) {
  if (!localDocument.value) return
  ;(localDocument.value.regexScripts[index] as unknown as Record<string, unknown>)[field] = value
}

function updatePlacement(index: number, rawValue: string) {
  if (!localDocument.value) return
  localDocument.value.regexScripts[index]!.placement = rawValue
    .split(/[,，]/)
    .map(item => Number(item.trim()))
    .filter(item => !Number.isNaN(item))
}

function toggleRegexField(index: number, field: keyof RegexScript, value: boolean) {
  if (!localDocument.value) return
  ;(localDocument.value.regexScripts[index] as unknown as Record<string, unknown>)[field] = value
}

function addStateTask() {
  localDocument.value?.stateTasks.push({
    id: `task_${Date.now()}`,
    name: '新任务',
    triggerTiming: 'initialization',
    interval: 0,
    commands: '<<taskjs>>\n\n<</taskjs>>',
    disabled: false,
  })
}

function removeStateTask(index: number) {
  localDocument.value?.stateTasks.splice(index, 1)
}

function updateTaskField(index: number, field: keyof StateTask, value: string) {
  if (!localDocument.value) return
  ;(localDocument.value.stateTasks[index] as unknown as Record<string, unknown>)[field] = value
}

function updateTaskNumber(index: number, field: keyof StateTask, value: number) {
  if (!localDocument.value) return
  ;(localDocument.value.stateTasks[index] as unknown as Record<string, unknown>)[field] = value
}

function toggleTaskField(index: number, field: keyof StateTask, value: boolean) {
  if (!localDocument.value) return
  ;(localDocument.value.stateTasks[index] as unknown as Record<string, unknown>)[field] = value
}

function isFrozen(section: string) {
  return !!localDocument.value?.status.frozen_sections.includes(section)
}

function toggleFrozen(section: string, event: Event) {
  if (!localDocument.value) return
  const target = event.target as HTMLInputElement
  const next = new Set(localDocument.value.status.frozen_sections || [])
  if (target.checked) next.add(section)
  else next.delete(section)
  localDocument.value.status.frozen_sections = [...next]
}

function toggleScriptFreeze(event: Event) {
  const target = event.target as HTMLInputElement
  const nextValue = target.checked
  const synthetic = { target: { checked: nextValue } } as unknown as Event
  toggleFrozen('regex', synthetic)
  toggleFrozen('state-tasks', synthetic)
}
</script>

<style scoped>
.studio-editor {
  display: flex;
  flex-direction: column;
  gap: 18px;
  height: 100%;
  min-width: 0;
  min-height: 0;
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 4px;
  scrollbar-gutter: stable;
}

.empty-card,
.overview-hero,
.editor-shell {
  border-radius: 28px;
  background: rgba(255, 255, 255, 0.88);
  border: 1px solid rgba(28, 55, 94, 0.08);
  box-shadow: 0 26px 42px rgba(20, 46, 82, 0.08);
}

.empty-card {
  padding: 36px;
  min-height: 320px;
}

.empty-mark {
  display: inline-flex;
  align-items: center;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(37, 99, 199, 0.1);
  color: #1f5fc3;
  font-size: 12px;
  font-weight: 600;
}

.empty-card h2 {
  margin: 18px 0 0;
  color: #102741;
  font-size: 30px;
}

.empty-card p {
  max-width: 560px;
  margin: 12px 0 0;
  color: #607794;
  line-height: 1.8;
}

.empty-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  margin-top: 22px;
}

.empty-tip {
  border-radius: 18px;
  padding: 16px;
  background: rgba(245, 249, 254, 0.92);
  border: 1px solid rgba(28, 55, 94, 0.08);
}

.empty-tip strong {
  display: block;
  color: #14304c;
}

.empty-tip p {
  margin: 8px 0 0;
  font-size: 13px;
  line-height: 1.7;
}

.overview-hero {
  padding: 22px;
  display: flex;
  justify-content: space-between;
  gap: 18px;
}

.hero-main {
  display: flex;
  gap: 18px;
  min-width: 0;
}

.avatar-shell {
  width: 116px;
  height: 164px;
  border-radius: 24px;
  overflow: hidden;
  background: linear-gradient(180deg, rgba(37, 99, 199, 0.14), rgba(17, 39, 65, 0.04));
  flex-shrink: 0;
}

.avatar-shell img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.avatar-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #1f5fc3;
  font-size: 32px;
  font-weight: 700;
}

.hero-copy {
  min-width: 0;
}

.hero-kicker {
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #6f84a2;
}

.hero-copy h2 {
  margin: 10px 0 0;
  color: #102741;
  font-size: 30px;
}

.hero-copy p {
  margin: 12px 0 0;
  color: #607794;
  line-height: 1.8;
}

.hero-meta {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 14px;
}

.meta-pill {
  border-radius: 999px;
  padding: 5px 10px;
  background: rgba(20, 56, 106, 0.07);
  color: #234977;
  font-size: 11px;
}

.hero-actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  align-content: flex-start;
  justify-content: flex-end;
}

.editor-shell {
  padding: 18px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.panel-stack {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.summary-grid,
.workspace-row,
.quick-grid,
.export-grid,
.form-grid {
  display: grid;
  gap: 14px;
}

.summary-grid {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.workspace-row {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.workspace-row.single {
  grid-template-columns: 1fr;
}

.quick-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin-top: 16px;
}

.export-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin-top: 16px;
}

.form-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.workspace-card,
.summary-card,
.quick-card,
.export-card {
  border-radius: 22px;
  border: 1px solid rgba(28, 55, 94, 0.08);
  background: rgba(255, 255, 255, 0.82);
}

.workspace-card,
.summary-card {
  padding: 18px;
}

.summary-card .summary-label {
  display: block;
  color: #6f84a2;
  font-size: 12px;
}

.summary-card strong {
  display: block;
  margin-top: 8px;
  font-size: 24px;
  color: #153250;
}

.summary-card p {
  margin: 8px 0 0;
  color: #607794;
  font-size: 13px;
  line-height: 1.6;
}

.card-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}

.card-head h3 {
  margin: 0;
  font-size: 18px;
  color: #112b47;
}

.card-head p {
  margin: 6px 0 0;
  color: #607794;
  font-size: 13px;
  line-height: 1.7;
}

.head-actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.quick-card,
.export-card {
  text-align: left;
  padding: 16px;
  cursor: pointer;
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}

.quick-card:hover,
.export-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 18px 28px rgba(20, 46, 82, 0.08);
  border-color: rgba(37, 99, 199, 0.16);
}

.quick-icon,
.export-icon {
  display: inline-flex;
  width: 36px;
  height: 36px;
  border-radius: 14px;
  align-items: center;
  justify-content: center;
  background: rgba(37, 99, 199, 0.1);
  color: #1f5fc3;
  font-size: 16px;
}

.quick-card strong,
.export-card strong {
  display: block;
  margin-top: 14px;
  color: #14304c;
  font-size: 14px;
}

.quick-card p,
.export-card p {
  margin: 8px 0 0;
  color: #607794;
  font-size: 13px;
  line-height: 1.6;
}

label {
  display: flex;
  flex-direction: column;
  gap: 6px;
  color: #516882;
  font-size: 12px;
}

.full {
  grid-column: 1 / -1;
}

input,
textarea,
select {
  border: 1px solid rgba(28, 55, 94, 0.12);
  background: rgba(245, 249, 254, 0.92);
  border-radius: 16px;
  padding: 12px 14px;
  color: #183351;
  font-size: 13px;
}

textarea {
  resize: vertical;
  line-height: 1.7;
}

.option-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 4px;
}

.toggle-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 9px 12px;
  border-radius: 999px;
  background: rgba(20, 56, 106, 0.06);
  color: #234977;
}

.freeze-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
  margin-top: 16px;
}

.freeze-item {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 16px;
  padding: 12px 14px;
  border-radius: 16px;
  background: rgba(245, 249, 254, 0.92);
  border: 1px solid rgba(28, 55, 94, 0.08);
}

.freeze-item-label {
  color: #183351;
  font-size: 14px;
  line-height: 1.5;
}

.freeze-item-control {
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.freeze-item-control input {
  width: 18px;
  height: 18px;
  margin: 0;
}

.ghost-btn,
.primary-btn,
.danger-btn {
  border: none;
  border-radius: 14px;
  cursor: pointer;
}

.ghost-btn {
  padding: 11px 14px;
  background: rgba(20, 56, 106, 0.07);
  color: #234977;
}

.primary-btn {
  padding: 11px 16px;
  background: linear-gradient(135deg, #2563c7, #4d86ee);
  color: #fff;
  box-shadow: 0 12px 24px rgba(37, 99, 199, 0.18);
}

.danger-btn {
  padding: 11px 14px;
  background: rgba(217, 55, 55, 0.12);
  color: #b83535;
}

.ghost-btn:disabled,
.primary-btn:disabled,
.danger-btn:disabled,
.export-card:disabled {
  opacity: 0.68;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.small {
  padding: 8px 12px;
  font-size: 12px;
}

.script-panel {
  margin-top: 16px;
}

.review-summary {
  margin-top: 14px;
  border-radius: 18px;
  padding: 16px;
  background: rgba(245, 249, 254, 0.92);
  border: 1px solid rgba(28, 55, 94, 0.08);
}

.review-summary strong {
  display: block;
  color: #14304c;
  font-size: 15px;
  line-height: 1.7;
}

.review-list {
  margin: 12px 0 0;
  padding-left: 18px;
  color: #516882;
  font-size: 13px;
  line-height: 1.7;
}

.review-list.suggestions {
  color: #2b5f9f;
}

@media (max-width: 1180px) {
  .summary-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 900px) {
  .overview-hero,
  .hero-main,
  .hero-actions,
  .workspace-row,
  .quick-grid,
  .export-grid,
  .summary-grid,
  .empty-grid,
  .form-grid,
  .card-head {
    grid-template-columns: 1fr;
    flex-direction: column;
  }
}
</style>
