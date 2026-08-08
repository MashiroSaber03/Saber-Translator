<template>
  <main class="studio-editor">
    <section v-if="!localDocument" class="studio-editor__onboarding">
      <ProductEmptyState
        class="studio-editor__onboarding-empty-state"
        eyebrow="角色工坊"
        icon-name="users"
        role="note"
        title="选择或创建角色文档"
        description="先从左侧候选锁定角色名，或直接新建空白文档。创建后建议先使用 AI 一键补全整卡，再进入各分区精修。"
      />
      <div class="studio-editor__onboarding-tip-grid">
        <ProductRecordCard class="studio-editor__onboarding-tip-card">
          <strong class="studio-editor__onboarding-tip-title">从候选开始</strong>
          <p class="studio-editor__onboarding-tip-description">如果当前书已有增强时间线，可以先从分析候选创建角色。候选只会预填角色名，后续由压缩摘要驱动的 AI 负责补完整张卡。</p>
        </ProductRecordCard>
        <ProductRecordCard class="studio-editor__onboarding-tip-card">
          <strong class="studio-editor__onboarding-tip-title">空白新建或导入</strong>
          <p class="studio-editor__onboarding-tip-description">如果你想手工构建角色，或者直接导入外部角色卡，也可以从左侧工具区快速开始；空白卡同样能用 AI 一键补全。</p>
        </ProductRecordCard>
      </div>
    </section>

    <template v-else>
      <StudioHeroSection
        :avatar-url="avatarUrl"
        :document="localDocument"
        :format-origin="formatOrigin"
        :is-generation-locked="isGenerationLocked"
        :is-generating="isGenerating"
        :pending-state="pendingState"
        @delete="$emit('delete')"
        @generate="$emit('generate', $event)"
      />

      <section class="studio-editor__shell">
        <ProductSegmentedTabs
          :active-tab="activeTab"
          aria-label="角色工坊编辑分区"
          layout="scroll"
          :tabs="tabItems"
          @update:active-tab="$emit('update:activeTab', normalizeTab($event))"
        >
          <template #tabIcon="{ tab }">{{ editorTabGlyph(tab.id) }}</template>
        </ProductSegmentedTabs>

        <StudioOverviewTab
          v-if="activeTab === 'overview'"
          :diagnostics="diagnostics"
          :document="localDocument"
          :flattened-lorebook-count="flattenedLorebookCount"
          :format-origin="formatOrigin"
          :freeze-items="freezeItems"
          :is-frozen="isFrozen"
          :latest-review="latestReview"
          :pending-state="pendingState"
          @tab="$emit('update:activeTab', $event)"
          @toggle-frozen="toggleFrozen"
          @validate="$emit('validate')"
        />

        <section v-else-if="activeTab === 'character'" class="studio-editor__panel-stack">
          <StudioEditorSectionPanel
            title="角色设定"
            description="聚合角色身份与世界观上下文，优先把角色基底写清楚，再去扩展运行时能力。"
          >
            <template #actions>
              <ProductActionRow appearance="accent" aria-label="角色设定生成操作" justify="start">
                <UiButton variant="secondary" :disabled="isGenerationLocked" @click="$emit('generate', 'identity')" size="sm">
                  {{ isGenerating('identity') ? '重写中...' : 'AI 重写本区' }}
                </UiButton>
                <UiButton variant="secondary" :disabled="isGenerationLocked" @click="$emit('generate', 'translate')" size="sm">
                  {{ isGenerating('translate') ? '翻译中...' : '整卡翻译' }}
                </UiButton>
              </ProductActionRow>
            </template>

            <UiFormGrid class="studio-editor__form-grid">
              <UiField variant="settings" label="角色名称" control-id="studioCharacterName">
                <UiInput id="studioCharacterName" v-model="localDocument.identity.name" type="text" variant="studio" size="lg" />
              </UiField>
              <UiField variant="settings" label="别名（逗号分隔）" control-id="studioCharacterAliases">
                <UiInput
                  id="studioCharacterAliases"
                  :model-value="localDocument.identity.aliases.join(', ')"
                  type="text"
                  variant="studio"
                  size="lg"
                  @update:model-value="updateAliases(String($event))"
                />
              </UiField>
              <UiField class="studio-editor__field--full" variant="settings" label="角色简介" control-id="studioCharacterDescription">
                <UiTextarea id="studioCharacterDescription" v-model="localDocument.identity.description" rows="6" variant="studio" size="lg" />
              </UiField>
              <UiField class="studio-editor__field--full" variant="settings" label="性格 / 人设" control-id="studioCharacterPersonality">
                <UiTextarea id="studioCharacterPersonality" v-model="localDocument.identity.personality" rows="5" variant="studio" size="lg" />
              </UiField>
              <UiField class="studio-editor__field--full" variant="settings" label="当前场景" control-id="studioCharacterScenario">
                <UiTextarea id="studioCharacterScenario" v-model="localDocument.identity.scenario" rows="5" variant="studio" size="lg" />
              </UiField>
              <UiField class="studio-editor__field--full" variant="settings" label="标签（逗号分隔）" control-id="studioCharacterTags">
                <UiInput
                  id="studioCharacterTags"
                  :model-value="localDocument.meta.tags.join(', ')"
                  type="text"
                  variant="studio"
                  size="lg"
                  @update:model-value="updateTags(String($event))"
                />
              </UiField>
              <div class="studio-editor__field--full studio-editor__option-row">
                <UiCheckbox v-model="localDocument.status.is_favorite" class="studio-editor__toggle-chip" label="收藏当前角色" />
                <UiCheckbox :model-value="isFrozen('identity')" class="studio-editor__toggle-chip" label="钉住角色设定" @change="toggleFrozen('identity', $event)" />
              </div>
            </UiFormGrid>
          </StudioEditorSectionPanel>
        </section>

        <section v-else-if="activeTab === 'greetings'" class="studio-editor__panel-stack">
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

          <StudioEditorSectionPanel
            title="对话元信息"
            description="这里的系统提示词和示例对话会影响预览聊天和导出卡片的整体语气。"
          >
            <UiFormGrid class="studio-editor__form-grid">
              <UiField class="studio-editor__field--full" variant="settings" label="示例对话" control-id="studioMessageExample">
                <UiTextarea id="studioMessageExample" v-model="localDocument.coreMessages.message_example" rows="5" variant="studio" size="lg" />
              </UiField>
              <UiField class="studio-editor__field--full" variant="settings" label="系统提示词（System Prompt）" control-id="studioSystemPrompt">
                <UiTextarea id="studioSystemPrompt" v-model="localDocument.coreMessages.system_prompt" rows="4" variant="studio" size="lg" />
              </UiField>
              <UiField class="studio-editor__field--full" variant="settings" label="历史后置说明（Post History）" control-id="studioPostHistoryInstructions">
                <UiTextarea id="studioPostHistoryInstructions" v-model="localDocument.coreMessages.post_history_instructions" rows="3" variant="studio" size="lg" />
              </UiField>
              <UiField class="studio-editor__field--full" variant="settings" label="备注" control-id="studioCreatorNotes">
                <UiTextarea id="studioCreatorNotes" v-model="localDocument.coreMessages.creator_notes" rows="3" variant="studio" size="lg" />
              </UiField>
              <UiField variant="settings" label="角色版本" control-id="studioCharacterVersion">
                <UiInput id="studioCharacterVersion" v-model="localDocument.coreMessages.character_version" type="text" variant="studio" size="lg" />
              </UiField>
              <div class="studio-editor__option-row">
                <UiCheckbox :model-value="isFrozen('greetings')" class="studio-editor__toggle-chip" label="钉住问候语区" @change="toggleFrozen('greetings', $event)" />
              </div>
            </UiFormGrid>
          </StudioEditorSectionPanel>
        </section>

        <section v-else-if="activeTab === 'lorebook'" class="studio-editor__panel-stack">
          <StudioEditorSectionPanel
            title="世界书"
            description="把角色设定、关系、场景、专有名词沉淀成可命中的知识树。条目设计越清晰，预览聊天越稳定。"
          >
            <template #actions>
              <UiCheckbox :model-value="isFrozen('lorebook')" class="studio-editor__toggle-chip" label="钉住世界书区" @change="toggleFrozen('lorebook', $event)" />
            </template>
            <LorebookTreeEditor
              :entries="localDocument.lorebook.entries"
              :importing="pendingState.importingWorldbook"
              @update:entries="localDocument.lorebook.entries = $event"
              @import-worldbook="$emit('import-worldbook', $event)"
            />
          </StudioEditorSectionPanel>
        </section>

        <section v-else-if="activeTab === 'scripts'" class="studio-editor__panel-stack">
          <StudioEditorSectionPanel
            title="脚本与任务"
            description="把运行时逻辑拆成两个子区：正则脚本负责输入输出变换，状态任务负责变量初始化与节奏控制。"
          >
            <template #actions>
              <UiCheckbox :model-value="isFrozen('regex') || isFrozen('state-tasks')" class="studio-editor__toggle-chip" label="统一钉住脚本区" @change="toggleScriptFreeze" />
            </template>

            <ProductSegmentedTabs
              :active-tab="activeScriptTab"
              aria-label="脚本与任务分区"
              layout="scroll"
              :tabs="scriptTabs"
              @update:active-tab="$emit('update:activeScriptTab', normalizeScriptTab($event))"
            >
              <template #tabIcon="{ tab }">{{ scriptTabGlyph(tab.id) }}</template>
            </ProductSegmentedTabs>

            <div class="studio-editor__script-panel">
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
          </StudioEditorSectionPanel>
        </section>

        <section v-else class="studio-editor__panel-stack">
          <StudioEditorSectionPanel
            title="导出与诊断"
            description="在这里完成结构诊断、上下文裁剪确认和最终导出。导出前建议先跑一遍诊断。"
          >
            <template #actions>
              <ProductActionRow appearance="accent" aria-label="导出与诊断操作" justify="start">
                <UiButton variant="secondary" :disabled="pendingState.validating" @click="$emit('validate')" size="sm">
                  {{ pendingState.validating ? '诊断中...' : '重新诊断' }}
                </UiButton>
                <UiButton variant="primary" :disabled="pendingState.saving" @click="$emit('save')" size="sm">
                  {{ pendingState.saving ? '保存中...' : '保存文档' }}
                </UiButton>
              </ProductActionRow>
            </template>

            <DiagnosticsPanel :diagnostics="diagnostics" />

            <div class="studio-editor__download-grid">
              <ProductRecordCard
                v-for="item in exportDownloadItems"
                :key="item.format"
                as="button"
                class="studio-editor__download-card"
                :aria-label="item.label"
                :disabled="isDownloading(item.format)"
                @click="$emit('download', item.format)"
              >
                <template #icon>
                  <span class="studio-editor__download-icon">
                    <UiIcon :name="item.iconName" size="18" />
                  </span>
                </template>
                <strong class="studio-editor__download-title">{{ isDownloading(item.format) ? item.pendingLabel : item.label }}</strong>
                <p class="studio-editor__download-description">{{ item.description }}</p>
              </ProductRecordCard>
            </div>
          </StudioEditorSectionPanel>
        </section>
      </section>
    </template>
  </main>
</template>

<script setup lang="ts">

import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'

import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductSegmentedTabs from '@/components/product/ProductSegmentedTabs.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import type { UiIconName } from '@/components/ui/iconRegistry'
import { computed, nextTick, ref, watch } from 'vue'
import type {
  CharacterStudioDocument,
  CharacterStudioEditorPendingState,
  ExportDiagnostic,
} from '@/types/characterStudio'
import { deepClone } from '@/utils/deepClone'
import LorebookTreeEditor from './LorebookTreeEditor.vue'
import GreetingWorkbench from './GreetingWorkbench.vue'
import RegexWorkbench from './RegexWorkbench.vue'
import TaskWorkbench from './TaskWorkbench.vue'
import DiagnosticsPanel from './DiagnosticsPanel.vue'
import { editorTabItems, freezeItems, scriptTabItems } from './characterStudioEditorConfig'
import StudioHeroSection from './editor/StudioHeroSection.vue'
import StudioEditorSectionPanel from './editor/StudioEditorSectionPanel.vue'
import StudioOverviewTab from './editor/StudioOverviewTab.vue'

type RegexTextField = 'scriptName' | 'findRegex' | 'replaceString'
type RegexToggleField = 'markdownOnly' | 'promptOnly' | 'runOnEdit' | 'disabled'
type StateTaskTextField = 'name' | 'triggerTiming' | 'commands'
type StateTaskNumberField = 'interval'
type StateTaskToggleField = 'disabled'
type ExportDownloadFormat = 'v3' | 'v2' | 'png' | 'worldbook'

type ExportDownloadItem = {
  format: ExportDownloadFormat
  label: string
  pendingLabel: string
  description: string
  iconName: UiIconName
}

const exportDownloadItems: readonly ExportDownloadItem[] = [
  {
    format: 'v3',
    label: '导出 V3 JSON',
    pendingLabel: '导出中...',
    description: '当前工作台的主导出格式。',
    iconName: 'file-text',
  },
  {
    format: 'v2',
    label: '导出 V2 JSON',
    pendingLabel: '导出中...',
    description: '部分平台可能存在裁剪。',
    iconName: 'download',
  },
  {
    format: 'png',
    label: '导出 PNG',
    pendingLabel: '导出中...',
    description: '便于分享和回流导入。',
    iconName: 'image',
  },
  {
    format: 'worldbook',
    label: '导出世界书',
    pendingLabel: '导出中...',
    description: '单独导出当前角色知识树。',
    iconName: 'book-open',
  },
]

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

const tabItems = editorTabItems.map(item => ({
  id: item.value,
  label: item.label,
  iconName: item.iconName,
}))
const scriptTabs = scriptTabItems.map(item => ({
  id: item.value,
  label: item.label,
  iconName: item.iconName,
}))
const editorTabGlyphs = new Map(editorTabItems.map(item => [item.value, item.glyph]))
const scriptTabGlyphs = new Map(scriptTabItems.map(item => [item.value, item.glyph]))

function editorTabGlyph(tabId: string): string {
  return editorTabGlyphs.get(tabId as typeof editorTabItems[number]['value']) ?? ''
}

function scriptTabGlyph(tabId: string): string {
  return scriptTabGlyphs.get(tabId as typeof scriptTabItems[number]['value']) ?? ''
}

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
  localDocument.value = value ? deepClone(value) : null
  void nextTick(() => {
    syncing = false
  })
}, { immediate: true, deep: true })

watch(localDocument, value => {
  if (syncing) return
  if (value) {
    const normalizedName = String(value.identity.name || '').trim()
    if (value.meta.title !== normalizedName) {
      value.meta.title = normalizedName
    }
  }
  emit('update:document', value ? deepClone(value) : null)
}, { deep: true })

watch(() => localDocument.value?.identity.name, value => {
  if (!localDocument.value) return
  const normalizedName = String(value || '').trim()
  if (localDocument.value.meta.title === normalizedName) return
  localDocument.value.meta.title = normalizedName
}, { flush: 'sync' })

function normalizeTab(value: string): 'overview' | 'character' | 'greetings' | 'lorebook' | 'scripts' | 'export' {
  return tabItems.some(item => item.id === value) ? value as typeof props.activeTab : 'overview'
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

function updateAliases(value: string) {
  if (!localDocument.value) return
  localDocument.value.identity.aliases = value.split(/[,，]/).map(item => item.trim()).filter(Boolean)
}

function updateTags(value: string) {
  if (!localDocument.value) return
  localDocument.value.meta.tags = value.split(/[,，]/).map(item => item.trim()).filter(Boolean)
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

function updateRegexField(index: number, field: RegexTextField, value: string) {
  if (!localDocument.value) return
  const script = localDocument.value.regexScripts[index]
  if (!script) return
  script[field] = value
}

function updatePlacement(index: number, rawValue: string) {
  if (!localDocument.value) return
  localDocument.value.regexScripts[index]!.placement = rawValue
    .split(/[,，]/)
    .map(item => Number(item.trim()))
    .filter(item => !Number.isNaN(item))
}

function toggleRegexField(index: number, field: RegexToggleField, value: boolean) {
  if (!localDocument.value) return
  const script = localDocument.value.regexScripts[index]
  if (!script) return
  script[field] = value
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

function updateTaskField(index: number, field: StateTaskTextField, value: string) {
  if (!localDocument.value) return
  const task = localDocument.value.stateTasks[index]
  if (!task) return
  task[field] = value
}

function updateTaskNumber(index: number, field: StateTaskNumberField, value: number) {
  if (!localDocument.value) return
  const task = localDocument.value.stateTasks[index]
  if (!task) return
  task[field] = value
}

function toggleTaskField(index: number, field: StateTaskToggleField, value: boolean) {
  if (!localDocument.value) return
  const task = localDocument.value.stateTasks[index]
  if (!task) return
  task[field] = value
}

function isFrozen(section: string) {
  return !!localDocument.value?.status.frozen_sections.includes(section)
}

function toggleFrozen(section: string, checked: boolean) {
  if (!localDocument.value) return
  const next = new Set(localDocument.value.status.frozen_sections || [])
  if (checked) next.add(section)
  else next.delete(section)
  localDocument.value.status.frozen_sections = [...next]
}

function toggleScriptFreeze(checked: boolean) {
  toggleFrozen('regex', checked)
  toggleFrozen('state-tasks', checked)
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

.studio-editor__onboarding,
.studio-editor__shell {
  border-radius: 28px;
  background: color-mix(in srgb, var(--color-surface-card) 88%, transparent);
  border: 1px solid var(--studio-border-default);
  box-shadow: 0 26px 42px var(--studio-shadow-floating);
}

.studio-editor__onboarding {
  min-height: 320px;
  padding: 36px;
}

.studio-editor__onboarding-empty-state {
  --product-empty-state-align-items: flex-start;
  --product-empty-state-justify-content: flex-start;
  --product-empty-state-max-width: 560px;
  --product-empty-state-min-height: 0;
  --product-empty-state-margin-inline: 0;
  --product-empty-state-padding: 0;
  --product-empty-state-text-align: left;
  --product-empty-state-icon-display: none;
  --product-empty-state-title-margin: 18px 0 0;
  --product-empty-state-title-font-size: 30px;
  --product-empty-state-title-font-weight: 700;
  --product-empty-state-title: var(--studio-text-strong);
  --product-empty-state-description-margin: 12px 0 0;
  --product-empty-state-description-font-size: 0.95rem;
  --product-empty-state-description-line-height: 1.8;
}

.studio-editor__onboarding-tip-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr));
  gap: 14px;
  margin-top: 24px;
}

.studio-editor__onboarding-tip-title {
  display: block;
  color: var(--studio-text-strong);
}

.studio-editor__onboarding-tip-card {
  --product-record-card-background: var(--color-surface-quiet);
  --product-record-card-border: var(--studio-border-default);
  --product-record-card-radius: 18px;
  --product-record-card-padding: 16px;
}

.studio-editor__onboarding-tip-description {
  margin: 8px 0 0;
  color: var(--studio-text-muted);
  font-size: 13px;
  line-height: 1.7;
}

.studio-editor__shell {
  padding: 18px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.studio-editor__panel-stack {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.studio-editor__download-grid,
.studio-editor__form-grid {
  display: grid;
  gap: 14px;
}

.studio-editor__download-grid {
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr));
  margin-top: 16px;
}

.studio-editor__form-grid {
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr));
}

.studio-editor__download-card {
  --product-record-card-background: color-mix(in srgb, var(--color-surface-card) 82%, transparent);
  --product-record-card-border: var(--studio-border-default);
  --product-record-card-accent: color-mix(in srgb, var(--color-action-brand) 16%, transparent);
  --product-record-card-radius: 22px;
  --product-record-card-padding: 16px;
  --product-record-card-gap: 12px;
  --product-record-card-shadow-hover: 0 18px 28px var(--studio-shadow-floating);
}

.studio-editor__download-icon {
  display: inline-flex;
  width: 36px;
  height: 36px;
  border-radius: 14px;
  align-items: center;
  justify-content: center;
  background: var(--studio-surface-tint);
  color: var(--color-text-link-strong);
  font-size: 16px;
}

.studio-editor__download-title {
  display: block;
  color: var(--studio-text-strong);
  font-size: 14px;
}

.studio-editor__download-description {
  margin: 0;
  color: var(--studio-text-muted);
  font-size: 13px;
  line-height: 1.6;
}

.studio-editor__field--full {
  grid-column: 1 / -1;
}

.studio-editor__option-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 4px;
}

.studio-editor__toggle-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 9px 12px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--color-action-brand) 6%, transparent);
  color: var(--studio-text-default);
}

.studio-editor__script-panel {
  margin-top: 16px;
}

@media (--breakpoint-lg-down) {
  .studio-editor__download-grid,
  .studio-editor__onboarding-tip-grid,
  .studio-editor__form-grid {
    grid-template-columns: 1fr;
  }
}
</style>
