<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiInput from '@/components/ui/UiInput.vue'
import DiagnosticsPanel from '../DiagnosticsPanel.vue'
import type { CharacterStudioDocument, CharacterStudioEditorPendingState, ExportDiagnostic } from '@/types/characterStudio'

type FreezeItem = {
  key: string
  label: string
}

type ReviewSummary = {
  summary: string
  issues: string[]
  suggestions: string[]
} | null

defineProps<{
  diagnostics: ExportDiagnostic | null
  document: CharacterStudioDocument
  flattenedLorebookCount: number
  formatOrigin: (origin: CharacterStudioDocument['origin']['type']) => string
  freezeItems: FreezeItem[]
  isFrozen: (section: string) => boolean
  latestReview: ReviewSummary
  pendingState: CharacterStudioEditorPendingState
}>()

defineEmits<{
  (event: 'tab', value: 'character' | 'greetings' | 'lorebook' | 'scripts'): void
  (event: 'toggleFrozen', section: string, value: Event): void
  (event: 'validate'): void
}>()
</script>

<template>
  <section class="panel-stack">
    <div class="summary-grid">
      <article class="summary-card">
        <span class="summary-label">来源摘要</span>
        <strong>{{ formatOrigin(document.origin.type) }}</strong>
        <p v-if="document.origin.source_character">通过分析候选「{{ document.origin.source_character }}」锁定角色名创建</p>
        <p v-else>当前文档为手工或外部导入角色。</p>
      </article>
      <article class="summary-card">
        <span class="summary-label">运行时资源</span>
        <strong>{{ document.regexScripts.length + document.stateTasks.length }}</strong>
        <p>{{ document.regexScripts.length }} 个脚本 · {{ document.stateTasks.length }} 个任务</p>
      </article>
      <article class="summary-card">
        <span class="summary-label">问候语库存</span>
        <strong>{{ document.coreMessages.alternate_greetings.length + (document.coreMessages.first_message ? 1 : 0) }}</strong>
        <p>1 条主问候 + {{ document.coreMessages.alternate_greetings.length }} 条备用问候</p>
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
          <UiButton variant="toolbar" class="quick-card" @click="$emit('tab', 'character')">
            <span class="quick-icon">🧬</span>
            <strong>角色设定</strong>
            <p>完善简介、性格、场景、标签。</p>
          </UiButton>
          <UiButton variant="toolbar" class="quick-card" @click="$emit('tab', 'greetings')">
            <span class="quick-icon">💬</span>
            <strong>问候语</strong>
            <p>打磨主问候和备用开场。</p>
          </UiButton>
          <UiButton variant="toolbar" class="quick-card" @click="$emit('tab', 'lorebook')">
            <span class="quick-icon">📚</span>
            <strong>世界书</strong>
            <p>维护角色知识树和触发条目。</p>
          </UiButton>
          <UiButton variant="toolbar" class="quick-card" @click="$emit('tab', 'scripts')">
            <span class="quick-icon">⚙️</span>
            <strong>脚本任务</strong>
            <p>配置正则脚本和状态任务。</p>
          </UiButton>
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
              <UiInput class="freeze-checkbox" :checked="isFrozen(item.key)" type="checkbox" @change="$emit('toggleFrozen', item.key, $event)" />
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
          <UiButton
            variant="toolbar"
            class="action-ghost"
            :disabled="pendingState.validating"
            size="sm"
            @click="$emit('validate')"
          >
            {{ pendingState.validating ? '诊断中...' : '重新诊断' }}
          </UiButton>
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
</template>

<style scoped>
.panel-stack {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.summary-grid,
.workspace-row,
.quick-grid {
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

.workspace-card,
.summary-card,
.quick-card {
  border: 1px solid var(--studio-border-default);
  border-radius: 22px;
  background: var(--character-studio-editor-surface-muted);
}

.workspace-card,
.summary-card {
  padding: 18px;
}

.summary-card .summary-label {
  display: block;
  color: var(--character-studio-editor-text-muted);
  font-size: 12px;
}

.summary-card strong {
  display: block;
  margin-top: 8px;
  color: var(--character-studio-editor-text-subtle);
  font-size: 24px;
}

.summary-card p {
  margin: 8px 0 0;
  color: var(--studio-text-muted);
  font-size: 13px;
  line-height: 1.6;
}

.card-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.card-head h3 {
  margin: 0;
  color: var(--character-studio-editor-text-supporting);
  font-size: 18px;
}

.card-head p {
  margin: 6px 0 0;
  color: var(--studio-text-muted);
  font-size: 13px;
  line-height: 1.7;
}

.quick-card {
  padding: 16px;
  text-align: left;
  cursor: pointer;
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}

.quick-card:hover {
  border-color: var(--character-studio-editor-border-default);
  transform: translateY(-2px);
  box-shadow: 0 18px 28px var(--studio-shadow-floating);
}

.quick-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: 14px;
  background: var(--studio-surface-tint);
  color: var(--color-text-primary-strong);
  font-size: 16px;
}

.quick-card strong {
  display: block;
  margin-top: 14px;
  color: var(--character-studio-editor-text-secondary);
  font-size: 14px;
}

.quick-card p {
  margin: 8px 0 0;
  color: var(--studio-text-muted);
  font-size: 13px;
  line-height: 1.6;
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
  border: 1px solid var(--studio-border-default);
  border-radius: 16px;
  background: var(--studio-surface-soft);
}

.freeze-item-label {
  color: var(--studio-text-strong);
  font-size: 14px;
  line-height: 1.5;
}

.freeze-item-control {
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.freeze-checkbox {
  width: 18px;
  height: 18px;
  margin: 0;
}

.action-ghost {
  padding: 11px 14px;
  border: none;
  border-radius: 14px;
  background: var(--studio-surface-muted);
  color: var(--studio-text-default);
  cursor: pointer;
}

.review-summary {
  margin-top: 14px;
  color: var(--studio-text-default);
  font-size: 13px;
  line-height: 1.7;
}

.review-list {
  margin: 10px 0 0;
  padding-left: 18px;
}

.review-list.suggestions {
  color: var(--character-studio-editor-text-inverse);
}

@media (--breakpoint-studio-down) {
  .summary-grid,
  .workspace-row {
    grid-template-columns: 1fr;
  }
}

@media (--breakpoint-preview-down) {
  .quick-grid {
    grid-template-columns: 1fr;
  }
}
</style>
