<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import DiagnosticsPanel from '../DiagnosticsPanel.vue'
import StudioEditorSectionPanel from './StudioEditorSectionPanel.vue'
import type {
  CharacterStudioDocument,
  CharacterStudioEditorPendingState,
  CharacterStudioSection,
  ExportDiagnostic,
} from '@/types/characterStudio'

type FreezeItem = {
  key: CharacterStudioSection
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
  freezeItems: readonly FreezeItem[]
  isFrozen: (section: CharacterStudioSection) => boolean
  latestReview: ReviewSummary
  pendingState: CharacterStudioEditorPendingState
}>()

defineEmits<{
  (event: 'tab', value: 'character' | 'greetings' | 'lorebook' | 'scripts'): void
  (event: 'toggleFrozen', section: CharacterStudioSection, value: boolean): void
  (event: 'validate'): void
}>()
</script>

<template>
  <section class="studio-overview-tab">
    <div class="studio-overview-tab__summary-grid">
      <ProductRecordCard class="studio-overview-tab__summary-card">
        <span class="studio-overview-tab__summary-label">来源摘要</span>
        <strong class="studio-overview-tab__summary-value">{{
          formatOrigin(document.origin.type)
        }}</strong>
        <p v-if="document.origin.source_character" class="studio-overview-tab__summary-description">
          通过分析候选「{{ document.origin.source_character }}」锁定角色名创建
        </p>
        <p v-else class="studio-overview-tab__summary-description">
          当前文档为手工或外部导入角色。
        </p>
      </ProductRecordCard>
      <ProductRecordCard class="studio-overview-tab__summary-card">
        <span class="studio-overview-tab__summary-label">运行时资源</span>
        <strong class="studio-overview-tab__summary-value">{{
          document.regexScripts.length + document.stateTasks.length
        }}</strong>
        <p class="studio-overview-tab__summary-description">
          {{ document.regexScripts.length }} 个脚本 · {{ document.stateTasks.length }} 个任务
        </p>
      </ProductRecordCard>
      <ProductRecordCard class="studio-overview-tab__summary-card">
        <span class="studio-overview-tab__summary-label">问候语库存</span>
        <strong class="studio-overview-tab__summary-value">{{
          document.coreMessages.alternate_greetings.length +
            (document.coreMessages.first_message ? 1 : 0)
        }}</strong>
        <p class="studio-overview-tab__summary-description">
          {{ document.coreMessages.first_message ? 1 : 0 }} 条主问候 +
          {{ document.coreMessages.alternate_greetings.length }} 条备用问候
        </p>
      </ProductRecordCard>
      <ProductRecordCard class="studio-overview-tab__summary-card">
        <span class="studio-overview-tab__summary-label">知识量</span>
        <strong class="studio-overview-tab__summary-value">{{ flattenedLorebookCount }}</strong>
        <p class="studio-overview-tab__summary-description">
          世界书树当前共有 {{ flattenedLorebookCount }} 个节点。
        </p>
      </ProductRecordCard>
    </div>

    <div class="studio-overview-tab__workspace-row">
      <StudioEditorSectionPanel title="快速入口" description="直接跳到你现在最可能继续编辑的模块。">
        <div class="studio-overview-tab__quick-grid">
          <ProductRecordCard
            as="button"
            class="studio-overview-tab__quick-card"
            aria-label="打开角色设定"
            @click="$emit('tab', 'character')"
          >
            <span class="studio-overview-tab__quick-icon">
              <UiIcon name="users" size="18" />
            </span>
            <strong class="studio-overview-tab__quick-title">角色设定</strong>
            <p class="studio-overview-tab__quick-description">完善简介、性格、场景、标签。</p>
          </ProductRecordCard>
          <ProductRecordCard
            as="button"
            class="studio-overview-tab__quick-card"
            aria-label="打开问候语"
            @click="$emit('tab', 'greetings')"
          >
            <span class="studio-overview-tab__quick-icon">
              <UiIcon name="message" size="18" />
            </span>
            <strong class="studio-overview-tab__quick-title">问候语</strong>
            <p class="studio-overview-tab__quick-description">打磨主问候和备用开场。</p>
          </ProductRecordCard>
          <ProductRecordCard
            as="button"
            class="studio-overview-tab__quick-card"
            aria-label="打开世界书"
            @click="$emit('tab', 'lorebook')"
          >
            <span class="studio-overview-tab__quick-icon">
              <UiIcon name="book-open" size="18" />
            </span>
            <strong class="studio-overview-tab__quick-title">世界书</strong>
            <p class="studio-overview-tab__quick-description">维护角色知识树和触发条目。</p>
          </ProductRecordCard>
          <ProductRecordCard
            as="button"
            class="studio-overview-tab__quick-card"
            aria-label="打开脚本任务"
            @click="$emit('tab', 'scripts')"
          >
            <span class="studio-overview-tab__quick-icon">
              <UiIcon name="settings" size="18" />
            </span>
            <strong class="studio-overview-tab__quick-title">脚本任务</strong>
            <p class="studio-overview-tab__quick-description">配置正则脚本和状态任务。</p>
          </ProductRecordCard>
        </div>
      </StudioEditorSectionPanel>

      <StudioEditorSectionPanel
        title="保护设置"
        description="被钉住的区块不会被 AI 再生成或 Agent patch 覆盖。"
      >
        <div class="studio-overview-tab__freeze-grid">
          <div v-for="item in freezeItems" :key="item.key" class="studio-overview-tab__freeze-item">
            <span class="studio-overview-tab__freeze-item-label">{{ item.label }}</span>
            <span class="studio-overview-tab__freeze-item-control">
              <UiCheckbox
                :input-id="`studio-freeze-${item.key}`"
                :model-value="isFrozen(item.key)"
                :aria-label="`钉住${item.label}`"
                @change="$emit('toggleFrozen', item.key, $event)"
              />
            </span>
          </div>
        </div>
      </StudioEditorSectionPanel>
    </div>

    <div class="studio-overview-tab__workspace-row studio-overview-tab__workspace-row--single">
      <StudioEditorSectionPanel
        title="最近诊断摘要"
        description="导出前先看这里，能快速判断当前角色是否存在结构性问题。"
      >
        <template #actions>
          <ProductActionRow appearance="accent" aria-label="诊断操作">
            <UiButton
              variant="secondary"
              :disabled="pendingState.validating"
              size="sm"
              @click="$emit('validate')"
            >
              {{ pendingState.validating ? '诊断中...' : '重新诊断' }}
            </UiButton>
          </ProductActionRow>
        </template>
        <DiagnosticsPanel :diagnostics="diagnostics" />
      </StudioEditorSectionPanel>
    </div>

    <div
      v-if="latestReview"
      class="studio-overview-tab__workspace-row studio-overview-tab__workspace-row--single"
    >
      <StudioEditorSectionPanel
        title="最近审查"
        description="这里展示最近一次“AI 审查当前角色”的结果，方便你直接据此继续补卡。"
      >
        <div class="studio-overview-tab__review-summary">
          <strong class="studio-overview-tab__review-title">{{ latestReview.summary }}</strong>
          <ul v-if="latestReview.issues.length > 0" class="studio-overview-tab__review-list">
            <li v-for="(item, index) in latestReview.issues" :key="`review-issue-${index}`">
              {{ item }}
            </li>
          </ul>
          <ul
            v-if="latestReview.suggestions.length > 0"
            class="studio-overview-tab__review-list studio-overview-tab__review-list--suggestions"
          >
            <li
              v-for="(item, index) in latestReview.suggestions"
              :key="`review-suggestion-${index}`"
            >
              {{ item }}
            </li>
          </ul>
        </div>
      </StudioEditorSectionPanel>
    </div>
  </section>
</template>

<style scoped>
.studio-overview-tab {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.studio-overview-tab__summary-grid,
.studio-overview-tab__workspace-row,
.studio-overview-tab__quick-grid {
  display: grid;
  gap: 14px;
}

.studio-overview-tab__summary-grid {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.studio-overview-tab__workspace-row {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.studio-overview-tab__workspace-row--single {
  grid-template-columns: 1fr;
}

.studio-overview-tab__quick-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin-top: 16px;
}

.studio-overview-tab__summary-card,
.studio-overview-tab__quick-card {
  --product-record-card-background: color-mix(in srgb, var(--color-surface-card) 82%, transparent);
  --product-record-card-border: var(--studio-border-default);
  --product-record-card-radius: 22px;
}

.studio-overview-tab__summary-card {
  --product-record-card-padding: 18px;
}

.studio-overview-tab__summary-label {
  display: block;
  color: color-mix(
    in srgb,
    var(--color-action-primary) 27%,
    color-mix(in srgb, var(--color-action-brand-strong) 17.808%, var(--color-text-subtle))
  );
  font-size: 12px;
}

.studio-overview-tab__summary-value {
  display: block;
  margin-top: 8px;
  color: var(--color-text-default);
  font-size: 24px;
}

.studio-overview-tab__summary-description {
  margin: 8px 0 0;
  color: var(--studio-text-muted);
  font-size: 13px;
  line-height: 1.6;
}

.studio-overview-tab__quick-card {
  --product-record-card-padding: 16px;

  padding: 16px;
  transition:
    transform 0.18s ease,
    box-shadow 0.18s ease,
    border-color 0.18s ease;
}

.studio-overview-tab__quick-card:hover {
  border-color: color-mix(in srgb, var(--color-action-brand) 16%, transparent);
  transform: translateY(-2px);
  box-shadow: 0 18px 28px var(--studio-shadow-floating);
}

.studio-overview-tab__quick-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: 14px;
  background: var(--studio-surface-tint);
  color: var(--color-text-link-strong);
  font-size: 16px;
}

.studio-overview-tab__quick-title {
  display: block;
  margin-top: 14px;
  color: var(--studio-text-strong);
  font-size: 14px;
}

.studio-overview-tab__quick-description {
  margin: 8px 0 0;
  color: var(--studio-text-muted);
  font-size: 13px;
  line-height: 1.6;
}

.studio-overview-tab__freeze-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
  margin-top: 16px;
}

.studio-overview-tab__freeze-item {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 16px;
  padding: 12px 14px;
  border: 1px solid var(--studio-border-default);
  border-radius: 16px;
  background: var(--studio-surface-soft);
}

.studio-overview-tab__freeze-item-label {
  color: var(--studio-text-strong);
  font-size: 14px;
  line-height: 1.5;
}

.studio-overview-tab__freeze-item-control {
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.studio-overview-tab__review-summary {
  margin-top: 14px;
  color: var(--studio-text-default);
  font-size: 13px;
  line-height: 1.7;
}

.studio-overview-tab__review-title {
  display: block;
}

.studio-overview-tab__review-list {
  margin: 10px 0 0;
  padding-left: 18px;
}

.studio-overview-tab__review-list--suggestions {
  color: var(--color-text-brand);
}

@media (--breakpoint-studio-down) {
  .studio-overview-tab__summary-grid,
  .studio-overview-tab__workspace-row {
    grid-template-columns: 1fr;
  }
}

@media (--breakpoint-preview-down) {
  .studio-overview-tab__quick-grid {
    grid-template-columns: 1fr;
  }
}
</style>
