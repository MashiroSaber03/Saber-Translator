<template>
  <section class="diagnostics-panel">
    <div class="diagnostics-panel__summary-grid">
      <ProductRecordCard class="diagnostics-panel__summary-card">
        <span class="diagnostics-panel__summary-label">结构诊断</span>
        <strong class="diagnostics-panel__summary-value">{{ diagnosticStatus }}</strong>
      </ProductRecordCard>
      <ProductRecordCard class="diagnostics-panel__summary-card">
        <span class="diagnostics-panel__summary-label">错误数</span>
        <strong class="diagnostics-panel__summary-value">{{ diagnostics?.errors.length || 0 }}</strong>
      </ProductRecordCard>
      <ProductRecordCard class="diagnostics-panel__summary-card">
        <span class="diagnostics-panel__summary-label">警告数</span>
        <strong class="diagnostics-panel__summary-value">{{ diagnostics?.warnings.length || 0 }}</strong>
      </ProductRecordCard>
    </div>

    <ProductEmptyState
      v-if="!diagnostics"
      description="建议在导出前至少执行一次结构检查。"
      icon-name="scan-search"
      role="note"
      size="compact"
      title="还没有运行诊断"
    />
    <template v-else>
      <ProductRecordCard
        v-if="diagnostics.errors.length > 0"
        class="diagnostics-panel__issue-card diagnostics-panel__issue-card--danger"
      >
        <h4 class="diagnostics-panel__issue-title">错误</h4>
        <ul class="diagnostics-panel__issue-list">
          <li v-for="(item, index) in diagnostics.errors" :key="`error-${index}`">{{ item }}</li>
        </ul>
      </ProductRecordCard>

      <ProductRecordCard
        v-if="diagnostics.warnings.length > 0"
        class="diagnostics-panel__issue-card diagnostics-panel__issue-card--warning"
      >
        <h4 class="diagnostics-panel__issue-title">警告</h4>
        <ul class="diagnostics-panel__issue-list">
          <li v-for="(item, index) in diagnostics.warnings" :key="`warning-${index}`">{{ item }}</li>
        </ul>
      </ProductRecordCard>

      <ProductRecordCard class="diagnostics-panel__checks-card">
        <h4 class="diagnostics-panel__checks-title">检查项</h4>
        <ProductChipList class="diagnostics-panel__check-list" :items="checkItems" aria-label="诊断检查项" />
      </ProductRecordCard>
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import type { ExportDiagnostic } from '@/types/characterStudio'

const props = defineProps<{
  diagnostics: ExportDiagnostic | null
}>()

const diagnosticStatus = computed(() => {
  if (!props.diagnostics) return '未执行'
  if (props.diagnostics.errors.length > 0) return '存在错误'
  if (props.diagnostics.warnings.length > 0) return '存在警告'
  return props.diagnostics.valid ? '通过' : '未通过'
})

const checkItems = computed<ProductChipItem[]>(() => {
  if (!props.diagnostics) return []

  return Object.entries(props.diagnostics.checks).map(([key, value]) => ({
    id: key,
    label: `${key} · ${value ? '通过' : '失败'}`,
    tone: value ? 'success' : 'danger',
  }))
})
</script>

<style scoped>
.diagnostics-panel {
  --diagnostics-panel-card-border: var(--studio-border-default);
  --diagnostics-panel-card-background: color-mix(in srgb, var(--color-surface-card) 82%, transparent);
  --diagnostics-panel-error-background: var(--color-surface-danger-soft);
  --diagnostics-panel-warning-background: var(--color-status-warning-surface-soft);
  --diagnostics-panel-summary-value-text: var(--studio-text-strong);
  --diagnostics-panel-issue-text: var(--studio-text-muted);

  display: flex;
  flex-direction: column;
  gap: 16px;
}

.diagnostics-panel__summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 160px), 1fr));
  gap: 12px;
}

.diagnostics-panel__summary-card,
.diagnostics-panel__issue-card,
.diagnostics-panel__checks-card {
  --product-record-card-background: var(--diagnostics-panel-card-background);
  --product-record-card-border: var(--diagnostics-panel-card-border);
  --product-record-card-radius: 18px;
  --product-record-card-padding: 16px;
}

.diagnostics-panel__summary-label {
  display: block;
  font-size: 12px;
  color: var(--studio-text-subtle);
}

.diagnostics-panel__summary-value {
  display: block;
  margin-top: 8px;
  color: var(--diagnostics-panel-summary-value-text);
  font-size: 20px;
}

.diagnostics-panel__issue-title,
.diagnostics-panel__checks-title {
  margin: 0;
}

.diagnostics-panel__issue-list {
  margin: 12px 0 0;
  padding-left: 18px;
  color: var(--diagnostics-panel-issue-text);
  font-size: 13px;
  line-height: 1.7;
}

.diagnostics-panel__issue-card--danger {
  --product-record-card-background: var(--diagnostics-panel-error-background);
}

.diagnostics-panel__issue-card--warning {
  --product-record-card-background: var(--diagnostics-panel-warning-background);
}

.diagnostics-panel__check-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

@media (--breakpoint-lg-down) {
  .diagnostics-panel__summary-grid {
    grid-template-columns: 1fr;
  }
}
</style>
