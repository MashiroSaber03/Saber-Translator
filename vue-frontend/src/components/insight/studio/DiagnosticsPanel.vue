<template>
  <section class="diagnostics-shell">
    <div class="summary-grid">
      <article class="summary-card">
        <span class="label">结构诊断</span>
        <strong>{{ diagnostics ? (diagnostics.valid ? '通过' : '存在错误') : '未执行' }}</strong>
      </article>
      <article class="summary-card">
        <span class="label">错误数</span>
        <strong>{{ diagnostics?.errors.length || 0 }}</strong>
      </article>
      <article class="summary-card">
        <span class="label">警告数</span>
        <strong>{{ diagnostics?.warnings.length || 0 }}</strong>
      </article>
    </div>

    <div v-if="!diagnostics" class="empty-copy">还没有运行诊断。建议在导出前至少执行一次结构检查。</div>
    <template v-else>
      <div v-if="diagnostics.errors.length > 0" class="block danger">
        <h4>错误</h4>
        <ul>
          <li v-for="(item, index) in diagnostics.errors" :key="`error-${index}`">{{ item }}</li>
        </ul>
      </div>

      <div v-if="diagnostics.warnings.length > 0" class="block warning">
        <h4>警告</h4>
        <ul>
          <li v-for="(item, index) in diagnostics.warnings" :key="`warning-${index}`">{{ item }}</li>
        </ul>
      </div>

      <div class="checks-block">
        <h4>检查项</h4>
        <div class="check-list">
          <span v-for="(value, key) in diagnostics.checks" :key="key" :class="['check-pill', value ? 'ok' : 'bad']">
            {{ key }} · {{ value ? '通过' : '失败' }}
          </span>
        </div>
      </div>
    </template>
  </section>
</template>

<script setup lang="ts">
import type { ExportDiagnostic } from '@/types/characterStudio'

defineProps<{
  diagnostics: ExportDiagnostic | null
}>()
</script>

<style scoped>
.diagnostics-shell {
  --diagnostics-panel-card-border: rgba(25, 55, 94, .08);
  --diagnostics-panel-card-background: rgba(255, 255, 255, .86);
  --diagnostics-panel-error-background: rgba(255, 244, 244, .86);
  --diagnostics-panel-warning-background: rgba(255, 249, 240, .86);
  --diagnostics-panel-success-pill-background: rgba(32, 170, 103, .14);
  --diagnostics-panel-summary-value-text: #153250;
  --diagnostics-panel-issue-text: #516882;
  --diagnostics-panel-success-pill-text: #17784b;

  display: flex;
  flex-direction: column;
  gap: 16px;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.summary-card,
.block,
.checks-block {
  border-radius: 18px;
  padding: 16px;
  background: var(--diagnostics-panel-card-background);
  border: 1px solid var(--diagnostics-panel-card-border);
}

.summary-card .label {
  display: block;
  font-size: 12px;
  color: var(--studio-text-subtle);
}

.summary-card strong {
  display: block;
  margin-top: 8px;
  color: var(--diagnostics-panel-summary-value-text);
  font-size: 20px;
}

.block h4,
.checks-block h4 {
  margin: 0;
}

.block ul {
  margin: 12px 0 0;
  padding-left: 18px;
  color: var(--diagnostics-panel-issue-text);
  font-size: 13px;
  line-height: 1.7;
}

.danger {
  background: var(--diagnostics-panel-error-background);
}

.warning {
  background: var(--diagnostics-panel-warning-background);
}

.check-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

.check-pill {
  border-radius: 999px;
  padding: 5px 10px;
  font-size: 11px;
}

.check-pill.ok {
  background: var(--diagnostics-panel-success-pill-background);
  color: var(--diagnostics-panel-success-pill-text);
}

.check-pill.bad {
  background: var(--color-surface-danger-soft);
  color: var(--studio-text-danger);
}

.empty-copy {
  color: var(--studio-text-subtle);
  font-size: 13px;
  line-height: 1.6;
}

@media (--breakpoint-lg-down) {
  .summary-grid {
    grid-template-columns: 1fr;
  }
}
</style>
