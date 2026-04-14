<template>
  <div class="validation-panel">
    <div class="panel-header">
      <h4>校验结果</h4>
      <span v-if="result" class="status" :class="{ valid: result.valid, invalid: !result.valid }">
        {{ result.valid ? '通过' : '未通过' }}
      </span>
    </div>

    <div v-if="!result" class="placeholder">尚未编译校验</div>

    <template v-else>
      <div v-if="(result.errors || []).length > 0" class="issues error">
        <h5>错误</h5>
        <ul>
          <li v-for="(item, index) in result.errors" :key="`e-${index}`">{{ item }}</li>
        </ul>
      </div>

      <div v-if="(result.warnings || []).length > 0" class="issues warning">
        <h5>警告</h5>
        <ul>
          <li v-for="(item, index) in result.warnings" :key="`w-${index}`">{{ item }}</li>
        </ul>
      </div>

      <div
        v-if="result.compatibility_reports && Object.keys(result.compatibility_reports).length > 0"
        class="issues compat"
      >
        <h5>兼容性诊断</h5>
        <div
          v-for="(report, character) in result.compatibility_reports"
          :key="`compat-${character}`"
          class="compat-item"
        >
          <div class="compat-title">
            <strong>{{ character }}</strong>
            <span :class="{ ok: report.compatible, bad: !report.compatible }">
              {{ report.compatible ? 'V2可用' : 'V2不通过' }}
            </span>
            <span :class="{ ok: report.helper_ready, bad: !report.helper_ready }">
              {{ report.helper_ready ? 'Helper增强可用' : 'Helper增强不完整' }}
            </span>
          </div>
          <div class="compat-checks">
            <span v-for="(status, key) in report.checks" :key="`${character}-${key}`" :class="{ ok: status, bad: !status }">
              {{ key }}: {{ status ? 'ok' : 'missing' }}
            </span>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import type { CharacterCardCompileResponse } from '@/types/characterCard'

defineProps<{
  result: CharacterCardCompileResponse | null
}>()
</script>

<style scoped>
.validation-panel {
  background: var(--bg-secondary, #fff);
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 10px;
  padding: 14px;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.panel-header h4 {
  margin: 0;
  font-size: 14px;
}

.status {
  font-size: 12px;
  padding: 2px 8px;
  border-radius: 999px;
}

.status.valid {
  background: rgba(34, 197, 94, 0.15);
  color: #15803d;
}

.status.invalid {
  background: rgba(239, 68, 68, 0.15);
  color: #b91c1c;
}

.placeholder {
  color: var(--text-secondary, #64748b);
  font-size: 13px;
}

.issues {
  margin-top: 10px;
}

.issues h5 {
  margin: 0 0 6px;
  font-size: 13px;
}

.issues ul {
  margin: 0;
  padding-left: 18px;
  font-size: 12px;
  line-height: 1.5;
}

.issues.error h5 {
  color: #b91c1c;
}

.issues.warning h5 {
  color: #92400e;
}

.compat-item {
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 8px;
  padding: 8px;
  margin-bottom: 8px;
}

.compat-title {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 6px;
}

.compat-title span,
.compat-checks span {
  font-size: 11px;
  border-radius: 999px;
  padding: 2px 8px;
}

.compat-checks {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.ok {
  color: #166534;
  background: rgba(34, 197, 94, 0.12);
}

.bad {
  color: #b91c1c;
  background: rgba(239, 68, 68, 0.12);
}
</style>
