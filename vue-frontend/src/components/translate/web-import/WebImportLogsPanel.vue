<script setup lang="ts">
import type { AgentLog, WebImportState } from '@/types/webImport'

defineProps<{
  expanded: boolean
  logs: AgentLog[]
  status: WebImportState['status']
}>()

defineEmits<{
  (event: 'toggle'): void
}>()
</script>

<template>
  <div v-if="logs.length > 0" class="logs-section">
    <div class="logs-header" @click="$emit('toggle')">
      <span class="logs-toggle">{{ expanded ? '▼' : '▶' }}</span>
      <span>AI 工作日志</span>
      <span v-if="status === 'extracting'" class="extracting-hint">(提取中...)</span>
    </div>
    <div v-if="expanded" class="logs-content">
      <div
        v-for="(log, index) in logs"
        :key="index"
        class="log-item"
        :class="`log-${log.type.replaceAll('_', '-')}`"
      >
        <span class="log-time">[{{ log.timestamp }}]</span>
        <span class="log-message">{{ log.message }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.logs-section {
  margin-bottom: 16px;
  overflow: hidden;
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
}

.logs-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  background: var(--web-import-modal-settings-surface-base);
  font-weight: 500;
  font-size: 14px;
  cursor: pointer;
  user-select: none;
}

.logs-toggle {
  color: var(--color-text-supporting);
  font-size: 10px;
}

.extracting-hint {
  color: var(--color-action-primary);
  font-weight: normal;
  font-size: 13px;
}

.logs-content {
  max-height: 200px;
  padding: 12px;
  overflow-y: auto;
  background: var(--web-import-modal-settings-surface-raised);
  font-family: Consolas, Monaco, monospace;
  font-size: 12px;
}

.log-item {
  padding: 2px 0;
  color: var(--web-import-modal-settings-text-secondary);
}

.log-time {
  margin-right: 8px;
  color: var(--color-text-subtle);
}

.log-info .log-message { color: var(--web-import-modal-settings-text-muted); }
.log-tool-call .log-message { color: var(--web-import-modal-settings-text-subtle); }
.log-tool-result .log-message { color: var(--web-import-modal-settings-text-supporting); }
.log-thinking .log-message { color: var(--web-import-modal-settings-text-disabled); }
.log-error .log-message { color: var(--web-import-modal-settings-text-inverse); }
</style>
