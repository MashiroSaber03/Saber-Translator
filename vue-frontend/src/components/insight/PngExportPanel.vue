<template>
  <div class="export-panel">
    <h4>PNG 导出</h4>
    <p class="hint">每个角色导出一张内嵌完整数据的角色卡 PNG。</p>

    <div class="actions">
      <button
        class="btn primary"
        :disabled="!selectedCharacter || !canExport || exportingSingle"
        @click="$emit('export-single', selectedCharacter)"
      >
        {{ exportingSingle ? '导出中...' : `导出当前角色 PNG` }}
      </button>
      <button
        class="btn secondary"
        :disabled="selectedCharacters.length === 0 || !canExport || exportingBatch"
        @click="$emit('export-batch', selectedCharacters)"
      >
        {{ exportingBatch ? '打包中...' : `批量导出 (${selectedCharacters.length})` }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  selectedCharacter: string
  selectedCharacters: string[]
  canExport: boolean
  exportingSingle: boolean
  exportingBatch: boolean
}>()

defineEmits<{
  (e: 'export-single', character: string): void
  (e: 'export-batch', characters: string[]): void
}>()
</script>

<style scoped>
.export-panel {
  background: var(--bg-secondary, #fff);
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 10px;
  padding: 14px;
}

.export-panel h4 {
  margin: 0 0 6px;
  font-size: 14px;
}

.hint {
  margin: 0 0 12px;
  font-size: 12px;
  color: var(--text-secondary, #64748b);
}

.actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.btn {
  border: none;
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 13px;
  cursor: pointer;
}

.btn.primary {
  background: var(--color-primary, #6366f1);
  color: #fff;
}

.btn.secondary {
  background: var(--bg-tertiary, #f1f5f9);
  color: var(--text-primary, #0f172a);
  border: 1px solid var(--border-color, #e2e8f0);
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
</style>
