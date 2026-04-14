<template>
  <div class="editor-card">
    <div class="header">
      <h4>字段锁</h4>
      <button class="btn" @click="$emit('unlock-all')">全部解锁</button>
    </div>
    <p class="hint">锁定后，生成预设/批量编辑/手动编辑都不会改动该字段。</p>

    <div class="lock-list">
      <label v-for="item in options" :key="item.path" class="lock-item">
        <input
          type="checkbox"
          :checked="!!locks[item.path]"
          @change="onToggle(item.path, $event)"
        >
        <div class="lock-body">
          <div class="title">{{ item.label }}</div>
          <div class="path">{{ item.path }}</div>
        </div>
      </label>
    </div>
  </div>
</template>

<script setup lang="ts">
interface LockOption {
  path: string
  label: string
}

defineProps<{
  locks: Record<string, boolean>
  options: LockOption[]
}>()

const emit = defineEmits<{
  (e: 'toggle', path: string, value: boolean): void
  (e: 'unlock-all'): void
}>()

function onToggle(path: string, event: Event) {
  const target = event.target as HTMLInputElement
  emit('toggle', path, target.checked)
}
</script>

<style scoped>
.editor-card {
  background: var(--bg-secondary, #fff);
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 10px;
  padding: 14px;
}

.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.header h4 {
  margin: 0;
  font-size: 14px;
}

.hint {
  margin: 0 0 10px;
  color: var(--text-secondary, #64748b);
  font-size: 12px;
}

.lock-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.lock-item {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 8px;
  align-items: flex-start;
  padding: 8px;
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 8px;
}

.lock-body .title {
  font-size: 13px;
  color: var(--text-primary, #0f172a);
}

.lock-body .path {
  font-size: 11px;
  color: var(--text-secondary, #64748b);
  word-break: break-all;
}

.btn {
  border: 1px solid var(--border-color, #e2e8f0);
  background: var(--bg-tertiary, #f1f5f9);
  border-radius: 6px;
  padding: 5px 8px;
  font-size: 12px;
  cursor: pointer;
}
</style>
