<template>
  <div class="editor-card">
    <div class="header">
      <h4>正则模板（extensions.saber_tavern.regex_profiles）</h4>
      <button class="btn" :disabled="locked" @click="addRule">+ 规则</button>
    </div>

    <div v-if="localRules.length === 0" class="placeholder">暂无规则</div>

    <div v-else class="rules">
      <div v-for="(rule, idx) in localRules" :key="rule.id || idx" class="rule-item">
        <div class="rule-top">
          <strong>{{ rule.name || `规则${idx + 1}` }}</strong>
          <button class="btn danger" :disabled="locked" @click="removeRule(idx)">删除</button>
        </div>
        <div class="grid">
          <label>
            id
            <input v-model="rule.id" type="text" :disabled="locked">
          </label>
          <label>
            名称
            <input v-model="rule.name" type="text" :disabled="locked">
          </label>
          <label>
            source
            <input v-model="rule.source" type="text" :disabled="locked">
          </label>
          <label>
            flags
            <input v-model="rule.flags" type="text" :disabled="locked">
          </label>
          <label class="full">
            pattern
            <input v-model="rule.pattern" type="text" :disabled="locked">
          </label>
          <label class="full">
            replacement
            <input v-model="rule.replacement" type="text" :disabled="locked">
          </label>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import type { RegexProfile } from '@/types/characterCard'

const props = defineProps<{
  rules: RegexProfile[]
  locked?: boolean
}>()

const emit = defineEmits<{
  (e: 'update', rules: RegexProfile[]): void
}>()

const localRules = ref<RegexProfile[]>([])
let syncing = false

watch(
  () => props.rules,
  value => {
    syncing = true
    localRules.value = JSON.parse(JSON.stringify(value || []))
    syncing = false
  },
  { immediate: true, deep: true }
)

watch(
  localRules,
  value => {
    if (syncing || props.locked) return
    emit('update', JSON.parse(JSON.stringify(value || [])))
  },
  { deep: true }
)

function addRule() {
  if (props.locked) return
  const next: RegexProfile = {
    id: `rule_${Date.now()}`,
    name: '新规则',
    enabled: true,
    scope: 'character',
    source: 'ai_output',
    pattern: '',
    replacement: '',
    flags: 'g',
    depth_min: 0,
    depth_max: 99,
    order: 100,
    notes: '',
  }
  localRules.value.push(next)
}

function removeRule(index: number) {
  if (props.locked) return
  localRules.value.splice(index, 1)
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
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.header h4 {
  margin: 0;
  font-size: 14px;
}

.placeholder {
  font-size: 13px;
  color: var(--text-secondary, #64748b);
}

.rules {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.rule-item {
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 8px;
  padding: 10px;
}

.rule-top {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
}

.grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}

label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  color: var(--text-secondary, #64748b);
}

label.full {
  grid-column: 1 / -1;
}

input {
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 8px;
  padding: 8px;
  font-size: 13px;
  background: var(--bg-primary, #f8fafc);
}

.btn {
  border: 1px solid var(--border-color, #e2e8f0);
  background: var(--bg-tertiary, #f1f5f9);
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 12px;
  cursor: pointer;
}

.btn.danger {
  color: #b91c1c;
  border-color: rgba(239, 68, 68, 0.35);
}

@media (max-width: 900px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
</style>
