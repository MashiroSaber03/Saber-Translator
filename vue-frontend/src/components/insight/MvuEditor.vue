<template>
  <div class="editor-card">
    <div class="header">
      <h4>MVU 变量（extensions.saber_tavern.mvu.variables）</h4>
      <button class="btn" :disabled="locked" @click="addVar">+ 变量</button>
    </div>

    <div v-if="localVars.length === 0" class="placeholder">暂无变量</div>

    <div v-else class="vars">
      <div v-for="(item, idx) in localVars" :key="item.name + idx" class="var-item">
        <div class="var-top">
          <strong>{{ item.name || `var_${idx + 1}` }}</strong>
          <button class="btn danger" :disabled="locked" @click="removeVar(idx)">删除</button>
        </div>

        <div class="grid">
          <label>
            name
            <input v-model="item.name" type="text" :disabled="locked">
          </label>
          <label>
            type
            <input v-model="item.type" type="text" :disabled="locked">
          </label>
          <label>
            scope
            <input v-model="item.scope" type="text" :disabled="locked">
          </label>
          <label>
            value
            <input v-model="valueMirror[idx]" type="text" :disabled="locked" @change="syncValue(idx)">
          </label>
          <label class="full">
            description
            <input v-model="item.description" type="text" :disabled="locked">
          </label>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import type { MvuVariable } from '@/types/characterCard'

const props = defineProps<{
  variables: MvuVariable[]
  locked?: boolean
}>()

const emit = defineEmits<{
  (e: 'update', variables: MvuVariable[]): void
}>()

const localVars = ref<MvuVariable[]>([])
const valueMirror = ref<string[]>([])
let syncing = false

watch(
  () => props.variables,
  value => {
    syncing = true
    localVars.value = JSON.parse(JSON.stringify(value || []))
    valueMirror.value = localVars.value.map(v => String(v.value ?? ''))
    syncing = false
  },
  { immediate: true, deep: true }
)

watch(
  localVars,
  value => {
    if (syncing || props.locked) return
    emit('update', JSON.parse(JSON.stringify(value || [])))
  },
  { deep: true }
)

function syncValue(index: number) {
  if (props.locked) return
  if (!localVars.value[index]) return
  localVars.value[index]!.value = valueMirror.value[index] ?? ''
}

function addVar() {
  if (props.locked) return
  localVars.value.push({
    name: `var_${Date.now()}`,
    type: 'string',
    scope: 'chat',
    default: '',
    value: '',
    validator: {},
    description: '',
  })
  valueMirror.value = localVars.value.map(v => String(v.value ?? ''))
}

function removeVar(index: number) {
  if (props.locked) return
  localVars.value.splice(index, 1)
  valueMirror.value = localVars.value.map(v => String(v.value ?? ''))
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
  color: var(--text-secondary, #64748b);
  font-size: 13px;
}

.vars {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.var-item {
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 8px;
  padding: 10px;
}

.var-top {
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
