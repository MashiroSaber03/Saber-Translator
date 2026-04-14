<template>
  <div class="editor-card">
    <h4>Tavern Helper UI 模板（extensions.saber_tavern.ui_manifest）</h4>

    <div v-if="!localManifest" class="placeholder">暂无 UI manifest</div>

    <div v-else class="layout">
      <div class="meta-grid">
        <label>
          layout
          <input v-model="localManifest.layout" type="text" :disabled="locked">
        </label>
        <label>
          theme
          <input v-model="localManifest.theme" type="text" :disabled="locked">
        </label>
      </div>

      <label>
        manifest JSON（高级编辑）
        <textarea v-model="jsonText" rows="10" :disabled="locked"></textarea>
      </label>
      <div class="actions">
        <button class="btn" :disabled="locked" @click="applyJson">应用 JSON</button>
        <span v-if="errorMessage" class="error">{{ errorMessage }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import type { HelperUiManifest } from '@/types/characterCard'

const props = defineProps<{
  manifest: HelperUiManifest | null
  locked?: boolean
}>()

const emit = defineEmits<{
  (e: 'update', manifest: HelperUiManifest): void
}>()

const localManifest = ref<HelperUiManifest | null>(null)
const jsonText = ref('')
const errorMessage = ref('')
let syncing = false

watch(
  () => props.manifest,
  value => {
    syncing = true
    localManifest.value = value ? JSON.parse(JSON.stringify(value)) : null
    jsonText.value = value ? JSON.stringify(value, null, 2) : ''
    errorMessage.value = ''
    syncing = false
  },
  { immediate: true, deep: true }
)

watch(
  localManifest,
  value => {
    if (syncing || !value || props.locked) return
    jsonText.value = JSON.stringify(value, null, 2)
    emit('update', JSON.parse(JSON.stringify(value)))
  },
  { deep: true }
)

function applyJson() {
  if (props.locked) return
  if (!jsonText.value.trim()) return
  try {
    const parsed = JSON.parse(jsonText.value) as HelperUiManifest
    localManifest.value = parsed
    errorMessage.value = ''
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : 'JSON 解析失败'
  }
}
</script>

<style scoped>
.editor-card {
  background: var(--bg-secondary, #fff);
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 10px;
  padding: 14px;
}

.editor-card h4 {
  margin: 0 0 10px;
  font-size: 14px;
}

.placeholder {
  font-size: 13px;
  color: var(--text-secondary, #64748b);
}

.layout {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.meta-grid {
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

input,
textarea {
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 8px;
  padding: 8px;
  font-size: 13px;
  background: var(--bg-primary, #f8fafc);
}

.actions {
  display: flex;
  align-items: center;
  gap: 10px;
}

.btn {
  border: 1px solid var(--border-color, #e2e8f0);
  background: var(--bg-tertiary, #f1f5f9);
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 12px;
  cursor: pointer;
}

.error {
  font-size: 12px;
  color: #b91c1c;
}

@media (max-width: 900px) {
  .meta-grid {
    grid-template-columns: 1fr;
  }
}
</style>
