<template>
  <div class="editor-card">
    <h4>核心字段</h4>
    <div v-if="!localCard" class="placeholder">请选择角色卡进行编辑</div>

    <div v-else class="form-grid">
      <label>
        角色名
        <input v-model="localCard.data.name" type="text" :disabled="isLocked('data.name')">
      </label>
      <label>
        版本
        <input v-model="localCard.data.character_version" type="text" :disabled="isLocked('data.character_version')">
      </label>

      <label class="full">
        描述
        <textarea v-model="localCard.data.description" rows="3" :disabled="isLocked('data.description')"></textarea>
      </label>
      <label class="full">
        人格
        <textarea v-model="localCard.data.personality" rows="3" :disabled="isLocked('data.personality')"></textarea>
      </label>
      <label class="full">
        场景
        <textarea v-model="localCard.data.scenario" rows="3" :disabled="isLocked('data.scenario')"></textarea>
      </label>
      <label class="full">
        首句
        <textarea v-model="localCard.data.first_mes" rows="2" :disabled="isLocked('data.first_mes')"></textarea>
      </label>
      <label class="full">
        示例对话
        <textarea v-model="localCard.data.mes_example" rows="5" :disabled="isLocked('data.mes_example')"></textarea>
      </label>
      <label class="full">
        备注
        <textarea v-model="localCard.data.creator_notes" rows="2" :disabled="isLocked('data.creator_notes')"></textarea>
      </label>
      <label class="full">
        System Prompt
        <textarea v-model="localCard.data.system_prompt" rows="2" :disabled="isLocked('data.system_prompt')"></textarea>
      </label>
      <label class="full">
        Post History Instructions
        <textarea
          v-model="localCard.data.post_history_instructions"
          rows="2"
          :disabled="isLocked('data.post_history_instructions')"
        ></textarea>
      </label>
      <label class="full">
        Alternate Greetings（每行一条）
        <textarea v-model="alternateGreetingsText" rows="3" :disabled="isLocked('data.alternate_greetings')"></textarea>
      </label>
      <label class="full">
        Tags（逗号分隔）
        <input v-model="tagsText" type="text" :disabled="isLocked('data.tags')">
      </label>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { CharacterCardV2 } from '@/types/characterCard'

const props = defineProps<{
  card: CharacterCardV2 | null
  lockedPaths?: Record<string, boolean>
}>()

const emit = defineEmits<{
  (e: 'update', card: CharacterCardV2): void
}>()

const localCard = ref<CharacterCardV2 | null>(null)
let syncing = false

function cloneCard(card: CharacterCardV2 | null): CharacterCardV2 | null {
  if (!card) return null
  return JSON.parse(JSON.stringify(card))
}

function isLocked(path: string): boolean {
  return !!props.lockedPaths?.[path]
}

watch(
  () => props.card,
  value => {
    syncing = true
    localCard.value = cloneCard(value)
    syncing = false
  },
  { immediate: true, deep: true }
)

watch(
  localCard,
  value => {
    if (syncing || !value) return
    emit('update', cloneCard(value) as CharacterCardV2)
  },
  { deep: true }
)

const alternateGreetingsText = computed({
  get() {
    if (!localCard.value) return ''
    return (localCard.value.data.alternate_greetings || []).join('\n')
  },
  set(value: string) {
    if (!localCard.value) return
    localCard.value.data.alternate_greetings = value
      .split('\n')
      .map(v => v.trim())
      .filter(Boolean)
  },
})

const tagsText = computed({
  get() {
    if (!localCard.value) return ''
    return (localCard.value.data.tags || []).join(', ')
  },
  set(value: string) {
    if (!localCard.value) return
    localCard.value.data.tags = value
      .split(',')
      .map(v => v.trim())
      .filter(Boolean)
  },
})
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

.form-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
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

input,
textarea {
  width: 100%;
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 8px;
  padding: 8px;
  font-size: 13px;
  color: var(--text-primary, #0f172a);
  background: var(--bg-primary, #f8fafc);
}

textarea {
  resize: vertical;
}

@media (max-width: 900px) {
  .form-grid {
    grid-template-columns: 1fr;
  }
}
</style>
