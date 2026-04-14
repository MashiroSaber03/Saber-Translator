<template>
  <div class="editor-card">
    <h4>世界书（character_book）</h4>
    <div v-if="!localBook" class="placeholder">当前角色卡没有世界书数据</div>

    <div v-else class="layout">
      <div class="meta-grid">
        <label>
          名称
          <input v-model="localBook.name" type="text" :disabled="locked">
        </label>
        <label>
          描述
          <input v-model="localBook.description" type="text" :disabled="locked">
        </label>
        <label>
          scan_depth
          <input v-model.number="localBook.scan_depth" type="number" min="0" max="20" :disabled="locked">
        </label>
        <label>
          token_budget
          <input v-model.number="localBook.token_budget" type="number" min="64" max="4096" :disabled="locked">
        </label>
      </div>

      <div class="entry-header">
        <span>条目（{{ localBook.entries.length }}）</span>
        <button class="btn" :disabled="locked" @click="addEntry">+ 新增条目</button>
      </div>

      <div class="entries">
        <div v-for="(entry, idx) in localBook.entries" :key="entry.uid || idx" class="entry-item">
          <div class="entry-top">
            <strong>#{{ idx + 1 }}</strong>
            <button class="btn danger" :disabled="locked" @click="removeEntry(idx)">删除</button>
          </div>
          <label>
            comment
            <input v-model="entry.comment" type="text" :disabled="locked">
          </label>
          <label>
            key（逗号分隔）
            <input :value="entry.key.join(', ')" type="text" :disabled="locked" @input="onKeyInput(entry, $event)">
          </label>
          <label>
            content
            <textarea v-model="entry.content" rows="3" :disabled="locked"></textarea>
          </label>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import type { CharacterBookEntry, CharacterBookSchema } from '@/types/characterCard'

const props = defineProps<{
  book: CharacterBookSchema | null
  locked?: boolean
}>()

const emit = defineEmits<{
  (e: 'update', book: CharacterBookSchema): void
}>()

const localBook = ref<CharacterBookSchema | null>(null)
let syncing = false

function cloneBook(book: CharacterBookSchema | null): CharacterBookSchema | null {
  if (!book) return null
  return JSON.parse(JSON.stringify(book))
}

watch(
  () => props.book,
  value => {
    syncing = true
    localBook.value = cloneBook(value)
    syncing = false
  },
  { immediate: true, deep: true }
)

watch(
  localBook,
  value => {
    if (syncing || !value || props.locked) return
    emit('update', cloneBook(value) as CharacterBookSchema)
  },
  { deep: true }
)

function addEntry() {
  if (props.locked) return
  if (!localBook.value) return
  const nextUid =
    (localBook.value.entries || []).reduce((max, e) => Math.max(max, Number(e.uid || 0)), 0) + 1
  const entry: CharacterBookEntry = {
    uid: nextUid,
    key: [localBook.value.name || 'keyword'],
    keysecondary: [],
    comment: '',
    content: '',
    constant: false,
    selective: false,
    insertion_order: 100,
    enabled: true,
    position: 'before_char',
    extensions: {},
  }
  localBook.value.entries.push(entry)
}

function removeEntry(index: number) {
  if (props.locked) return
  if (!localBook.value) return
  localBook.value.entries.splice(index, 1)
}

function onKeyInput(entry: CharacterBookEntry, event: Event) {
  if (props.locked) return
  const target = event.target as HTMLInputElement
  entry.key = target.value
    .split(',')
    .map(v => v.trim())
    .filter(Boolean)
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
  gap: 12px;
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

.entry-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
}

.entries {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.entry-item {
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 8px;
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.entry-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
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
  .meta-grid {
    grid-template-columns: 1fr;
  }
}
</style>
