<template>
  <div class="workshop-card">
    <div class="section-head">
      <div>
        <h3>世界书树</h3>
        <p>支持根条目与子条目，适合逐步积累设定与触发知识。</p>
      </div>
      <div class="actions">
        <button class="secondary-btn" @click="addRootEntry">添加根条目</button>
        <button class="ghost-btn" :disabled="importing" @click="pickWorldbook">
          {{ importing ? '导入中...' : '导入世界书' }}
        </button>
      </div>
    </div>

    <input
      ref="worldbookInput"
      hidden
      type="file"
      accept=".json"
      @change="handleWorldbookSelect"
    >

    <div v-if="localEntries.length === 0" class="placeholder">暂无世界书条目。</div>
    <div v-else class="tree-list">
      <LorebookTreeBranch
        v-for="(entry, index) in localEntries"
        :key="entry.id"
        :entry="entry"
        :index="index"
        @update:entry="replaceRootEntry(index, $event)"
        @remove="removeRootEntry(index)"
        @move="moveRootEntry(index, $event)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'
import type { LorebookEntryNode } from '@/types/characterStudio'
import LorebookTreeBranch from './LorebookTreeBranch.vue'

const props = defineProps<{
  entries: LorebookEntryNode[]
  importing: boolean
}>()

const emit = defineEmits<{
  (e: 'update:entries', value: LorebookEntryNode[]): void
  (e: 'import-worldbook', file: File): void
}>()

const localEntries = ref<LorebookEntryNode[]>([])
const worldbookInput = ref<HTMLInputElement | null>(null)
let syncing = false

function cloneEntries(entries: LorebookEntryNode[]) {
  return JSON.parse(JSON.stringify(entries || [])) as LorebookEntryNode[]
}

watch(() => props.entries, value => {
  syncing = true
  localEntries.value = cloneEntries(value)
  void nextTick(() => {
    syncing = false
  })
}, { deep: true, immediate: true })

watch(localEntries, value => {
  if (syncing) return
  emit('update:entries', cloneEntries(value))
}, { deep: true })

function addRootEntry() {
  localEntries.value.push({
    id: `entry_${Date.now()}`,
    comment: '新根条目',
    keys: [],
    secondary_keys: [],
    content: '',
    enabled: true,
    constant: false,
    selective: true,
    priority: 100,
    position: 'before_char',
    depth: 4,
    probability: 100,
    prevent_recursion: true,
    children: [],
  })
}

function pickWorldbook() {
  worldbookInput.value?.click()
}

function handleWorldbookSelect(event: Event) {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) return
  emit('import-worldbook', file)
  target.value = ''
}

function replaceRootEntry(index: number, value: LorebookEntryNode) {
  localEntries.value[index] = value
}

function removeRootEntry(index: number) {
  localEntries.value.splice(index, 1)
}

function moveRootEntry(index: number, offset: -1 | 1) {
  const target = index + offset
  if (target < 0 || target >= localEntries.value.length) return
  const [item] = localEntries.value.splice(index, 1)
  localEntries.value.splice(target, 0, item!)
}
</script>

<style scoped>
.workshop-card {
  border-radius: 22px;
  padding: 18px;
  background: rgba(255, 255, 255, 0.84);
  border: 1px solid rgba(34, 72, 125, 0.12);
  box-shadow: 0 18px 38px rgba(21, 44, 77, 0.08);
}

.section-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}

.section-head h3 {
  margin: 0;
}

.section-head p {
  margin: 6px 0 0;
  color: #5d738c;
  font-size: 13px;
}

.actions {
  display: flex;
  gap: 10px;
  align-items: flex-start;
}

.secondary-btn,
.ghost-btn {
  border: none;
  border-radius: 12px;
  padding: 10px 14px;
  cursor: pointer;
}

.secondary-btn {
  background: rgba(41, 96, 193, 0.1);
  color: #275ebe;
}

.ghost-btn {
  background: rgba(18, 47, 86, 0.08);
  color: #244979;
}

.secondary-btn:disabled,
.ghost-btn:disabled {
  opacity: 0.68;
  cursor: not-allowed;
}

.placeholder {
  color: #72869c;
  font-size: 13px;
}

.tree-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

@media (max-width: 900px) {
  .section-head {
    flex-direction: column;
  }
}
</style>
