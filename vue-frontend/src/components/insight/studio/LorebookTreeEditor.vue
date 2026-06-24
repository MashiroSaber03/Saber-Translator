<template>
  <div class="workshop-card">
    <div class="section-head">
      <div>
        <h3>世界书树</h3>
        <p>支持根条目与子条目，适合逐步积累设定与触发知识。</p>
      </div>
      <div class="actions">
        <UiButton variant="toolbar" class="action-secondary" @click="addRootEntry">添加根条目</UiButton>
        <UiButton variant="toolbar" class="action-ghost" :disabled="importing" @click="pickWorldbook">
          {{ importing ? '导入中...' : '导入世界书' }}
        </UiButton>
      </div>
    </div>

    <UiFileInput
      ref="worldbookInput"
      hidden
      accept=".json"
      @change="handleWorldbookSelect"
    />

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
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
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
  --lorebook-tree-editor-border-default: rgba(34, 72, 125, .12);
  --lorebook-tree-editor-shadow-default: rgba(21, 44, 77, .08);
  --lorebook-tree-editor-surface-base: rgba(255, 255, 255, .84);
  --lorebook-tree-editor-surface-raised: rgba(41, 96, 193, .1);
  --lorebook-tree-editor-surface-muted: rgba(18, 47, 86, .08);
  --lorebook-tree-editor-text-primary: #5d738c;
  --lorebook-tree-editor-text-secondary: #275ebe;
  --lorebook-tree-editor-text-muted: #244979;
  --lorebook-tree-editor-text-subtle: #72869c;

  border-radius: 22px;
  padding: 18px;
  background: var(--lorebook-tree-editor-surface-base);
  border: 1px solid var(--lorebook-tree-editor-border-default);
  box-shadow: 0 18px 38px var(--lorebook-tree-editor-shadow-default);
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
  color: var(--lorebook-tree-editor-text-primary);
  font-size: 13px;
}

.actions {
  display: flex;
  gap: 10px;
  align-items: flex-start;
}

.action-secondary,
.action-ghost {
  border: none;
  border-radius: 12px;
  padding: 10px 14px;
  cursor: pointer;
}

.action-secondary {
  background: var(--lorebook-tree-editor-surface-raised);
  color: var(--lorebook-tree-editor-text-secondary);
}

.action-ghost {
  background: var(--lorebook-tree-editor-surface-muted);
  color: var(--lorebook-tree-editor-text-muted);
}

.action-secondary:disabled,
.action-ghost:disabled {
  opacity: 0.68;
  cursor: not-allowed;
}

.placeholder {
  color: var(--lorebook-tree-editor-text-subtle);
  font-size: 13px;
}

.tree-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

@media (--breakpoint-lg-down) {
  .section-head {
    flex-direction: column;
  }
}
</style>
