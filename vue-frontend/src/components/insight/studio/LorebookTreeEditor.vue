<template>
  <div class="lorebook-tree-editor">
    <div class="lorebook-tree-editor__head">
      <div class="lorebook-tree-editor__head-copy">
        <h3 class="lorebook-tree-editor__title">世界书树</h3>
        <p class="lorebook-tree-editor__description">支持根条目与子条目，适合逐步积累设定与触发知识。</p>
      </div>
      <ProductActionRow aria-label="世界书树操作">
        <UiButton variant="primary" @click="addRootEntry">添加根条目</UiButton>
        <UiButton variant="secondary" :disabled="importing" @click="pickWorldbook">
          {{ importing ? '导入中...' : '导入世界书' }}
        </UiButton>
      </ProductActionRow>
    </div>

    <UiFileInput
      ref="worldbookInput"
      hidden
      accept=".json"
      @files-change="handleWorldbookSelect"
    />

    <ProductEmptyState
      v-if="localEntries.length === 0"
      icon-name="book-open"
      role="note"
      size="compact"
      title="暂无世界书条目"
      description="添加根条目或导入世界书后，会在这里维护触发知识树。"
    />
    <div v-else class="lorebook-tree-editor__tree-list">
      <LorebookTreeBranch
        v-for="(entry, index) in localEntries"
        :key="entry.id"
        :entry="entry"
        :index="index"
        :sibling-count="localEntries.length"
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
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import { nextTick, ref, watch } from 'vue'
import type { LorebookEntryNode } from '@/types/characterStudio'
import { deepClone } from '@/utils/deepClone'
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
const worldbookInput = ref<InstanceType<typeof UiFileInput> | null>(null)
let syncing = false

watch(() => props.entries, value => {
  syncing = true
  localEntries.value = deepClone(value || [])
  void nextTick(() => {
    syncing = false
  })
}, { deep: true, immediate: true })

watch(localEntries, value => {
  if (syncing) return
  emit('update:entries', deepClone(value))
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

function handleWorldbookSelect(files: File[]) {
  const file = files[0]
  if (!file) return
  emit('import-worldbook', file)
  worldbookInput.value?.clear()
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
.lorebook-tree-editor {
  --lorebook-tree-editor-card-border: var(--studio-border-default);
  --lorebook-tree-editor-card-shadow: var(--studio-shadow-floating);
  --lorebook-tree-editor-card-background: color-mix(in srgb, var(--color-surface-card) 82%, transparent);
  --lorebook-tree-editor-description-text: var(--studio-text-muted);

  border-radius: 22px;
  padding: 18px;
  background: var(--lorebook-tree-editor-card-background);
  border: 1px solid var(--lorebook-tree-editor-card-border);
  box-shadow: 0 18px 38px var(--lorebook-tree-editor-card-shadow);
}

.lorebook-tree-editor__head {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}

.lorebook-tree-editor__head-copy {
  min-width: 0;
}

.lorebook-tree-editor__title {
  margin: 0;
}

.lorebook-tree-editor__description {
  margin: 6px 0 0;
  color: var(--lorebook-tree-editor-description-text);
  font-size: 13px;
}

.lorebook-tree-editor__tree-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

@media (--breakpoint-lg-down) {
  .lorebook-tree-editor__head {
    flex-direction: column;
  }
}
</style>
