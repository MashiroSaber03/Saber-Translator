<template>
  <div class="lorebook-tree-branch">
    <details class="lorebook-tree-branch__details" open>
      <summary class="lorebook-tree-branch__summary">
        <div class="lorebook-tree-branch__summary-main">
          <UiInput v-model="localEntry.comment" class="lorebook-tree-branch__title-input" type="text" variant="studio" />
          <div class="lorebook-tree-branch__meta">
            <span class="lorebook-tree-branch__meta-item">{{ localEntry.keys.length }} 个关键词</span>
            <span class="lorebook-tree-branch__meta-item">优先级 {{ localEntry.priority }}</span>
            <span class="lorebook-tree-branch__meta-item">{{ localEntry.position }}</span>
          </div>
        </div>
        <ProductActionRow class="lorebook-tree-branch__actions" aria-label="世界书条目操作" @click.prevent>
          <UiButton variant="secondary" size="sm" @click="move(-1)" :disabled="index === 0">上移</UiButton>
          <UiButton variant="secondary" size="sm" @click="move(1)" :disabled="index >= siblingCount - 1">下移</UiButton>
          <UiButton variant="secondary" size="sm" @click="addChild">子项</UiButton>
          <UiButton variant="secondary" tone="danger" size="sm" @click="$emit('remove')">删除</UiButton>
        </ProductActionRow>
      </summary>

      <div class="lorebook-tree-branch__body">
        <UiFormGrid class="lorebook-tree-branch__grid">
          <UiField variant="settings" label="关键词（逗号分隔）" :control-id="entryControlId('keys')">
            <UiInput
              :id="entryControlId('keys')"
              :model-value="localEntry.keys.join(', ')"
              type="text"
              variant="studio"
              @update:model-value="updateKeys(String($event))"
            />
          </UiField>
          <UiField variant="settings" label="次级关键词（逗号分隔）" :control-id="entryControlId('secondary-keys')">
            <UiInput
              :id="entryControlId('secondary-keys')"
              :model-value="(localEntry.secondary_keys || []).join(', ')"
              type="text"
              variant="studio"
              @update:model-value="updateSecondaryKeys(String($event))"
            />
          </UiField>
          <UiField class="lorebook-tree-branch__field--full" variant="settings" label="内容" :control-id="entryControlId('content')">
            <UiTextarea :id="entryControlId('content')" v-model="localEntry.content" rows="4" variant="studio" />
          </UiField>
          <UiField variant="settings" label="优先级" :control-id="entryControlId('priority')">
            <UiNumberField
              :input-id="entryControlId('priority')"
              v-model="localEntry.priority"
              aria-label="世界书条目优先级"
              :min="0"
              :step="10"
              variant="studio"
            />
          </UiField>
          <UiField variant="settings" label="注入位置" :control-id="entryControlId('position')">
            <UiSelect
              :id="entryControlId('position')"
              v-model="localEntry.position"
              :options="LOREBOOK_POSITION_OPTIONS"
              variant="studio"
            />
          </UiField>
          <UiField variant="settings" label="深度" :control-id="entryControlId('depth')">
            <UiNumberField
              :input-id="entryControlId('depth')"
              v-model="localEntry.depth"
              aria-label="世界书条目深度"
              :min="0"
              variant="studio"
            />
          </UiField>
          <UiField variant="settings" label="概率" :control-id="entryControlId('probability')">
            <UiNumberField
              :input-id="entryControlId('probability')"
              :model-value="localEntry.probability ?? null"
              aria-label="世界书条目概率"
              :min="0"
              :max="100"
              variant="studio"
              @update:model-value="value => { if (value !== null) localEntry.probability = value }"
            />
          </UiField>
        </UiFormGrid>

        <div class="lorebook-tree-branch__toggles">
          <UiCheckbox v-model="localEntry.enabled" label="启用" />
          <UiCheckbox v-model="localEntry.constant" label="常驻" />
          <UiCheckbox v-model="localEntry.selective" label="选择触发" />
          <UiCheckbox v-model="localEntry.prevent_recursion" label="防递归" />
          <UiCheckbox v-model="localEntry.use_regex" label="用正则匹配" />
        </div>

        <div v-if="localEntry.children.length > 0" class="lorebook-tree-branch__children">
          <LorebookTreeBranch
            v-for="(child, childIndex) in localEntry.children"
            :key="child.id"
            :entry="child"
            :index="childIndex"
            :sibling-count="localEntry.children.length"
            @update:entry="replaceChild(childIndex, $event)"
            @remove="removeChild(childIndex)"
            @move="moveChild(childIndex, $event)"
          />
        </div>
      </div>
    </details>
  </div>
</template>

<script setup lang="ts">
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import { nextTick, ref, watch } from 'vue'
import type { LorebookEntryNode } from '@/types/characterStudio'
import type { UiSelectOption } from '@/components/ui/selectTypes'
import { deepClone } from '@/utils/deepClone'

const LOREBOOK_POSITION_OPTIONS: UiSelectOption[] = [
  { label: 'before_char', value: 'before_char' },
  { label: 'after_char', value: 'after_char' },
  { label: 'at_depth', value: 'at_depth' },
]

const props = defineProps<{
  entry: LorebookEntryNode
  index: number
  siblingCount: number
}>()

const emit = defineEmits<{
  (e: 'update:entry', value: LorebookEntryNode): void
  (e: 'remove'): void
  (e: 'move', offset: -1 | 1): void
}>()

const localEntry = ref<LorebookEntryNode>(deepClone(props.entry))
let syncing = false

watch(() => props.entry, value => {
  syncing = true
  localEntry.value = deepClone(value)
  void nextTick(() => {
    syncing = false
  })
}, { deep: true, immediate: true })

watch(localEntry, value => {
  if (syncing) return
  emit('update:entry', deepClone(value))
}, { deep: true })

function splitCsv(value: string) {
  return value.split(/[,，]/).map(item => item.trim()).filter(Boolean)
}

function entryControlId(field: string) {
  return `lorebook-${localEntry.value.id}-${field}`
}

function updateKeys(value: string) {
  localEntry.value.keys = splitCsv(value)
}

function updateSecondaryKeys(value: string) {
  localEntry.value.secondary_keys = splitCsv(value)
}

function addChild() {
  localEntry.value.children.push({
    id: `entry_${Date.now()}_${Math.random().toString(16).slice(2, 6)}`,
    comment: '新子条目',
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

function replaceChild(index: number, value: LorebookEntryNode) {
  localEntry.value.children[index] = value
}

function removeChild(index: number) {
  localEntry.value.children.splice(index, 1)
}

function moveChild(index: number, offset: -1 | 1) {
  const target = index + offset
  if (target < 0 || target >= localEntry.value.children.length) return
  const [item] = localEntry.value.children.splice(index, 1)
  localEntry.value.children.splice(target, 0, item!)
}

function move(offset: -1 | 1) {
  emit('move', offset)
}
</script>

<style scoped>
.lorebook-tree-branch {
  --lorebook-tree-branch-border-default: var(--studio-border-default);
  --lorebook-tree-branch-surface-base: color-mix(in srgb, var(--color-surface-card) 82%, transparent);
  --lorebook-tree-branch-text-primary: var(--studio-text-strong);

  border-radius: 18px;
  background: var(--lorebook-tree-branch-surface-base);
  border: 1px solid var(--studio-border-default);
}

.lorebook-tree-branch__details {
  border-radius: 18px;
}

.lorebook-tree-branch__summary {
  list-style: none;
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 16px;
  cursor: pointer;
}

.lorebook-tree-branch__summary::-webkit-details-marker {
  display: none;
}

.lorebook-tree-branch__summary-main {
  min-width: 0;
  flex: 1;
}

.lorebook-tree-branch__title-input {
  width: 100%;
  border: none;
  background: transparent;
  color: var(--lorebook-tree-branch-text-primary);
  font-size: 14px;
  font-weight: 600;
  padding: 0;
}

.lorebook-tree-branch__meta {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 8px;
  color: var(--studio-text-subtle);
  font-size: 11px;
}

.lorebook-tree-branch__actions {
  align-items: flex-start;
}

.lorebook-tree-branch__body {
  padding: 0 16px 16px;
}

.lorebook-tree-branch__grid {
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr));
  gap: 10px;
  margin-bottom: 0;
}

.lorebook-tree-branch__field--full {
  grid-column: 1 / -1;
}

.lorebook-tree-branch__toggles {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 12px;
}

.lorebook-tree-branch__children {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 14px;
  padding-left: 16px;
  border-left: 2px solid var(--lorebook-tree-branch-border-default);
}

@media (--breakpoint-lg-down) {
  .lorebook-tree-branch__summary,
  .lorebook-tree-branch__actions,
  .lorebook-tree-branch__grid {
    grid-template-columns: 1fr;
    flex-direction: column;
  }
}
</style>
