<template>
  <div class="lorebook-tree-branch">
    <details class="lorebook-tree-branch__details" open>
      <summary class="lorebook-tree-branch__summary">
        <div class="lorebook-tree-branch__summary-main">
          <UiInput
            :model-value="entry.comment"
            class="lorebook-tree-branch__title-input"
            type="text"
            variant="studio"
            @update:model-value="updateStringField('comment', String($event))"
          />
          <div class="lorebook-tree-branch__meta">
            <span class="lorebook-tree-branch__meta-item">{{ entry.keys.length }} 个关键词</span>
            <span class="lorebook-tree-branch__meta-item">优先级 {{ entry.priority }}</span>
            <span class="lorebook-tree-branch__meta-item">{{ entry.position }}</span>
          </div>
        </div>
        <ProductActionRow
          appearance="accent"
          class="lorebook-tree-branch__actions"
          aria-label="世界书条目操作"
          @click.prevent
        >
          <UiButton variant="secondary" size="sm" @click="move(-1)" :disabled="index === 0">
            上移
          </UiButton>
          <UiButton
            variant="secondary"
            size="sm"
            @click="move(1)"
            :disabled="index >= siblingCount - 1"
          >
            下移
          </UiButton>
          <UiButton variant="secondary" size="sm" @click="addChild">子项</UiButton>
          <UiButton variant="secondary" tone="danger" size="sm" @click="$emit('remove')">
            删除
          </UiButton>
        </ProductActionRow>
      </summary>

      <div class="lorebook-tree-branch__body">
        <UiFormGrid class="lorebook-tree-branch__grid">
          <UiField
            variant="settings"
            label="关键词（逗号分隔）"
            :control-id="entryControlId('keys')"
          >
            <UiInput
              :id="entryControlId('keys')"
              :model-value="entry.keys.join(', ')"
              type="text"
              variant="studio"
              @update:model-value="updateKeys(String($event))"
            />
          </UiField>
          <UiField
            variant="settings"
            label="次级关键词（逗号分隔）"
            :control-id="entryControlId('secondary-keys')"
          >
            <UiInput
              :id="entryControlId('secondary-keys')"
              :model-value="(entry.secondary_keys || []).join(', ')"
              type="text"
              variant="studio"
              @update:model-value="updateSecondaryKeys(String($event))"
            />
          </UiField>
          <UiField
            class="lorebook-tree-branch__field--full"
            variant="settings"
            label="内容"
            :control-id="entryControlId('content')"
          >
            <UiTextarea
              :id="entryControlId('content')"
              :model-value="entry.content"
              rows="4"
              variant="studio"
              @update:model-value="updateStringField('content', $event)"
            />
          </UiField>
          <UiField variant="settings" label="优先级" :control-id="entryControlId('priority')">
            <UiNumberField
              :input-id="entryControlId('priority')"
              :model-value="entry.priority"
              aria-label="世界书条目优先级"
              :min="0"
              :step="10"
              variant="studio"
              @update:model-value="updateNumberField('priority', $event ?? 0)"
            />
          </UiField>
          <UiField variant="settings" label="注入位置" :control-id="entryControlId('position')">
            <UiSelect
              :id="entryControlId('position')"
              :model-value="entry.position"
              :options="LOREBOOK_POSITION_OPTIONS"
              variant="studio"
              @change="updateStringField('position', String($event))"
            />
          </UiField>
          <UiField variant="settings" label="深度" :control-id="entryControlId('depth')">
            <UiNumberField
              :input-id="entryControlId('depth')"
              :model-value="entry.depth"
              aria-label="世界书条目深度"
              :min="0"
              variant="studio"
              @update:model-value="updateNumberField('depth', $event ?? 0)"
            />
          </UiField>
          <UiField variant="settings" label="概率" :control-id="entryControlId('probability')">
            <UiNumberField
              :input-id="entryControlId('probability')"
              :model-value="entry.probability ?? null"
              aria-label="世界书条目概率"
              :min="0"
              :max="100"
              variant="studio"
              @update:model-value="updateProbability"
            />
          </UiField>
        </UiFormGrid>

        <div class="lorebook-tree-branch__toggles">
          <UiCheckbox
            :model-value="entry.enabled"
            label="启用"
            @change="updateBooleanField('enabled', $event)"
          />
          <UiCheckbox
            :model-value="entry.constant"
            label="常驻"
            @change="updateBooleanField('constant', $event)"
          />
          <UiCheckbox
            :model-value="entry.selective"
            label="选择触发"
            @change="updateBooleanField('selective', $event)"
          />
          <UiCheckbox
            :model-value="entry.prevent_recursion"
            label="防递归"
            @change="updateBooleanField('prevent_recursion', $event)"
          />
          <UiCheckbox
            :model-value="entry.use_regex"
            label="用正则匹配"
            @change="updateBooleanField('use_regex', $event)"
          />
        </div>

        <div v-if="entry.children.length > 0" class="lorebook-tree-branch__children">
          <LorebookTreeBranch
            v-for="(child, childIndex) in entry.children"
            :key="child.id"
            :entry="child"
            :index="childIndex"
            :sibling-count="entry.children.length"
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
import type { LorebookEntryNode } from '@/types/characterStudio'
import type { UiSelectOption } from '@/components/ui/selectTypes'

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

function splitCsv(value: string) {
  return value
    .split(/[,，]/)
    .map(item => item.trim())
    .filter(Boolean)
}

function entryControlId(field: string) {
  return `lorebook-${props.entry.id}-${field}`
}

function updateKeys(value: string) {
  updateEntry({ keys: splitCsv(value) })
}

function updateSecondaryKeys(value: string) {
  updateEntry({ secondary_keys: splitCsv(value) })
}

function updateEntry(patch: Partial<LorebookEntryNode>) {
  emit('update:entry', { ...props.entry, ...patch })
}

function updateStringField(field: 'comment' | 'content' | 'position', value: string) {
  updateEntry({ [field]: value })
}

function updateNumberField(field: 'priority' | 'depth', value: number) {
  updateEntry({ [field]: value })
}

function updateBooleanField(
  field: 'enabled' | 'constant' | 'selective' | 'prevent_recursion' | 'use_regex',
  value: boolean
) {
  updateEntry({ [field]: value })
}

function updateProbability(value: number | null) {
  updateEntry({ probability: value ?? undefined })
}

function addChild() {
  updateEntry({
    children: [
      ...props.entry.children,
      {
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
      },
    ],
  })
}

function replaceChild(index: number, value: LorebookEntryNode) {
  updateEntry({
    children: props.entry.children.map((child, childIndex) =>
      childIndex === index ? value : child
    ),
  })
}

function removeChild(index: number) {
  updateEntry({
    children: props.entry.children.filter((_child, childIndex) => childIndex !== index),
  })
}

function moveChild(index: number, offset: -1 | 1) {
  const target = index + offset
  if (target < 0 || target >= props.entry.children.length) return
  const children = [...props.entry.children]
  const [item] = children.splice(index, 1)
  children.splice(target, 0, item!)
  updateEntry({ children })
}

function move(offset: -1 | 1) {
  emit('move', offset)
}
</script>

<style scoped>
.lorebook-tree-branch {
  --lorebook-tree-branch-border-default: var(--studio-border-default);
  --lorebook-tree-branch-surface-base: color-mix(
    in srgb,
    var(--color-surface-card) 82%,
    transparent
  );
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
