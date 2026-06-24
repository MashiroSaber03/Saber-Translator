<template>
  <div class="branch-node">
    <details class="node-details" open>
      <summary class="node-summary">
        <div class="summary-main">
          <UiInput v-model="localEntry.comment" class="title-input" type="text" />
          <div class="meta-line">
            <span>{{ localEntry.keys.length }} 个关键词</span>
            <span>优先级 {{ localEntry.priority }}</span>
            <span>{{ localEntry.position }}</span>
          </div>
        </div>
        <div class="summary-actions" @click.prevent>
          <UiButton variant="toolbar" class="mini-btn" @click="move(-1)" :disabled="index === 0">上移</UiButton>
          <UiButton variant="toolbar" class="mini-btn" @click="move(1)">下移</UiButton>
          <UiButton variant="toolbar" class="mini-btn" @click="addChild">子项</UiButton>
          <UiButton variant="toolbar" class="action-danger" @click="$emit('remove')">删除</UiButton>
        </div>
      </summary>

      <div class="node-body">
        <div class="grid">
          <label>
            关键词（逗号分隔）
            <UiInput :value="localEntry.keys.join(', ')" type="text" @input="updateKeys($event)" />
          </label>
          <label>
            次级关键词（逗号分隔）
            <UiInput :value="(localEntry.secondary_keys || []).join(', ')" type="text" @input="updateSecondaryKeys($event)" />
          </label>
          <label class="full">
            内容
            <UiTextarea v-model="localEntry.content" rows="4" />
          </label>
          <label>
            优先级
            <UiInput v-model.number="localEntry.priority" type="number" min="0" step="10" />
          </label>
          <label>
            注入位置
            <UiSelect v-model="localEntry.position">
              <option value="before_char">before_char</option>
              <option value="after_char">after_char</option>
              <option value="at_depth">at_depth</option>
            </UiSelect>
          </label>
          <label>
            深度
            <UiInput v-model.number="localEntry.depth" type="number" min="0" />
          </label>
          <label>
            概率
            <UiInput v-model.number="localEntry.probability" type="number" min="0" max="100" />
          </label>
        </div>

        <div class="toggles">
          <label><UiInput v-model="localEntry.enabled" type="checkbox" /> 启用</label>
          <label><UiInput v-model="localEntry.constant" type="checkbox" /> 常驻</label>
          <label><UiInput v-model="localEntry.selective" type="checkbox" /> 选择触发</label>
          <label><UiInput v-model="localEntry.prevent_recursion" type="checkbox" /> 防递归</label>
          <label><UiInput v-model="localEntry.use_regex" type="checkbox" /> 用正则匹配</label>
        </div>

        <div v-if="localEntry.children.length > 0" class="children">
          <LorebookTreeBranch
            v-for="(child, childIndex) in localEntry.children"
            :key="child.id"
            :entry="child"
            :index="childIndex"
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
import UiSelect from '@/components/ui/UiSelect.vue'
import UiButton from '@/components/ui/UiButton.vue'
import { nextTick, ref, watch } from 'vue'
import type { LorebookEntryNode } from '@/types/characterStudio'

const props = defineProps<{
  entry: LorebookEntryNode
  index: number
}>()

const emit = defineEmits<{
  (e: 'update:entry', value: LorebookEntryNode): void
  (e: 'remove'): void
  (e: 'move', offset: -1 | 1): void
}>()

const localEntry = ref<LorebookEntryNode>(JSON.parse(JSON.stringify(props.entry)) as LorebookEntryNode)
let syncing = false

watch(() => props.entry, value => {
  syncing = true
  localEntry.value = JSON.parse(JSON.stringify(value)) as LorebookEntryNode
  void nextTick(() => {
    syncing = false
  })
}, { deep: true, immediate: true })

watch(localEntry, value => {
  if (syncing) return
  emit('update:entry', JSON.parse(JSON.stringify(value)) as LorebookEntryNode)
}, { deep: true })

function splitCsv(value: string) {
  return value.split(/[,，]/).map(item => item.trim()).filter(Boolean)
}

function updateKeys(event: Event) {
  const target = event.target as HTMLInputElement
  localEntry.value.keys = splitCsv(target.value)
}

function updateSecondaryKeys(event: Event) {
  const target = event.target as HTMLInputElement
  localEntry.value.secondary_keys = splitCsv(target.value)
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
.branch-node {
  --lorebook-tree-branch-border-default: rgba(37, 99, 199, .12);
  --lorebook-tree-branch-surface-base: rgba(255, 255, 255, .82);
  --lorebook-tree-branch-text-primary: #14304c;
  --lorebook-tree-branch-text-secondary: #516882;
  --ui-input-border: 1px solid var(--studio-border-strong);
  --ui-input-background: var(--studio-surface-soft);
  --ui-input-radius: 14px;
  --ui-input-padding: 10px 12px;
  --ui-input-color: var(--studio-text-strong);
  --ui-input-font-size: 13px;
  --ui-select-border: 1px solid var(--studio-border-strong);
  --ui-select-background: var(--studio-surface-soft);
  --ui-select-radius: 14px;
  --ui-select-padding: 10px 12px;
  --ui-select-color: var(--studio-text-strong);
  --ui-select-font-size: 13px;
  --ui-textarea-border: 1px solid var(--studio-border-strong);
  --ui-textarea-background: var(--studio-surface-soft);
  --ui-textarea-radius: 14px;
  --ui-textarea-padding: 10px 12px;
  --ui-textarea-color: var(--studio-text-strong);
  --ui-textarea-font-size: 13px;

  border-radius: 18px;
  background: var(--lorebook-tree-branch-surface-base);
  border: 1px solid var(--studio-border-default);
}

.node-details {
  border-radius: 18px;
}

.node-summary {
  list-style: none;
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 16px;
  cursor: pointer;
}

.node-summary::-webkit-details-marker {
  display: none;
}

.summary-main {
  min-width: 0;
  flex: 1;
}

.title-input {
  width: 100%;
  border: none;
  background: transparent;
  color: var(--lorebook-tree-branch-text-primary);
  font-size: 14px;
  font-weight: 600;
  padding: 0;
}

.meta-line {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 8px;
  color: var(--studio-text-subtle);
  font-size: 11px;
}

.summary-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: flex-start;
}

.node-body {
  padding: 0 16px 16px;
}

.grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.full {
  grid-column: 1 / -1;
}

label {
  display: flex;
  flex-direction: column;
  gap: 6px;
  color: var(--lorebook-tree-branch-text-secondary);
  font-size: 12px;
}

.toggles {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 12px;
}

.mini-btn,
.action-danger {
  border: none;
  border-radius: 12px;
  padding: 7px 10px;
  cursor: pointer;
  font-size: 12px;
}

.mini-btn {
  background: var(--studio-surface-muted);
  color: var(--studio-text-default);
}

.action-danger {
  background: var(--color-surface-danger-soft);
  color: var(--studio-text-danger);
}

.children {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 14px;
  padding-left: 16px;
  border-left: 2px solid var(--lorebook-tree-branch-border-default);
}

@media (--breakpoint-lg-down) {
  .node-summary,
  .summary-actions,
  .grid {
    grid-template-columns: 1fr;
    flex-direction: column;
  }
}
</style>
