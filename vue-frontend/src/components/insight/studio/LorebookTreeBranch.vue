<template>
  <div class="branch-node">
    <details class="node-details" open>
      <summary class="node-summary">
        <div class="summary-main">
          <input v-model="localEntry.comment" class="title-input" type="text">
          <div class="meta-line">
            <span>{{ localEntry.keys.length }} 个关键词</span>
            <span>优先级 {{ localEntry.priority }}</span>
            <span>{{ localEntry.position }}</span>
          </div>
        </div>
        <div class="summary-actions" @click.prevent>
          <button class="mini-btn" @click="move(-1)" :disabled="index === 0">上移</button>
          <button class="mini-btn" @click="move(1)">下移</button>
          <button class="mini-btn" @click="addChild">子项</button>
          <button class="danger-btn" @click="$emit('remove')">删除</button>
        </div>
      </summary>

      <div class="node-body">
        <div class="grid">
          <label>
            关键词（逗号分隔）
            <input :value="localEntry.keys.join(', ')" type="text" @input="updateKeys($event)">
          </label>
          <label>
            次级关键词（逗号分隔）
            <input :value="(localEntry.secondary_keys || []).join(', ')" type="text" @input="updateSecondaryKeys($event)">
          </label>
          <label class="full">
            内容
            <textarea v-model="localEntry.content" rows="4"></textarea>
          </label>
          <label>
            优先级
            <input v-model.number="localEntry.priority" type="number" min="0" step="10">
          </label>
          <label>
            注入位置
            <select v-model="localEntry.position">
              <option value="before_char">before_char</option>
              <option value="after_char">after_char</option>
              <option value="at_depth">at_depth</option>
            </select>
          </label>
          <label>
            深度
            <input v-model.number="localEntry.depth" type="number" min="0">
          </label>
          <label>
            概率
            <input v-model.number="localEntry.probability" type="number" min="0" max="100">
          </label>
        </div>

        <div class="toggles">
          <label><input v-model="localEntry.enabled" type="checkbox"> 启用</label>
          <label><input v-model="localEntry.constant" type="checkbox"> 常驻</label>
          <label><input v-model="localEntry.selective" type="checkbox"> 选择触发</label>
          <label><input v-model="localEntry.prevent_recursion" type="checkbox"> 防递归</label>
          <label><input v-model="localEntry.use_regex" type="checkbox"> 用正则匹配</label>
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
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.82);
  border: 1px solid rgba(28, 55, 94, 0.08);
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
  color: #14304c;
  font-size: 14px;
  font-weight: 600;
  padding: 0;
}

.meta-line {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 8px;
  color: #6d839f;
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
  color: #516882;
  font-size: 12px;
}

input,
textarea,
select {
  border: 1px solid rgba(28, 55, 94, 0.12);
  background: rgba(245, 249, 254, 0.92);
  border-radius: 14px;
  padding: 10px 12px;
  color: #183351;
  font-size: 13px;
}

textarea {
  resize: vertical;
}

.toggles {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 12px;
}

.mini-btn,
.danger-btn {
  border: none;
  border-radius: 12px;
  padding: 7px 10px;
  cursor: pointer;
  font-size: 12px;
}

.mini-btn {
  background: rgba(20, 56, 106, 0.07);
  color: #234977;
}

.danger-btn {
  background: rgba(217, 55, 55, 0.12);
  color: #b83535;
}

.children {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 14px;
  padding-left: 16px;
  border-left: 2px solid rgba(37, 99, 199, 0.12);
}

@media (max-width: 900px) {
  .node-summary,
  .summary-actions,
  .grid {
    grid-template-columns: 1fr;
    flex-direction: column;
  }
}
</style>
