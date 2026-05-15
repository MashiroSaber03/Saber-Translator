<template>
  <div class="workbench">
    <div class="workbench-head">
      <div>
        <h3>正则脚本</h3>
        <p>统一维护提示替换、显示替换与运行位置，避免把运行时逻辑埋进大表单。</p>
      </div>
      <div class="actions">
        <button class="ghost-btn" :disabled="generating" @click="$emit('generate')">
          {{ generating ? '生成中...' : 'AI 生成脚本' }}
        </button>
        <button class="secondary-btn" @click="$emit('add')">添加脚本</button>
      </div>
    </div>

    <div v-if="scripts.length === 0" class="empty-copy">还没有正则脚本，可用于隐藏状态块、格式修复或 HTML 呈现。</div>
    <div v-else class="script-list">
      <article v-for="(script, index) in scripts" :key="script.id" class="script-card">
        <div class="card-head">
          <input class="title-input" :value="script.scriptName" type="text" @input="$emit('update:field', index, 'scriptName', ($event.target as HTMLInputElement).value)">
          <button class="danger-btn small" @click="$emit('remove', index)">删除</button>
        </div>
        <div class="grid">
          <label class="full">
            查找正则
            <input :value="script.findRegex" type="text" @input="$emit('update:field', index, 'findRegex', ($event.target as HTMLInputElement).value)">
          </label>
          <label class="full">
            替换内容
            <textarea :value="script.replaceString" rows="4" @input="$emit('update:field', index, 'replaceString', ($event.target as HTMLTextAreaElement).value)"></textarea>
          </label>
          <label>
            作用位置（Placement，逗号分隔）
            <input :value="script.placement.join(', ')" type="text" @input="$emit('update:placement', index, ($event.target as HTMLInputElement).value)">
          </label>
          <div class="toggles">
            <label><input :checked="script.markdownOnly" type="checkbox" @change="$emit('toggle:field', index, 'markdownOnly', ($event.target as HTMLInputElement).checked)"> 仅显示</label>
            <label><input :checked="script.promptOnly" type="checkbox" @change="$emit('toggle:field', index, 'promptOnly', ($event.target as HTMLInputElement).checked)"> 仅发送</label>
            <label><input :checked="script.runOnEdit" type="checkbox" @change="$emit('toggle:field', index, 'runOnEdit', ($event.target as HTMLInputElement).checked)"> 编辑时运行</label>
            <label><input :checked="script.disabled" type="checkbox" @change="$emit('toggle:field', index, 'disabled', ($event.target as HTMLInputElement).checked)"> 禁用</label>
          </div>
        </div>
      </article>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { RegexScript } from '@/types/characterStudio'

defineProps<{
  scripts: RegexScript[]
  generating: boolean
}>()

defineEmits<{
  (e: 'generate'): void
  (e: 'add'): void
  (e: 'remove', index: number): void
  (e: 'update:field', index: number, field: keyof RegexScript, value: string): void
  (e: 'update:placement', index: number, rawValue: string): void
  (e: 'toggle:field', index: number, field: keyof RegexScript, value: boolean): void
}>()
</script>

<style scoped>
.workbench {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.workbench-head,
.card-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}

.workbench-head h3 {
  margin: 0;
}

.workbench-head p {
  margin: 6px 0 0;
  color: #607794;
  font-size: 13px;
  line-height: 1.6;
}

.actions,
.toggles {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.script-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.script-card {
  border-radius: 18px;
  padding: 16px;
  background: rgba(255, 255, 255, 0.84);
  border: 1px solid rgba(25, 55, 94, 0.08);
}

.grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 14px;
}

.full {
  grid-column: 1 / -1;
}

.title-input,
input,
textarea {
  border: 1px solid rgba(28, 55, 94, 0.12);
  background: rgba(245, 249, 254, 0.92);
  border-radius: 14px;
  padding: 11px 12px;
  color: #183351;
  font-size: 13px;
}

.title-input {
  flex: 1;
  font-weight: 600;
}

textarea {
  resize: vertical;
}

label {
  display: flex;
  flex-direction: column;
  gap: 6px;
  color: #516882;
  font-size: 12px;
}

.secondary-btn,
.ghost-btn,
.danger-btn {
  border: none;
  border-radius: 12px;
  cursor: pointer;
}

.secondary-btn,
.ghost-btn {
  padding: 10px 14px;
  background: rgba(20, 56, 106, 0.07);
  color: #234977;
}

.danger-btn {
  padding: 10px 14px;
  background: rgba(217, 55, 55, 0.12);
  color: #b83535;
}

.secondary-btn:disabled,
.ghost-btn:disabled,
.danger-btn:disabled {
  opacity: 0.68;
  cursor: not-allowed;
}

.small {
  padding: 7px 10px;
  font-size: 12px;
}

.empty-copy {
  color: #6d839f;
  font-size: 13px;
}

@media (max-width: 900px) {
  .workbench-head,
  .card-head,
  .grid {
    grid-template-columns: 1fr;
    flex-direction: column;
  }
}
</style>
