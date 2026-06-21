<template>
  <div class="workbench">
    <div class="workbench-head">
      <div>
        <h3>正则脚本</h3>
        <p>统一维护提示替换、显示替换与运行位置，避免把运行时逻辑埋进大表单。</p>
      </div>
      <div class="actions">
        <UiButton variant="toolbar" class="action-ghost" :disabled="generating" @click="$emit('generate')">
          {{ generating ? '生成中...' : 'AI 生成脚本' }}
        </UiButton>
        <UiButton variant="toolbar" class="action-secondary" @click="$emit('add')">添加脚本</UiButton>
      </div>
    </div>

    <div v-if="scripts.length === 0" class="empty-copy">还没有正则脚本，可用于隐藏状态块、格式修复或 HTML 呈现。</div>
    <div v-else class="script-list">
      <article v-for="(script, index) in scripts" :key="script.id" class="script-card">
        <div class="card-head">
          <UiInput class="title-input" :value="script.scriptName" type="text" @input="$emit('update:field', index, 'scriptName', ($event.target as HTMLInputElement).value)" />
          <UiButton variant="toolbar" class="action-danger" @click="$emit('remove', index)" size="sm">删除</UiButton>
        </div>
        <div class="grid">
          <label class="full">
            查找正则
            <UiInput :value="script.findRegex" type="text" @input="$emit('update:field', index, 'findRegex', ($event.target as HTMLInputElement).value)" />
          </label>
          <label class="full">
            替换内容
            <UiTextarea :value="script.replaceString" rows="4" @input="$emit('update:field', index, 'replaceString', ($event.target as HTMLTextAreaElement).value)" />
          </label>
          <label>
            作用位置（Placement，逗号分隔）
            <UiInput :value="script.placement.join(', ')" type="text" @input="$emit('update:placement', index, ($event.target as HTMLInputElement).value)" />
          </label>
          <div class="toggles">
            <label><UiInput :checked="script.markdownOnly" type="checkbox" @change="$emit('toggle:field', index, 'markdownOnly', ($event.target as HTMLInputElement).checked)" /> 仅显示</label>
            <label><UiInput :checked="script.promptOnly" type="checkbox" @change="$emit('toggle:field', index, 'promptOnly', ($event.target as HTMLInputElement).checked)" /> 仅发送</label>
            <label><UiInput :checked="script.runOnEdit" type="checkbox" @change="$emit('toggle:field', index, 'runOnEdit', ($event.target as HTMLInputElement).checked)" /> 编辑时运行</label>
            <label><UiInput :checked="script.disabled" type="checkbox" @change="$emit('toggle:field', index, 'disabled', ($event.target as HTMLInputElement).checked)" /> 禁用</label>
          </div>
        </div>
      </article>
    </div>
  </div>
</template>

<script setup lang="ts">
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
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
  color: var(--color-text-studio-muted);
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
  background: var(--regex-workbench-surface-base);
  border: 1px solid var(--regex-workbench-border-default);
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
  border: 1px solid var(--color-border-studio-strong);
  background: var(--color-surface-studio-soft);
  border-radius: 14px;
  padding: 11px 12px;
  color: var(--color-text-studio-strong);
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
  color: var(--regex-workbench-text-primary);
  font-size: 12px;
}

.action-secondary,
.action-ghost,
.action-danger {
  border: none;
  border-radius: 12px;
  cursor: pointer;
}

.action-secondary,
.action-ghost {
  padding: 10px 14px;
  background: var(--color-surface-studio-muted);
  color: var(--color-text-studio);
}

.action-danger {
  padding: 10px 14px;
  background: var(--color-surface-danger-soft);
  color: var(--color-text-studio-danger);
}

.action-secondary:disabled,
.action-ghost:disabled,
.action-danger:disabled {
  opacity: 0.68;
  cursor: not-allowed;
}

.small {
  padding: 7px 10px;
  font-size: 12px;
}

.empty-copy {
  color: var(--color-text-studio-subtle);
  font-size: 13px;
}

@media (--breakpoint-lg-down) {
  .workbench-head,
  .card-head,
  .grid {
    grid-template-columns: 1fr;
    flex-direction: column;
  }
}
</style>
