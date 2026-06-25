<template>
  <div class="workbench">
    <div class="workbench-head">
      <div>
        <h3>状态任务</h3>
        <p>用于初始化变量或挂载受控运行时逻辑；在当前预览里，任务间隔按事件触发次数计算。</p>
      </div>
      <div class="actions">
        <UiButton variant="toolbar" class="action-ghost" :disabled="generating" @click="$emit('generate')">
          {{ generating ? '生成中...' : 'AI 生成任务' }}
        </UiButton>
        <UiButton variant="toolbar" class="action-secondary" @click="$emit('add')">添加任务</UiButton>
      </div>
    </div>

    <div v-if="tasks.length === 0" class="empty-copy">还没有状态任务，建议至少保留一个初始化任务。</div>
    <div v-else class="task-list">
      <article v-for="(task, index) in tasks" :key="task.id" class="task-card">
        <div class="card-head">
          <UiInput class="title-input" :value="task.name" type="text" @input="$emit('update:field', index, 'name', ($event.target as HTMLInputElement).value)" />
          <UiButton variant="toolbar" class="action-danger" @click="$emit('remove', index)" size="sm">删除</UiButton>
        </div>
        <div class="grid">
          <label>
            触发时机
            <UiSelect :model-value="task.triggerTiming" @change="$emit('update:field', index, 'triggerTiming', $event)">
              <option value="initialization">初始化</option>
              <option value="message_received">收到消息</option>
              <option value="message_sent">发送消息</option>
            </UiSelect>
          </label>
          <label>
            间隔（事件次数）
            <UiInput :value="String(task.interval)" type="number" min="0" @input="$emit('update:number', index, 'interval', Number(($event.target as HTMLInputElement).value || 0))" />
          </label>
          <label class="full">
            任务脚本
            <UiTextarea :value="task.commands" rows="6" @input="$emit('update:field', index, 'commands', ($event.target as HTMLTextAreaElement).value)" />
          </label>
          <div class="toggles full">
            <UiCheckbox :model-value="task.disabled" label="禁用任务" @change="$emit('toggle:field', index, 'disabled', $event)" />
          </div>
        </div>
      </article>
    </div>
  </div>
</template>

<script setup lang="ts">
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import type { StateTask } from '@/types/characterStudio'

defineProps<{
  tasks: StateTask[]
  generating: boolean
}>()

defineEmits<{
  (e: 'generate'): void
  (e: 'add'): void
  (e: 'remove', index: number): void
  (e: 'update:field', index: number, field: keyof StateTask, value: string): void
  (e: 'update:number', index: number, field: keyof StateTask, value: number): void
  (e: 'toggle:field', index: number, field: keyof StateTask, value: boolean): void
}>()
</script>

<style scoped>
.workbench {
  --task-workbench-border-default: rgba(25, 55, 94, .08);
  --task-workbench-surface-base: rgba(255, 255, 255, .84);
  --task-workbench-text-primary: #516882;
  --ui-input-border: 1px solid var(--studio-border-strong);
  --ui-input-background: var(--studio-surface-soft);
  --ui-input-radius: 14px;
  --ui-input-padding: 11px 12px;
  --ui-input-color: var(--studio-text-strong);
  --ui-input-font-size: 13px;
  --ui-select-border: 1px solid var(--studio-border-strong);
  --ui-select-background: var(--studio-surface-soft);
  --ui-select-radius: 14px;
  --ui-select-padding: 11px 12px;
  --ui-select-color: var(--studio-text-strong);
  --ui-select-font-size: 13px;
  --ui-textarea-border: 1px solid var(--studio-border-strong);
  --ui-textarea-background: var(--studio-surface-soft);
  --ui-textarea-radius: 14px;
  --ui-textarea-padding: 11px 12px;
  --ui-textarea-color: var(--studio-text-strong);
  --ui-textarea-font-size: 13px;

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
  color: var(--studio-text-muted);
  font-size: 13px;
  line-height: 1.6;
}

.actions,
.toggles {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.task-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.task-card {
  border-radius: 18px;
  padding: 16px;
  background: var(--task-workbench-surface-base);
  border: 1px solid var(--task-workbench-border-default);
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

.title-input {
  flex: 1;
  font-weight: 600;
}

label {
  display: flex;
  flex-direction: column;
  gap: 6px;
  color: var(--task-workbench-text-primary);
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
  background: var(--studio-surface-muted);
  color: var(--studio-text-default);
}

.action-danger {
  padding: 10px 14px;
  background: var(--color-surface-danger-soft);
  color: var(--studio-text-danger);
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
  color: var(--studio-text-subtle);
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
