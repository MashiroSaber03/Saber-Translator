<template>
  <div class="workbench">
    <div class="workbench-head">
      <div>
        <h3>状态任务</h3>
        <p>用于初始化变量或挂载受控运行时逻辑；在当前预览里，任务间隔按事件触发次数计算。</p>
      </div>
      <div class="actions">
        <button class="ghost-btn" @click="$emit('generate')">AI 生成任务</button>
        <button class="secondary-btn" @click="$emit('add')">添加任务</button>
      </div>
    </div>

    <div v-if="tasks.length === 0" class="empty-copy">还没有状态任务，建议至少保留一个初始化任务。</div>
    <div v-else class="task-list">
      <article v-for="(task, index) in tasks" :key="task.id" class="task-card">
        <div class="card-head">
          <input class="title-input" :value="task.name" type="text" @input="$emit('update:field', index, 'name', ($event.target as HTMLInputElement).value)">
          <button class="danger-btn small" @click="$emit('remove', index)">删除</button>
        </div>
        <div class="grid">
          <label>
            触发时机
            <select :value="task.triggerTiming" @change="$emit('update:field', index, 'triggerTiming', ($event.target as HTMLSelectElement).value)">
              <option value="initialization">初始化</option>
              <option value="message_received">收到消息</option>
              <option value="message_sent">发送消息</option>
            </select>
          </label>
          <label>
            间隔（事件次数）
            <input :value="String(task.interval)" type="number" min="0" @input="$emit('update:number', index, 'interval', Number(($event.target as HTMLInputElement).value || 0))">
          </label>
          <label class="full">
            任务脚本
            <textarea :value="task.commands" rows="6" @input="$emit('update:field', index, 'commands', ($event.target as HTMLTextAreaElement).value)"></textarea>
          </label>
          <div class="toggles full">
            <label><input :checked="task.disabled" type="checkbox" @change="$emit('toggle:field', index, 'disabled', ($event.target as HTMLInputElement).checked)"> 禁用任务</label>
          </div>
        </div>
      </article>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { StateTask } from '@/types/characterStudio'

defineProps<{
  tasks: StateTask[]
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

.task-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.task-card {
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
textarea,
select {
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
