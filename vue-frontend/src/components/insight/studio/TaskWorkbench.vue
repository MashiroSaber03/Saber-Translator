<template>
  <div class="task-workbench">
    <div class="task-workbench__head">
      <div class="task-workbench__head-copy">
        <h3 class="task-workbench__title">状态任务</h3>
        <p class="task-workbench__description">用于初始化变量或挂载受控运行时逻辑；在当前预览里，任务间隔按事件触发次数计算。</p>
      </div>
      <ProductActionRow appearance="accent" aria-label="状态任务操作">
        <UiButton variant="secondary" :disabled="generating" @click="$emit('generate')">
          {{ generating ? '生成中...' : 'AI 生成任务' }}
        </UiButton>
        <UiButton variant="secondary" @click="$emit('add')">添加任务</UiButton>
      </ProductActionRow>
    </div>

    <ProductEmptyState
      v-if="tasks.length === 0"
      description="建议至少保留一个初始化任务。"
      icon-name="list"
      role="note"
      size="compact"
      title="还没有状态任务"
    />
    <div v-else class="task-workbench__task-list">
      <ProductRecordCard v-for="(task, index) in tasks" :key="task.id" class="task-workbench__task-card">
        <div class="task-workbench__card-head">
          <UiInput
            class="task-workbench__title-input"
            :model-value="task.name"
            type="text"
            variant="studio"
            @update:model-value="$emit('update:field', index, 'name', String($event))"
          />
          <ProductActionRow appearance="accent" aria-label="状态任务条目操作">
            <UiButton variant="secondary" tone="danger" @click="$emit('remove', index)" size="sm">删除</UiButton>
          </ProductActionRow>
        </div>
        <UiFormGrid class="task-workbench__grid">
          <UiField variant="settings" label="触发时机" :control-id="`task-${task.id}-trigger`">
            <UiSelect
              :id="`task-${task.id}-trigger`"
              :model-value="task.triggerTiming"
              :options="TASK_TRIGGER_OPTIONS"
              variant="studio"
              @change="$emit('update:field', index, 'triggerTiming', String($event))"
            />
          </UiField>
          <UiField variant="settings" label="间隔（事件次数）" :control-id="`task-${task.id}-interval`">
            <UiNumberField
              :input-id="`task-${task.id}-interval`"
              :model-value="task.interval"
              :min="0"
              size="sm"
              variant="studio"
              @change="value => $emit('update:number', index, 'interval', value ?? 0)"
            />
          </UiField>
          <UiField class="task-workbench__field--full" variant="settings" label="任务脚本" :control-id="`task-${task.id}-commands`">
            <UiTextarea
              :id="`task-${task.id}-commands`"
              :model-value="task.commands"
              variant="studio"
              rows="6"
              @update:model-value="$emit('update:field', index, 'commands', $event)"
            />
          </UiField>
          <div class="task-workbench__toggles task-workbench__field--full">
            <UiCheckbox :model-value="task.disabled" label="禁用任务" @change="$emit('toggle:field', index, 'disabled', $event)" />
          </div>
        </UiFormGrid>
      </ProductRecordCard>
    </div>
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
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import type { StateTask } from '@/types/characterStudio'
import type { UiSelectOption } from '@/components/ui/selectTypes'

type StateTaskTextField = 'name' | 'triggerTiming' | 'commands'
type StateTaskNumberField = 'interval'
type StateTaskToggleField = 'disabled'

const TASK_TRIGGER_OPTIONS: UiSelectOption[] = [
  { label: '初始化', value: 'initialization' },
  { label: '收到消息', value: 'message_received' },
  { label: '发送消息', value: 'message_sent' },
]

defineProps<{
  tasks: StateTask[]
  generating: boolean
}>()

defineEmits<{
  (e: 'generate'): void
  (e: 'add'): void
  (e: 'remove', index: number): void
  (e: 'update:field', index: number, field: StateTaskTextField, value: string): void
  (e: 'update:number', index: number, field: StateTaskNumberField, value: number): void
  (e: 'toggle:field', index: number, field: StateTaskToggleField, value: boolean): void
}>()
</script>

<style scoped>
.task-workbench {
  --task-workbench-border-default: var(--studio-border-default);
  --task-workbench-surface-base: color-mix(in srgb, var(--color-surface-card) 82%, transparent);

  display: flex;
  flex-direction: column;
  gap: 16px;
}

.task-workbench__head {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}

.task-workbench__card-head {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}

.task-workbench__head-copy {
  min-width: 0;
}

.task-workbench__title {
  margin: 0;
}

.task-workbench__description {
  margin: 6px 0 0;
  color: var(--studio-text-muted);
  font-size: 13px;
  line-height: 1.6;
}

.task-workbench__toggles {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.task-workbench__task-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.task-workbench__task-card {
  --product-record-card-background: var(--task-workbench-surface-base);
  --product-record-card-border: var(--task-workbench-border-default);
  --product-record-card-radius: 18px;
  --product-record-card-padding: 16px;
  --product-record-card-gap: 14px;
}

.task-workbench__grid {
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr));
  margin-top: 14px;
  margin-bottom: 0;
}

.task-workbench__field--full {
  grid-column: 1 / -1;
}

.task-workbench__title-input {
  flex: 1 1 220px;
  min-width: 0;
  font-weight: 600;
}

@media (--breakpoint-lg-down) {
  .task-workbench__head,
  .task-workbench__card-head {
    flex-direction: column;
  }

  .task-workbench__grid {
    grid-template-columns: 1fr;
  }
}
</style>
