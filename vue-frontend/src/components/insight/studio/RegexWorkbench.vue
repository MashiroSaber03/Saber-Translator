<template>
  <div class="regex-workbench">
    <div class="regex-workbench__head">
      <div class="regex-workbench__head-copy">
        <h3 class="regex-workbench__title">正则脚本</h3>
        <p class="regex-workbench__description">统一维护提示替换、显示替换与运行位置，避免把运行时逻辑埋进大表单。</p>
      </div>
      <ProductActionRow aria-label="正则脚本操作">
        <UiButton variant="secondary" :disabled="generating" @click="$emit('generate')">
          {{ generating ? '生成中...' : 'AI 生成脚本' }}
        </UiButton>
        <UiButton variant="primary" @click="$emit('add')">添加脚本</UiButton>
      </ProductActionRow>
    </div>

    <ProductEmptyState
      v-if="scripts.length === 0"
      description="可用于隐藏状态块、格式修复或 HTML 呈现。"
      icon-name="case-sensitive"
      role="note"
      size="compact"
      title="还没有正则脚本"
    />
    <div v-else class="regex-workbench__script-list">
      <ProductRecordCard v-for="(script, index) in scripts" :key="script.id" class="regex-workbench__script-card">
        <div class="regex-workbench__card-head">
          <UiInput
            class="regex-workbench__title-input"
            :model-value="script.scriptName"
            type="text"
            variant="studio"
            @update:model-value="$emit('update:field', index, 'scriptName', String($event))"
          />
          <ProductActionRow aria-label="正则脚本条目操作">
            <UiButton variant="secondary" tone="danger" @click="$emit('remove', index)" size="sm">删除</UiButton>
          </ProductActionRow>
        </div>
        <UiFormGrid class="regex-workbench__grid">
          <UiField class="regex-workbench__field--full" variant="settings" label="查找正则" :control-id="`regex-${script.id}-find`">
            <UiInput
              :id="`regex-${script.id}-find`"
              :model-value="script.findRegex"
              type="text"
              variant="studio"
              @update:model-value="$emit('update:field', index, 'findRegex', String($event))"
            />
          </UiField>
          <UiField class="regex-workbench__field--full" variant="settings" label="替换内容" :control-id="`regex-${script.id}-replace`">
            <UiTextarea
              :id="`regex-${script.id}-replace`"
              :model-value="script.replaceString"
              variant="studio"
              rows="4"
              @update:model-value="$emit('update:field', index, 'replaceString', $event)"
            />
          </UiField>
          <UiField variant="settings" label="作用位置（Placement，逗号分隔）" :control-id="`regex-${script.id}-placement`">
            <UiInput
              :id="`regex-${script.id}-placement`"
              :model-value="script.placement.join(', ')"
              type="text"
              variant="studio"
              @update:model-value="$emit('update:placement', index, String($event))"
            />
          </UiField>
          <div class="regex-workbench__toggles">
            <UiCheckbox :model-value="script.markdownOnly" label="仅显示" @change="$emit('toggle:field', index, 'markdownOnly', $event)" />
            <UiCheckbox :model-value="script.promptOnly" label="仅发送" @change="$emit('toggle:field', index, 'promptOnly', $event)" />
            <UiCheckbox :model-value="script.runOnEdit" label="编辑时运行" @change="$emit('toggle:field', index, 'runOnEdit', $event)" />
            <UiCheckbox :model-value="script.disabled" label="禁用" @change="$emit('toggle:field', index, 'disabled', $event)" />
          </div>
        </UiFormGrid>
      </ProductRecordCard>
    </div>
  </div>
</template>

<script setup lang="ts">
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import type { RegexScript } from '@/types/characterStudio'

type RegexTextField = 'scriptName' | 'findRegex' | 'replaceString'
type RegexToggleField = 'markdownOnly' | 'promptOnly' | 'runOnEdit' | 'disabled'

defineProps<{
  scripts: RegexScript[]
  generating: boolean
}>()

defineEmits<{
  (e: 'generate'): void
  (e: 'add'): void
  (e: 'remove', index: number): void
  (e: 'update:field', index: number, field: RegexTextField, value: string): void
  (e: 'update:placement', index: number, rawValue: string): void
  (e: 'toggle:field', index: number, field: RegexToggleField, value: boolean): void
}>()
</script>

<style scoped>
.regex-workbench {
  --regex-workbench-border-default: var(--studio-border-default);
  --regex-workbench-surface-base: color-mix(in srgb, var(--color-surface-card) 82%, transparent);

  display: flex;
  flex-direction: column;
  gap: 16px;
}

.regex-workbench__head {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}

.regex-workbench__card-head {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}

.regex-workbench__head-copy {
  min-width: 0;
}

.regex-workbench__title {
  margin: 0;
}

.regex-workbench__description {
  margin: 6px 0 0;
  color: var(--studio-text-muted);
  font-size: 13px;
  line-height: 1.6;
}

.regex-workbench__toggles {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.regex-workbench__script-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.regex-workbench__script-card {
  --product-record-card-background: var(--regex-workbench-surface-base);
  --product-record-card-border: var(--regex-workbench-border-default);
  --product-record-card-radius: 18px;
  --product-record-card-padding: 16px;
  --product-record-card-gap: 14px;
}

.regex-workbench__grid {
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr));
  margin-top: 14px;
  margin-bottom: 0;
}

.regex-workbench__field--full {
  grid-column: 1 / -1;
}

.regex-workbench__title-input {
  flex: 1 1 220px;
  min-width: 0;
  font-weight: 600;
}

@media (--breakpoint-lg-down) {
  .regex-workbench__head,
  .regex-workbench__card-head {
    flex-direction: column;
  }

  .regex-workbench__grid {
    grid-template-columns: 1fr;
  }
}
</style>
