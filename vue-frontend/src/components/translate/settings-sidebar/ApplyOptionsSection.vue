<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiInput from '@/components/ui/UiInput.vue'
import type { ApplySettingsOptions } from '../useSettingsSidebar'

defineProps<{
  applyOptions: ApplySettingsOptions
  hasImages: boolean
  showApplyOptions: boolean
}>()

defineEmits<{
  (event: 'apply'): void
  (event: 'toggleOptions'): void
  (event: 'toggleSelectAll'): void
  (event: 'updateOption', key: keyof ApplySettingsOptions, value: boolean): void
}>()
</script>

<template>
  <div class="settings-sidebar__apply-group">
    <UiButton
      variant="toolbar"
      type="button"
      class="settings-sidebar__apply-button"
      :disabled="!hasImages"
      @click="$emit('apply')"
    >
      应用到全部
    </UiButton>
    <UiButton
      variant="toolbar"
      type="button"
      class="settings-sidebar__apply-options-button"
      title="选择要应用的参数"
      @click="$emit('toggleOptions')"
    >
      ⚙️
    </UiButton>

    <div v-if="showApplyOptions" class="apply-options-dropdown">
      <div class="apply-option">
        <UiInput
          id="apply_selectAll"
          class="apply-option-checkbox"
          type="checkbox"
          :checked="Object.values(applyOptions).every(Boolean)"
          @change="$emit('toggleSelectAll')"
        />
        <label for="apply_selectAll">全选</label>
      </div>
      <hr>
      <div class="apply-option">
        <UiInput
          id="apply_fontSize"
          class="apply-option-checkbox"
          type="checkbox"
          :model-value="applyOptions.fontSize"
          @update:model-value="$emit('updateOption', 'fontSize', Boolean($event))"
        />
        <label for="apply_fontSize">字号</label>
      </div>
      <div class="apply-option">
        <UiInput
          id="apply_fontFamily"
          class="apply-option-checkbox"
          type="checkbox"
          :model-value="applyOptions.fontFamily"
          @update:model-value="$emit('updateOption', 'fontFamily', Boolean($event))"
        />
        <label for="apply_fontFamily">字体</label>
      </div>
      <div class="apply-option">
        <UiInput
          id="apply_layoutDirection"
          class="apply-option-checkbox"
          type="checkbox"
          :model-value="applyOptions.layoutDirection"
          @update:model-value="$emit('updateOption', 'layoutDirection', Boolean($event))"
        />
        <label for="apply_layoutDirection">排版方向</label>
      </div>
      <div class="apply-option">
        <UiInput
          id="apply_lineSpacing"
          class="apply-option-checkbox"
          type="checkbox"
          :model-value="applyOptions.lineSpacing"
          @update:model-value="$emit('updateOption', 'lineSpacing', Boolean($event))"
        />
        <label for="apply_lineSpacing">行间距</label>
      </div>
      <div class="apply-option">
        <UiInput
          id="apply_textAlign"
          class="apply-option-checkbox"
          type="checkbox"
          :model-value="applyOptions.textAlign"
          @update:model-value="$emit('updateOption', 'textAlign', Boolean($event))"
        />
        <label for="apply_textAlign">对齐方式</label>
      </div>
      <div class="apply-option">
        <UiInput
          id="apply_textColor"
          class="apply-option-checkbox"
          type="checkbox"
          :model-value="applyOptions.textColor"
          @update:model-value="$emit('updateOption', 'textColor', Boolean($event))"
        />
        <label for="apply_textColor">文字颜色</label>
      </div>
      <div class="apply-option">
        <UiInput
          id="apply_fillColor"
          class="apply-option-checkbox"
          type="checkbox"
          :model-value="applyOptions.fillColor"
          @update:model-value="$emit('updateOption', 'fillColor', Boolean($event))"
        />
        <label for="apply_fillColor">填充颜色</label>
      </div>
      <div class="apply-option">
        <UiInput
          id="apply_strokeEnabled"
          class="apply-option-checkbox"
          type="checkbox"
          :model-value="applyOptions.strokeEnabled"
          @update:model-value="$emit('updateOption', 'strokeEnabled', Boolean($event))"
        />
        <label for="apply_strokeEnabled">描边开关</label>
      </div>
      <div class="apply-option">
        <UiInput
          id="apply_strokeColor"
          class="apply-option-checkbox"
          type="checkbox"
          :model-value="applyOptions.strokeColor"
          @update:model-value="$emit('updateOption', 'strokeColor', Boolean($event))"
        />
        <label for="apply_strokeColor">描边颜色</label>
      </div>
      <div class="apply-option">
        <UiInput
          id="apply_strokeWidth"
          class="apply-option-checkbox"
          type="checkbox"
          :model-value="applyOptions.strokeWidth"
          @update:model-value="$emit('updateOption', 'strokeWidth', Boolean($event))"
        />
        <label for="apply_strokeWidth">描边宽度</label>
      </div>
    </div>
  </div>
</template>

<style scoped>
.settings-sidebar__apply-group {
  display: flex;
  align-items: stretch;
  position: relative;
  width: 100%;
  height: 38px;
  margin-top: 8px;
}

.settings-sidebar__apply-button {
  flex: 1;
  min-width: 0;
  margin: 0;
  border: none;
  border-radius: 8px 0 0 8px;
  background: linear-gradient(135deg, var(--settings-sidebar-apply-actions-surface-base) 0%, var(--settings-sidebar-apply-actions-surface-raised) 100%);
  color: var(--color-text-inverse);
  font-weight: 600;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.settings-sidebar__apply-button:disabled {
  background: var(--settings-sidebar-apply-actions-surface-muted);
  cursor: not-allowed;
}

.settings-sidebar__apply-button:hover:not(:disabled) {
  background: linear-gradient(135deg, var(--settings-sidebar-apply-actions-surface-subtle) 0%, var(--settings-sidebar-apply-actions-surface-hover) 100%);
}

.settings-sidebar__apply-options-button {
  width: 38px;
  border: none;
  border-left: 1px solid var(--settings-sidebar-apply-actions-border-default);
  border-radius: 0 8px 8px 0;
  background: linear-gradient(135deg, var(--settings-sidebar-apply-actions-surface-raised) 0%, var(--settings-sidebar-apply-actions-surface-active) 100%);
  color: var(--color-text-inverse);
  font-size: 14px;
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.apply-options-dropdown {
  position: absolute;
  inset: auto 0 calc(100% + 6px) 0;
  z-index: var(--z-overlay);
  max-height: 260px;
  padding: 10px;
  overflow-y: auto;
  border: 1px solid var(--settings-sidebar-apply-actions-border-strong);
  border-radius: 10px;
  background: var(--color-surface-base);
  box-shadow: 0 12px 24px var(--settings-sidebar-apply-actions-shadow-default);
}

.apply-option {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 26px;
  color: var(--settings-sidebar-apply-actions-text-secondary);
  font-size: 13px;
  cursor: pointer;
}

.apply-option-checkbox {
  width: 14px;
  height: 14px;
  margin: 0;
  accent-color: var(--settings-sidebar-apply-actions-surface-base);
}

.apply-option:hover {
  color: var(--settings-sidebar-apply-actions-text-muted);
}

.apply-options-dropdown hr {
  margin: 6px 0;
  border: none;
  border-top: 1px solid var(--settings-sidebar-apply-actions-border-muted);
}
</style>
