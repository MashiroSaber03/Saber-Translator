<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
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
        <UiCheckbox
          :model-value="Object.values(applyOptions).every(Boolean)"
          label="全选"
          @change="$emit('toggleSelectAll')"
        />
      </div>
      <hr>
      <div class="apply-option">
        <UiCheckbox :model-value="applyOptions.fontSize" label="字号" @change="$emit('updateOption', 'fontSize', $event)" />
      </div>
      <div class="apply-option">
        <UiCheckbox :model-value="applyOptions.fontFamily" label="字体" @change="$emit('updateOption', 'fontFamily', $event)" />
      </div>
      <div class="apply-option">
        <UiCheckbox :model-value="applyOptions.layoutDirection" label="排版方向" @change="$emit('updateOption', 'layoutDirection', $event)" />
      </div>
      <div class="apply-option">
        <UiCheckbox :model-value="applyOptions.lineSpacing" label="行间距" @change="$emit('updateOption', 'lineSpacing', $event)" />
      </div>
      <div class="apply-option">
        <UiCheckbox :model-value="applyOptions.textAlign" label="对齐方式" @change="$emit('updateOption', 'textAlign', $event)" />
      </div>
      <div class="apply-option">
        <UiCheckbox :model-value="applyOptions.textColor" label="文字颜色" @change="$emit('updateOption', 'textColor', $event)" />
      </div>
      <div class="apply-option">
        <UiCheckbox :model-value="applyOptions.fillColor" label="填充颜色" @change="$emit('updateOption', 'fillColor', $event)" />
      </div>
      <div class="apply-option">
        <UiCheckbox :model-value="applyOptions.strokeEnabled" label="描边开关" @change="$emit('updateOption', 'strokeEnabled', $event)" />
      </div>
      <div class="apply-option">
        <UiCheckbox :model-value="applyOptions.strokeColor" label="描边颜色" @change="$emit('updateOption', 'strokeColor', $event)" />
      </div>
      <div class="apply-option">
        <UiCheckbox :model-value="applyOptions.strokeWidth" label="描边宽度" @change="$emit('updateOption', 'strokeWidth', $event)" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.settings-sidebar__apply-group {
  --settings-sidebar-apply-button-start: #4b89d0;
  --settings-sidebar-apply-button-end: #316fb6;
  --settings-sidebar-apply-button-disabled: #c2c9d4;
  --settings-sidebar-apply-button-hover-start: #3f7bc4;
  --settings-sidebar-apply-button-hover-end: #2b64a9;
  --settings-sidebar-apply-options-end: #285d99;
  --settings-sidebar-apply-button-divider: rgba(255, 255, 255, .24);
  --settings-sidebar-apply-menu-border: #d7e2f2;
  --settings-sidebar-apply-menu-divider: #e3ebf6;
  --settings-sidebar-apply-menu-shadow: rgba(22, 37, 58, .16);
  --settings-sidebar-apply-option-text: #405473;
  --settings-sidebar-apply-option-hover-text: #2b5f9d;

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
  background: linear-gradient(135deg, var(--settings-sidebar-apply-button-start) 0%, var(--settings-sidebar-apply-button-end) 100%);
  color: var(--color-text-inverse);
  font-weight: 600;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.settings-sidebar__apply-button:disabled {
  background: var(--settings-sidebar-apply-button-disabled);
  cursor: not-allowed;
}

.settings-sidebar__apply-button:hover:not(:disabled) {
  background: linear-gradient(135deg, var(--settings-sidebar-apply-button-hover-start) 0%, var(--settings-sidebar-apply-button-hover-end) 100%);
}

.settings-sidebar__apply-options-button {
  width: 38px;
  border: none;
  border-left: 1px solid var(--settings-sidebar-apply-button-divider);
  border-radius: 0 8px 8px 0;
  background: linear-gradient(135deg, var(--settings-sidebar-apply-button-end) 0%, var(--settings-sidebar-apply-options-end) 100%);
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
  border: 1px solid var(--settings-sidebar-apply-menu-border);
  border-radius: 10px;
  background: var(--color-surface-base);
  box-shadow: 0 12px 24px var(--settings-sidebar-apply-menu-shadow);
}

.apply-option {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 26px;
  color: var(--settings-sidebar-apply-option-text);
  font-size: 13px;
  cursor: pointer;
}

.apply-option:hover {
  color: var(--settings-sidebar-apply-option-hover-text);
}

.apply-options-dropdown hr {
  margin: 6px 0;
  border: none;
  border-top: 1px solid var(--settings-sidebar-apply-menu-divider);
}
</style>
