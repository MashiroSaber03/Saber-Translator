<script setup lang="ts">
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import type { ApplySettingsOptions } from '../useSettingsSidebar'

const APPLY_OPTIONS_MENU_ID = 'apply-options-section-menu'
const APPLY_OPTION_ITEMS = [
  { key: 'fontSize', label: '字号' },
  { key: 'fontFamily', label: '字体' },
  { key: 'layoutDirection', label: '排版方向' },
  { key: 'lineSpacing', label: '行间距' },
  { key: 'textAlign', label: '对齐方式' },
  { key: 'textColor', label: '文字颜色' },
  { key: 'fillColor', label: '填充颜色' },
  { key: 'strokeEnabled', label: '描边开关' },
  { key: 'strokeColor', label: '描边颜色' },
  { key: 'strokeWidth', label: '描边宽度' },
] satisfies ReadonlyArray<{
  key: keyof ApplySettingsOptions
  label: string
}>

withDefaults(defineProps<{
  applyOptions: ApplySettingsOptions
  disabled?: boolean
  hasImages: boolean
  showApplyOptions: boolean
}>(), {
  disabled: false,
})

defineEmits<{
  (event: 'apply'): void
  (event: 'toggleOptions'): void
  (event: 'toggleSelectAll'): void
  (event: 'updateOption', key: keyof ApplySettingsOptions, value: boolean): void
}>()
</script>

<template>
  <div class="apply-options-section">
    <ProductActionRow
      class="apply-options-section__actions"
      aria-label="批量应用文字设置"
      justify="between"
    >
      <UiButton
        variant="primary"
        type="button"
        class="apply-options-section__action"
        block
        :disabled="disabled || !hasImages"
        @click="$emit('apply')"
      >
        应用到全部
      </UiButton>
      <UiIconButton
        variant="soft"
        type="button"
        class="apply-options-section__options-action"
        label="选择要应用的参数"
        title="选择要应用的参数"
        aria-haspopup="true"
        :aria-expanded="showApplyOptions ? 'true' : 'false'"
        :aria-controls="showApplyOptions ? APPLY_OPTIONS_MENU_ID : undefined"
        :disabled="disabled"
        @click="$emit('toggleOptions')"
      >
        <UiIcon name="settings" size="15" />
      </UiIconButton>
    </ProductActionRow>

    <div
      v-if="showApplyOptions"
      :id="APPLY_OPTIONS_MENU_ID"
      class="apply-options-section__menu"
      role="group"
      aria-label="可应用的文字设置"
    >
      <div class="apply-options-section__option">
        <UiCheckbox
          :model-value="Object.values(applyOptions).every(Boolean)"
          :disabled="disabled"
          label="全选"
          @change="$emit('toggleSelectAll')"
        />
      </div>
      <hr class="apply-options-section__divider">
      <div
        v-for="option in APPLY_OPTION_ITEMS"
        :key="option.key"
        class="apply-options-section__option"
      >
        <UiCheckbox
          :model-value="applyOptions[option.key]"
          :disabled="disabled"
          :label="option.label"
          @change="$emit('updateOption', option.key, $event)"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
.apply-options-section {
  --apply-options-section-menu-border: var(--color-border-muted);
  --apply-options-section-menu-divider: var(--color-border-muted);
  --apply-options-section-menu-shadow: var(--shadow-medium);
  --apply-options-section-option-text: var(--color-text-secondary);
  --apply-options-section-option-hover-text: var(--color-action-primary-hover);
  --apply-options-section-action-background: linear-gradient(135deg, var(--color-action-primary-soft) 0%, var(--color-action-primary-hover) 100%);
  --apply-options-section-action-hover-background: linear-gradient(135deg, var(--color-action-primary) 0%, var(--color-text-link-strong) 100%);
  --apply-options-section-options-action-background: linear-gradient(135deg, var(--color-action-primary-hover) 0%, var(--color-text-link-strong) 100%);
  --apply-options-section-options-action-hover-background: linear-gradient(135deg, var(--color-text-link-strong) 0%, color-mix(in srgb, var(--color-text-link-strong) 78%, var(--color-surface-inverse)) 100%);
  --apply-options-section-options-action-border: color-mix(in srgb, var(--color-text-inverse) 24%, transparent);
  --apply-options-section-options-action-text: var(--color-text-inverse);

  display: flex;
  flex-direction: column;
  position: relative;
  width: 100%;
  margin-top: 19px;
}

.apply-options-section__actions {
  --ui-button-padding: 0;
  --ui-button-radius: 8px 0 0 8px;
  --ui-button-font-size: 13px;
  --ui-button-primary-background: var(--apply-options-section-action-background);
  --ui-button-primary-hover-background: var(--apply-options-section-action-hover-background);
  --ui-button-primary-shadow: none;
  --ui-button-primary-hover-shadow: none;
  --ui-button-primary-hover-transform: none;

  gap: 0;
  width: 100%;
  height: 38px;
}

.apply-options-section__action {
  flex: 1;
  min-width: 0;
  height: 38px;
  font-weight: 600;
}

.apply-options-section__options-action {
  flex: 0 0 auto;
  width: 38px;
  height: 38px;
  border: 0;
  border-left: 1px solid var(--apply-options-section-options-action-border);
  border-radius: 0 8px 8px 0;
  background: var(--apply-options-section-options-action-background);
  color: var(--apply-options-section-options-action-text);
}

.apply-options-section__options-action:hover:not(:disabled) {
  background: var(--apply-options-section-options-action-hover-background);
  transform: none;
}

.apply-options-section__menu {
  position: absolute;
  inset: auto 0 calc(100% + 6px) 0;
  z-index: var(--z-overlay);
  max-height: 260px;
  padding: 10px;
  overflow-y: auto;
  border: 1px solid var(--apply-options-section-menu-border);
  border-radius: 10px;
  background: var(--color-surface-base);
  box-shadow: 0 12px 24px var(--apply-options-section-menu-shadow);
}

.apply-options-section__option {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 26px;
  color: var(--apply-options-section-option-text);
  font-size: 13px;
  cursor: pointer;
}

.apply-options-section__option:hover {
  color: var(--apply-options-section-option-hover-text);
}

.apply-options-section__divider {
  margin: 6px 0;
  border: none;
  border-top: 1px solid var(--apply-options-section-menu-divider);
}
</style>
