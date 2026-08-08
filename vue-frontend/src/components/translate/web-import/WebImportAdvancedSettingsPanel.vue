<script setup lang="ts">
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiField from '@/components/ui/UiField.vue'
import UiInput from '@/components/ui/UiInput.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import type { WebImportSettings } from '@/types/webImport'
import type { WebImportSettingsActions } from './webImportSettingsActions'

defineProps<{
  draftSettings: WebImportSettings
  settingsActions: WebImportSettingsActions
}>()
</script>

<template>
  <ProductFormSection class="web-import-advanced__section">
    <template #title>自定义请求头</template>

    <UiField variant="settings" label="Cookie" control-id="webImportCustomCookie">
      <UiInput
        id="webImportCustomCookie"
        type="text"
        :model-value="draftSettings.advanced.customCookie"
        placeholder="name=value; name2=value2"
        @update:model-value="value => settingsActions.setCustomCookie(String(value))"
      />
    </UiField>

    <UiField variant="settings" label="Headers (JSON)" control-id="webImportCustomHeaders">
      <UiTextarea
        id="webImportCustomHeaders"
        :model-value="draftSettings.advanced.customHeaders"
        variant="panel"
        rows="3"
        placeholder="{&quot;X-Custom-Header&quot;: &quot;value&quot;}"
        @update:model-value="settingsActions.setCustomHeaders"
      />
    </UiField>

    <UiField variant="settings" control="checkbox">
      <UiCheckbox
        :model-value="draftSettings.advanced.bypassProxy"
        label="绕过系统代理 (连接本地服务时使用)"
        @change="settingsActions.setBypassProxy"
      />
    </UiField>
  </ProductFormSection>
</template>

<style scoped>
.web-import-advanced__section {
  --product-form-section-margin-bottom: 0;
  --product-form-section-title-margin-bottom: 12px;
  --product-form-section-title-padding-bottom: 0;
  --product-form-section-title-border-bottom: 0;
  --product-form-section-title-text: var(--color-text-default);
  --product-form-section-title-font-size: 14px;
  --product-form-section-title-font-weight: 600;
}
</style>
