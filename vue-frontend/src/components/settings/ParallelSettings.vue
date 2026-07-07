<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiSwitch from '@/components/ui/UiSwitch.vue'
import { computed } from 'vue'
import { useSettingsStore } from '@/stores/settings'

const settingsStore = useSettingsStore()

const parallelEnabled = computed({
  get: () => settingsStore.settings.parallel.enabled,
  set: (value: boolean) => {
    settingsStore.updateSettings({
      parallel: {
        ...settingsStore.settings.parallel,
        enabled: value
      }
    })
  }
})

const lockSize = computed({
  get: () => settingsStore.settings.parallel.deepLearningLockSize,
  set: (value: number) => {
    settingsStore.updateSettings({
      parallel: {
        ...settingsStore.settings.parallel,
        deepLearningLockSize: Math.max(1, Math.min(4, value))
      }
    })
  }
})
</script>

<template>
  <div class="parallel-settings">
    <ProductFormSection>
      <template #title>并行翻译</template>

      <UiField
        variant="settings"
        label="启用并行模式"
        control="checkbox"
        hint="使用流水线并行处理，可能提升批量翻译速度"
      >
        <UiSwitch
          :model-value="parallelEnabled"
          aria-label="启用并行模式"
          @change="parallelEnabled = $event"
        />
      </UiField>

      <UiField
        variant="settings"
        label="深度学习并发数"
        control-id="parallelDeepLearningLockSize"
        hint="控制检测/OCR/颜色/修复的最大并发数（建议1-2）"
        :class="{ 'parallel-settings__field--disabled': !parallelEnabled }"
      >
        <UiNumberField
          v-model="lockSize"
          input-id="parallelDeepLearningLockSize"
          aria-label="深度学习并发数"
          :min="1"
          :max="4"
          controls
          :disabled="!parallelEnabled"
        />
      </UiField>

      <div class="parallel-settings__note" v-if="parallelEnabled">
        <div class="parallel-settings__note-title">
          <UiIcon name="alert-triangle" />
          <span>注意事项</span>
        </div>
        <ul class="parallel-settings__note-list">
          <li class="parallel-settings__note-item">并发数设为1时为串行执行，最稳定</li>
          <li class="parallel-settings__note-item">增大并发数可能加速处理，但会占用更多GPU/CPU资源</li>
          <li class="parallel-settings__note-item">如果遇到显存不足，请将并发数设为1</li>
        </ul>
      </div>
    </ProductFormSection>
  </div>
</template>

<style scoped>
.parallel-settings {
  --parallel-settings-note-background: var(--color-status-warning-surface-soft);
  --parallel-settings-note-border: color-mix(in srgb, var(--color-status-warning) 30%, transparent);
  --parallel-settings-note-title: var(--color-status-warning);
}

.parallel-settings__field--disabled {
  opacity: 0.5;
  pointer-events: none;
}

.parallel-settings__note {
  margin-top: 12px;
  padding: 10px 12px;
  background: var(--parallel-settings-note-background);
  border: 1px solid var(--parallel-settings-note-border);
  border-radius: 6px;
  font-size: 12px;
}

.parallel-settings__note-title {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--parallel-settings-note-title);
  font-weight: 500;
  margin-bottom: 6px;
}

.parallel-settings__note-list {
  margin: 0;
  padding-left: 18px;
  color: var(--color-text-supporting);
}

.parallel-settings__note-item {
  margin: 3px 0;
}
</style>
