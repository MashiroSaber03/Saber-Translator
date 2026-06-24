<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import UiPanel from '@/components/ui/UiPanel.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
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
    <UiPanel variant="settings">
      <template #title>🚀 并行翻译</template>
      
      <UiField class="ui-settings-field">
        <label>启用并行模式:</label>
        <label class="toggle-switch">
          <UiInput type="checkbox" class="parallel-settings__toggle-input" v-model="parallelEnabled" />
          <span class="toggle-slider"></span>
        </label>
        <div class="ui-form-hint">使用流水线并行处理，可能提升批量翻译速度</div>
      </UiField>

      <UiField class="ui-settings-field" :class="{ 'item-disabled': !parallelEnabled }">
        <label>深度学习并发数:</label>
        <div class="number-control">
          <UiButton
            variant="secondary"
            class="number-control__button"
            @click="lockSize = Math.max(1, lockSize - 1)"
            :disabled="!parallelEnabled"
            size="sm"
          >
            -
          </UiButton>
          <UiInput 
            type="number" 
            v-model.number="lockSize" 
            min="1" 
            max="4"
            :disabled="!parallelEnabled"
            class="number-input"
          />
          <UiButton
            variant="secondary"
            class="number-control__button"
            @click="lockSize = Math.min(4, lockSize + 1)"
            :disabled="!parallelEnabled"
            size="sm"
          >
            +
          </UiButton>
        </div>
        <div class="ui-form-hint">控制检测/OCR/颜色/修复的最大并发数（建议1-2）</div>
      </UiField>

      <div class="settings-note" v-if="parallelEnabled">
        <div class="note-title">⚠️ 注意事项：</div>
        <ul>
          <li>并发数设为1时为串行执行，最稳定</li>
          <li>增大并发数可能加速处理，但会占用更多GPU/CPU资源</li>
          <li>如果遇到显存不足，请将并发数设为1</li>
        </ul>
      </div>
    </UiPanel>
  </div>
</template>

<style scoped>
.parallel-settings {
  --parallel-settings-border-default: rgba(255, 193, 7, .3);
  --parallel-settings-surface-base: rgba(255, 193, 7, .1);
  --parallel-settings-text-primary: #ffc107;
}

.toggle-switch {
  position: relative;
  display: inline-block;
  width: 44px;
  height: 24px;
  margin-left: 8px;
}

.parallel-settings__toggle-input {
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle-slider {
  position: absolute;
  cursor: pointer;
  inset: 0;
  background-color: var(--color-border-muted);
  transition: 0.3s;
  border-radius: 24px;
}

.toggle-slider::before {
  position: absolute;
  content: "";
  height: 18px;
  width: 18px;
  left: 3px;
  bottom: 3px;
  background-color: var(--color-text-inverse);
  transition: 0.3s;
  border-radius: 50%;
}

.parallel-settings__toggle-input:checked + .toggle-slider {
  background-color: var(--color-action-primary);
}

.parallel-settings__toggle-input:checked + .toggle-slider::before {
  transform: translateX(20px);
}

.number-control {
  display: flex;
  align-items: center;
  gap: 4px;
}

.number-control__button {
  width: 28px;
  height: 28px;
  padding: 0;
  font-size: 14px;
}

.number-input {
  width: 50px;
  height: 28px;
  text-align: center;
  border: 1px solid var(--color-border-muted);
  background: var(--color-surface-input);
  color: var(--color-text-strong);
  border-radius: 4px;
  font-size: 14px;
}

.number-input::-webkit-inner-spin-button,
.number-input::-webkit-outer-spin-button {
  appearance: none;
  margin: 0;
}

.item-disabled {
  opacity: 0.5;
  pointer-events: none;
}

.settings-note {
  margin-top: 12px;
  padding: 10px 12px;
  background: var(--parallel-settings-surface-base);
  border: 1px solid var(--parallel-settings-border-default);
  border-radius: 6px;
  font-size: 12px;
}

.note-title {
  color: var(--parallel-settings-text-primary);
  font-weight: 500;
  margin-bottom: 6px;
}

.settings-note ul {
  margin: 0;
  padding-left: 18px;
  color: var(--color-text-supporting);
}

.settings-note li {
  margin: 3px 0;
}
</style>
