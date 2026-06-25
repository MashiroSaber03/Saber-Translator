<script setup lang="ts">
import CollapsiblePanel from '@/components/common/CollapsiblePanel.vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiInput from '@/components/ui/UiInput.vue'
import type { TextStyleSettings } from '@/types/settings'
import type { ApplySettingsOptions } from '../useSettingsSidebar'
import ApplyOptionsSection from './ApplyOptionsSection.vue'

type SelectOption = { label: string; value: string | number }

defineProps<{
  applyOptions: ApplySettingsOptions
  fontSelectOptions: SelectOption[]
  hasImages: boolean
  inpaintMethodOptions: SelectOption[]
  layoutDirectionOptions: SelectOption[]
  showApplyOptions: boolean
  textAlignOptions: SelectOption[]
  textStyle: TextStyleSettings
}>()

defineEmits<{
  (event: 'apply'): void
  (event: 'fontSelectChange', value: string | number): void
  (event: 'inpaintMethodChange', value: string | number): void
  (event: 'layoutDirectionChange', value: string | number): void
  (event: 'selectAll'): void
  (event: 'textAlignChange', value: string | number): void
  (event: 'toggleApplyOptions'): void
  (event: 'updateApplyOption', key: keyof ApplySettingsOptions, value: boolean): void
  (event: 'updateAutoFontSize', value: boolean): void
  (event: 'updateFillColor', value: Event): void
  (event: 'updateFontSize', value: Event): void
  (event: 'updateLineSpacing', value: Event): void
  (event: 'updateStrokeColor', value: Event): void
  (event: 'updateStrokeEnabled', value: boolean): void
  (event: 'updateStrokeWidth', value: Event): void
  (event: 'updateTextColor', value: Event): void
  (event: 'updateUseAutoTextColor', value: boolean): void
}>()
</script>

<template>
  <CollapsiblePanel
    title="文字设置"
    :default-expanded="true"
    variant="settings"
    class="settings-panel text-settings-panel"
  >
    <div class="settings-form text-settings-form">
      <section class="setting-group setting-group-typography">
        <div class="group-title-row">
          <h3 class="group-title">字体排版</h3>
          <span class="group-note">影响新翻译文本</span>
        </div>
        <div class="settings-sidebar__field">
          <label for="fontSize">字号</label>
          <UiInput
            id="fontSize"
            type="number"
            :value="textStyle.fontSize"
            min="10"
            :disabled="textStyle.autoFontSize"
            :class="{ 'disabled-input': textStyle.autoFontSize }"
            :title="textStyle.autoFontSize ? '已启用自动字号，首次翻译时将自动计算' : ''"
            @input="$emit('updateFontSize', $event)"
          />
          <UiButton
            variant="toolbar"
            class="toggle-pill auto-fontsize-toggle"
            :aria-pressed="textStyle.autoFontSize"
            title="勾选后，首次翻译时自动为每个气泡计算合适的字号"
            @click="$emit('updateAutoFontSize', !textStyle.autoFontSize)"
          >
            <span>自动计算初始字号</span>
          </UiButton>
        </div>

        <div class="settings-sidebar__field">
          <label for="fontFamily">文本字体</label>
          <CustomSelect
            :model-value="textStyle.fontFamily"
            :options="fontSelectOptions"
            @change="$emit('fontSelectChange', $event)"
          />
        </div>

        <div class="settings-sidebar__field">
          <label for="layoutDirection">排版方向</label>
          <CustomSelect
            :model-value="textStyle.layoutDirection"
            :options="layoutDirectionOptions"
            @change="$emit('layoutDirectionChange', $event)"
          />
        </div>

        <div class="settings-sidebar__field">
          <label for="lineSpacing">行间距</label>
          <UiInput
            id="lineSpacing"
            type="number"
            :value="textStyle.lineSpacing"
            min="0.5"
            max="3"
            step="0.1"
            title="行间距倍数（0.5 - 3.0）"
            @change="$emit('updateLineSpacing', $event)"
          />
        </div>

        <div class="settings-sidebar__field">
          <label for="textAlign">对齐方式</label>
          <CustomSelect
            :model-value="textStyle.textAlign"
            :options="textAlignOptions"
            @change="$emit('textAlignChange', $event)"
          />
        </div>
      </section>

      <section class="setting-group setting-group-color">
        <div class="group-title-row">
          <h3 class="group-title">颜色与填充</h3>
        </div>
        <div class="settings-sidebar__field">
          <div class="label-row">
            <label for="textColor">文字颜色</label>
            <UiButton
              variant="toolbar"
              class="toggle-pill auto-color-toggle"
              :aria-pressed="textStyle.useAutoTextColor"
              title="翻译时自动使用识别到的文字颜色"
              @click="$emit('updateUseAutoTextColor', !textStyle.useAutoTextColor)"
            >
              <span>自动</span>
            </UiButton>
          </div>
          <UiInput
            id="textColor"
            type="color"
            class="color-input"
            :value="textStyle.textColor"
            :disabled="textStyle.useAutoTextColor"
            @input="$emit('updateTextColor', $event)"
          />
          <div v-if="textStyle.useAutoTextColor" class="inline-hint">
            翻译时将自动使用识别到的文字颜色
          </div>
        </div>

        <div class="settings-sidebar__field">
          <label for="useInpainting">气泡填充方式</label>
          <CustomSelect
            :model-value="textStyle.inpaintMethod"
            :options="inpaintMethodOptions"
            @change="$emit('inpaintMethodChange', $event)"
          />
        </div>

        <Transition name="slide-fade">
          <div
            v-if="textStyle.inpaintMethod === 'solid'"
            id="solidColorOptions"
            class="settings-sidebar__field inline-color-group"
          >
            <label for="fillColor">填充颜色</label>
            <UiInput
              id="fillColor"
              type="color"
              class="color-input compact"
              :value="textStyle.fillColor"
              @input="$emit('updateFillColor', $event)"
            />
          </div>
        </Transition>
      </section>

      <section class="setting-group setting-group-stroke">
        <div class="group-title-row">
          <h3 class="group-title">描边</h3>
          <UiButton
            variant="toolbar"
            class="toggle-pill stroke-toggle"
            :aria-pressed="textStyle.strokeEnabled"
            @click="$emit('updateStrokeEnabled', !textStyle.strokeEnabled)"
          >
            <span>启用描边</span>
          </UiButton>
        </div>

        <Transition name="stroke-slide">
          <div v-if="textStyle.strokeEnabled" id="strokeOptions" class="stroke-options">
            <div class="stroke-grid">
              <div class="settings-sidebar__field">
                <label for="strokeColor">描边颜色</label>
                <UiInput
                  id="strokeColor"
                  type="color"
                  class="color-input compact"
                  :value="textStyle.strokeColor"
                  @input="$emit('updateStrokeColor', $event)"
                />
              </div>
              <div class="settings-sidebar__field">
                <label for="strokeWidth">描边宽度 (px)</label>
                <UiInput
                  id="strokeWidth"
                  type="number"
                  class="compact-number-input"
                  :value="textStyle.strokeWidth"
                  min="0"
                  max="10"
                  @input="$emit('updateStrokeWidth', $event)"
                />
                <div class="ui-form-hint">0 表示无描边。</div>
              </div>
            </div>
          </div>
        </Transition>
      </section>
    </div>

    <ApplyOptionsSection
      :apply-options="applyOptions"
      :has-images="hasImages"
      :show-apply-options="showApplyOptions"
      @apply="$emit('apply')"
      @toggle-options="$emit('toggleApplyOptions')"
      @toggle-select-all="$emit('selectAll')"
      @update-option="(key, value) => $emit('updateApplyOption', key, value)"
    />
  </CollapsiblePanel>
</template>

<style scoped>
.settings-panel.collapsible-panel {
  --settings-sidebar-text-style-panel-border: #d8e3f1;
  --settings-sidebar-text-style-panel-background: #f5f8fd;
  --settings-sidebar-text-style-divider-typography: #d4deeb;
  --settings-sidebar-text-style-divider-color: #24a87a;
  --settings-sidebar-text-style-divider-stroke: #dc9a2f;
  --settings-sidebar-text-style-title-divider: #dfe8f4;
  --settings-sidebar-text-style-title: #273959;
  --settings-sidebar-text-style-title-note: #7d8ba4;
  --settings-sidebar-text-style-label: #2f3d56;
  --settings-sidebar-text-style-pill-border: #d3deed;
  --settings-sidebar-text-style-pill-border-active: #94b5e5;
  --settings-sidebar-text-style-pill-background: #f4f8fd;
  --settings-sidebar-text-style-pill-background-active: #e9f2ff;
  --settings-sidebar-text-style-pill-text: #5b6f8e;
  --settings-sidebar-text-style-pill-text-active: #21579c;
  --settings-sidebar-text-style-control-border: #cfdcec;
  --settings-sidebar-text-style-hint-border: #d2e2fa;
  --settings-sidebar-text-style-hint-background: #edf4ff;
  --settings-sidebar-text-style-hint-text: #3a6ea7;
  --settings-sidebar-text-style-stroke-divider: #d7e2ef;
  --settings-sidebar-text-style-form-hint: #6f8099;

  margin: 0 0 12px;
  padding: 12px;
  border: 1px solid var(--settings-sidebar-text-style-panel-border);
  border-radius: 12px;
  background: var(--settings-sidebar-text-style-panel-background);
}

.settings-form {
  display: flex;
  flex-direction: column;
}

.setting-group {
  margin: 0;
  padding: 10px 0;
  border-radius: 0;
  background: transparent;
  box-shadow: none;
}

.setting-group:last-child {
  margin-bottom: 0;
}

.setting-group + .setting-group {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 3px solid var(--settings-sidebar-group-divider-color);
}

.setting-group-typography {
  --settings-sidebar-group-divider-color: var(--settings-sidebar-text-style-divider-typography);
}

.setting-group-color {
  --settings-sidebar-group-divider-color: var(--settings-sidebar-text-style-divider-color);
}

.setting-group-stroke {
  --settings-sidebar-group-divider-color: var(--settings-sidebar-text-style-divider-stroke);
}

.setting-group-color + .setting-group-stroke {
  margin-top: 8px;
}

.group-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 12px;
  padding: 0 0 10px;
  border-bottom: 1px solid var(--settings-sidebar-text-style-title-divider);
}

.group-title {
  margin: 0;
  color: var(--settings-sidebar-text-style-title);
  font-weight: 700;
  font-size: 14px;
  line-height: 1.2;
}

.group-note {
  color: var(--settings-sidebar-text-style-title-note);
  font-size: 11px;
  line-height: 1.2;
}

.settings-sidebar__field {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 11px;
}

.settings-sidebar__field:last-child {
  margin-bottom: 0;
}

.settings-sidebar__field > label {
  margin: 0;
  color: var(--settings-sidebar-text-style-label);
  font-weight: 600;
  font-size: 13px;
}

.label-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.disabled-input {
  opacity: 0.55;
  cursor: not-allowed;
}

.toggle-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  width: fit-content;
  min-height: 31px;
  padding: 5px 10px;
  border: 1px solid var(--settings-sidebar-text-style-pill-border);
  border-radius: 999px;
  background: var(--settings-sidebar-text-style-pill-background);
  color: var(--settings-sidebar-text-style-pill-text);
  font-weight: 500;
  font-size: 12px;
  cursor: pointer;
  user-select: none;
}

.toggle-pill::before {
  content: '';
  box-sizing: border-box;
  display: inline-flex;
  width: 14px;
  height: 14px;
  align-items: center;
  justify-content: center;
  border: 1px solid currentcolor;
  border-radius: 3px;
  background: var(--color-surface-base);
  font-size: 11px;
  line-height: 1;
}

.toggle-pill[aria-pressed='true'] {
  border-color: var(--settings-sidebar-text-style-pill-border-active);
  background: var(--settings-sidebar-text-style-pill-background-active);
  color: var(--settings-sidebar-text-style-pill-text-active);
}

.toggle-pill[aria-pressed='true']::before {
  content: '✓';
  border-color: var(--settings-sidebar-text-style-pill-border-active);
  background: var(--settings-sidebar-text-style-pill-border-active);
  color: var(--color-text-inverse);
  font-weight: 700;
}

.auto-fontsize-toggle {
  margin-top: 2px;
}

.color-input {
  width: 58px;
  height: 34px;
  padding: 2px;
  border: 1px solid var(--settings-sidebar-text-style-control-border);
  border-radius: 8px;
  background: var(--color-surface-base);
  cursor: pointer;
}

.color-input.compact {
  width: 72px;
}

.color-input:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.inline-color-group {
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
}

.inline-hint {
  padding: 6px 8px;
  border: 1px solid var(--settings-sidebar-text-style-hint-border);
  border-radius: 8px;
  background: var(--settings-sidebar-text-style-hint-background);
  color: var(--settings-sidebar-text-style-hint-text);
  font-size: 12px;
  line-height: 1.35;
}

.stroke-options {
  margin-top: 8px;
  padding: 8px 0 0;
  border-top: 1px dashed var(--settings-sidebar-text-style-stroke-divider);
  border-radius: 0;
  background: transparent;
}

.stroke-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.compact-number-input {
  width: 100%;
  min-height: 36px;
}

.ui-form-hint {
  color: var(--settings-sidebar-text-style-form-hint);
  font-size: 11px;
  line-height: 1.3;
}

.slide-fade-enter-active {
  transition: all 0.28s ease-out;
}

.slide-fade-leave-active {
  transition: all 0.2s ease-in;
}

.slide-fade-enter-from,
.slide-fade-leave-to {
  max-height: 0;
  overflow: hidden;
  opacity: 0;
}

.slide-fade-enter-to,
.slide-fade-leave-from {
  max-height: 70px;
  opacity: 1;
}

.stroke-slide-enter-active {
  transition: all 0.3s ease-out;
}

.stroke-slide-leave-active {
  transition: all 0.2s ease-in;
}

.stroke-slide-enter-from,
.stroke-slide-leave-to {
  max-height: 0;
  overflow: hidden;
  opacity: 0;
}

.stroke-slide-enter-to,
.stroke-slide-leave-from {
  max-height: 220px;
  opacity: 1;
}
</style>
