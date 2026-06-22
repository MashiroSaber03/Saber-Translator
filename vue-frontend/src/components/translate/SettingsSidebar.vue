<script setup lang="ts">

import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiInput from '@/components/ui/UiInput.vue'

import UiButton from '@/components/ui/UiButton.vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import CollapsiblePanel from '@/components/common/CollapsiblePanel.vue'
import PageSelectionModal from '@/components/translate/PageSelectionModal.vue'
import { unref } from 'vue'
import { useSettingsSidebar, type SettingsSidebarEmit } from './useSettingsSidebar'

const emit = defineEmits<SettingsSidebarEmit>()

const {
  showApplyOptions,
  applyOptions,
  isPageSelectionEnabled,
  showPageSelectionModal,
  selectedWorkflowMode,
  rememberWorkflowModeEnabled,
  hasImages,
  totalImages,
  normalizedSelectedPages,
  hasValidPageSelection,
  canUseBookConstraints,
  canGoPrevious,
  canGoNext,
  canRunWorkflow,
  textStyle,
  supportsPageSelectionForCurrentMode,
  isPageSelectionActiveForCurrentMode,
  workflowModeOptions,
  workflowStartLabel,
  workflowContextTag,
  workflowModeTag,
  workflowDescription,
  isDangerousWorkflow,
  fontUploadInput,
  fontSelectOptions,
  layoutDirectionOptions,
  inpaintMethodOptions,
  textAlignOptions,
  createPageSelectionSummary,
  updateFontSize,
  updateAutoFontSize,
  handleFontUpload,
  handleFontSelectChange,
  handleLayoutDirectionChange,
  handleInpaintMethodChange,
  updateTextColor,
  updateLineSpacing,
  updateTextAlign,
  updateUseAutoTextColor,
  updateStrokeEnabled,
  updateStrokeColor,
  updateStrokeWidth,
  updateFillColor,
  toggleApplyOptions,
  toggleSelectAll,
  handleApplyToAll,
  openPageSelectionModal,
  handlePageSelectionConfirm,
  handleWorkflowModeChange,
  handleRememberWorkflowModeChange,
  handleRunWorkflow,
  handleOpenGlossary,
  handleOpenNonTranslate,
} = useSettingsSidebar(emit)
</script>

<template>
  <aside class="settings-sidebar">
    <div class="settings-card">
      <h2 class="sidebar-title">翻译设置</h2>

      <!-- 文字设置折叠面板 -->
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
                type="number"
                id="fontSize"
                :value="textStyle.fontSize"
                min="10"
                :disabled="textStyle.autoFontSize"
                :class="{ 'disabled-input': textStyle.autoFontSize }"
                :title="textStyle.autoFontSize ? '已启用自动字号，首次翻译时将自动计算' : ''"
                @input="updateFontSize"
              />
              <label
                class="toggle-pill auto-fontsize-toggle"
                for="autoFontSize"
                title="勾选后，首次翻译时自动为每个气泡计算合适的字号"
              >
                <UiInput
                  type="checkbox"
                  id="autoFontSize"
                  :checked="textStyle.autoFontSize"
                  @change="updateAutoFontSize"
                />
                <span>自动计算初始字号</span>
              </label>
            </div>

            <div class="settings-sidebar__field">
              <label for="fontFamily">文本字体</label>
              <CustomSelect
                :model-value="textStyle.fontFamily"
                :options="fontSelectOptions"
                @change="handleFontSelectChange"
              />
              <UiFileInput
                ref="fontUploadInput"
                id="fontUpload"
                accept=".ttf,.ttc,.otf"
                style="display: none"
                @change="handleFontUpload"
              />
            </div>

            <div class="settings-sidebar__field">
              <label for="layoutDirection">排版方向</label>
              <CustomSelect
                :model-value="textStyle.layoutDirection"
                :options="layoutDirectionOptions"
                @change="handleLayoutDirectionChange"
              />
            </div>

            <div class="settings-sidebar__field">
              <label for="lineSpacing">行间距</label>
              <UiInput
                type="number"
                id="lineSpacing"
                :value="textStyle.lineSpacing"
                min="0.5"
                max="3"
                step="0.1"
                title="行间距倍数（0.5 - 3.0）"
                @change="updateLineSpacing"
              />
            </div>

            <div class="settings-sidebar__field">
              <label for="textAlign">对齐方式</label>
              <CustomSelect
                :model-value="textStyle.textAlign"
                :options="textAlignOptions"
                @change="updateTextAlign"
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
                <label class="toggle-pill auto-color-toggle" title="翻译时自动使用识别到的文字颜色">
                  <UiInput
                    type="checkbox"
                    :checked="textStyle.useAutoTextColor"
                    @change="updateUseAutoTextColor"
                  />
                  <span>自动</span>
                </label>
              </div>
              <UiInput
                type="color"
                id="textColor"
                class="color-input"
                :value="textStyle.textColor"
                :disabled="textStyle.useAutoTextColor"
                @input="updateTextColor"
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
                @change="handleInpaintMethodChange"
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
                  type="color"
                  id="fillColor"
                  class="color-input compact"
                  :value="textStyle.fillColor"
                  @input="updateFillColor"
                />
              </div>
            </Transition>
          </section>

          <section class="setting-group setting-group-stroke">
            <div class="group-title-row">
              <h3 class="group-title">描边</h3>
              <label class="toggle-pill stroke-toggle" for="strokeEnabled">
                <UiInput
                  type="checkbox"
                  id="strokeEnabled"
                  :checked="textStyle.strokeEnabled"
                  @change="updateStrokeEnabled"
                />
                <span>启用描边</span>
              </label>
            </div>

            <Transition name="stroke-slide">
              <div v-if="textStyle.strokeEnabled" id="strokeOptions" class="stroke-options">
                <div class="stroke-grid">
                  <div class="settings-sidebar__field">
                    <label for="strokeColor">描边颜色</label>
                    <UiInput
                      type="color"
                      id="strokeColor"
                      class="color-input compact"
                      :value="textStyle.strokeColor"
                      @input="updateStrokeColor"
                    />
                  </div>
                  <div class="settings-sidebar__field">
                    <label for="strokeWidth">描边宽度 (px)</label>
                    <UiInput
                      type="number"
                      id="strokeWidth"
                      class="compact-number-input"
                      :value="textStyle.strokeWidth"
                      min="0"
                      max="10"
                      @input="updateStrokeWidth"
                    />
                    <div class="ui-form-hint">0 表示无描边。</div>
                  </div>
                </div>
              </div>
            </Transition>
          </section>
        </div>

        <!-- 应用到全部按钮 -->
        <div class="settings-sidebar__apply-group">
          <UiButton
            variant="toolbar"
            type="button"
            class="settings-sidebar__apply-button"
            :disabled="!unref(hasImages)"
            @click="handleApplyToAll"
          >
            应用到全部
          </UiButton>
          <UiButton
            variant="toolbar"
            type="button"
            class="settings-sidebar__apply-options-button"
            title="选择要应用的参数"
            @click="toggleApplyOptions"
          >
            ⚙️
          </UiButton>

          <!-- 应用选项下拉菜单 -->
          <div v-if="showApplyOptions" class="apply-options-dropdown">
            <div class="apply-option">
              <UiInput
                type="checkbox"
                id="apply_selectAll"
                :checked="Object.values(applyOptions).every(v => v)"
                @change="toggleSelectAll"
              />
              <label for="apply_selectAll">全选</label>
            </div>
            <hr />
            <div class="apply-option">
              <UiInput type="checkbox" id="apply_fontSize" v-model="applyOptions.fontSize" />
              <label for="apply_fontSize">字号</label>
            </div>
            <div class="apply-option">
              <UiInput type="checkbox" id="apply_fontFamily" v-model="applyOptions.fontFamily" />
              <label for="apply_fontFamily">字体</label>
            </div>
            <div class="apply-option">
              <UiInput
                type="checkbox"
                id="apply_layoutDirection"
                v-model="applyOptions.layoutDirection"
              />
              <label for="apply_layoutDirection">排版方向</label>
            </div>
            <div class="apply-option">
              <UiInput type="checkbox" id="apply_lineSpacing" v-model="applyOptions.lineSpacing" />
              <label for="apply_lineSpacing">行间距</label>
            </div>
            <div class="apply-option">
              <UiInput type="checkbox" id="apply_textAlign" v-model="applyOptions.textAlign" />
              <label for="apply_textAlign">对齐方式</label>
            </div>
            <div class="apply-option">
              <UiInput type="checkbox" id="apply_textColor" v-model="applyOptions.textColor" />
              <label for="apply_textColor">文字颜色</label>
            </div>
            <div class="apply-option">
              <UiInput type="checkbox" id="apply_fillColor" v-model="applyOptions.fillColor" />
              <label for="apply_fillColor">填充颜色</label>
            </div>
            <div class="apply-option">
              <UiInput
                type="checkbox"
                id="apply_strokeEnabled"
                v-model="applyOptions.strokeEnabled"
              />
              <label for="apply_strokeEnabled">描边开关</label>
            </div>
            <div class="apply-option">
              <UiInput type="checkbox" id="apply_strokeColor" v-model="applyOptions.strokeColor" />
              <label for="apply_strokeColor">描边颜色</label>
            </div>
            <div class="apply-option">
              <UiInput type="checkbox" id="apply_strokeWidth" v-model="applyOptions.strokeWidth" />
              <label for="apply_strokeWidth">描边宽度</label>
            </div>
          </div>
        </div>
      </CollapsiblePanel>

      <CollapsiblePanel
        title="指定翻译页码"
        :default-expanded="false"
        variant="settings"
        class="settings-panel"
      >
        <div class="settings-form page-selection-form">
          <div class="range-header-row">
            <label class="page-selection-toggle-compact">
              <UiInput
                type="checkbox"
                v-model="isPageSelectionEnabled"
                :disabled="unref(totalImages) === 0 || !unref(supportsPageSelectionForCurrentMode)"
              />
              <span>启用</span>
            </label>
            <span class="total-count">共 {{ totalImages }} 张</span>
          </div>

          <div v-if="!supportsPageSelectionForCurrentMode" class="page-selection-note">当前模式不支持指定翻译页码</div>

          <div v-if="isPageSelectionActiveForCurrentMode" class="page-selection-summary-block">
            <div class="page-selection-summary-value">
              {{ createPageSelectionSummary(normalizedSelectedPages) }}
            </div>
            <UiButton
              variant="toolbar"
              type="button"
              class="settings-button secondary-button page-selection-open-btn"
              :disabled="unref(totalImages) === 0"
              @click="openPageSelectionModal"
            >
              选择页码
            </UiButton>
          </div>

          <div
            v-if="isPageSelectionActiveForCurrentMode && !hasValidPageSelection && totalImages > 0"
            class="page-selection-error"
          >
            请至少选择一页
          </div>
        </div>
      </CollapsiblePanel>

      <div class="book-constraints-panel">
        <div class="book-constraints-title">书籍约束</div>
        <div class="book-constraints-hint">
          术语表和禁翻表按单本漫画保存，不与其他书共享。
        </div>
        <div class="book-constraints-actions">
          <UiButton
            variant="toolbar"
            type="button"
            class="settings-button secondary-button"
            :disabled="!unref(canUseBookConstraints)"
            @click="handleOpenGlossary"
          >
            术语表
          </UiButton>
          <UiButton
            variant="toolbar"
            type="button"
            class="settings-button secondary-button"
            :disabled="!unref(canUseBookConstraints)"
            @click="handleOpenNonTranslate"
          >
            禁翻表
          </UiButton>
        </div>
        <div v-if="!canUseBookConstraints" class="book-constraints-disabled-note">
          仅书架模式可用
        </div>
      </div>

      <!-- 工作流启动区 -->
      <div class="action-buttons workflow-controls">
        <div class="settings-sidebar__field">
          <label for="workflowModeSelect">操作模式:</label>
          <CustomSelect
            id="workflowModeSelect"
            :model-value="selectedWorkflowMode"
            :options="workflowModeOptions"
            fit
            variant="workflow"
            @change="handleWorkflowModeChange"
          />
          <label class="remember-workflow-mode-toggle">
            <UiInput
              id="rememberWorkflowModeCheckbox"
              type="checkbox"
              :checked="unref(rememberWorkflowModeEnabled)"
              @change="handleRememberWorkflowModeChange"
            />
            <span>记住操作模式</span>
          </label>
        </div>
        <div class="workflow-meta">
          <span class="workflow-chip">{{ workflowContextTag }}</span>
          <span class="workflow-chip" :class="{ 'danger-chip': isDangerousWorkflow }">
            {{ workflowModeTag }}
          </span>
        </div>
        <UiButton
          variant="toolbar"
          id="runWorkflowButton"
          class="settings-button workflow-run-button"
          :class="{ 'danger-button': isDangerousWorkflow }"
          :disabled="!unref(canRunWorkflow)"
          @click="handleRunWorkflow"
        >
          {{ workflowStartLabel }}
        </UiButton>
        <div class="workflow-description">
          {{ workflowDescription }}
        </div>
      </div>

      <!-- 导航按钮 -->
      <div class="navigation-buttons">
        <UiButton variant="toolbar" id="prevImageButton" :disabled="!unref(canGoPrevious)" @click="emit('previous')">
          上一张
        </UiButton>
        <UiButton variant="toolbar" id="nextImageButton" :disabled="!unref(canGoNext)" @click="emit('next')">下一张</UiButton>
      </div>
    </div>
    <PageSelectionModal
      :model-value="showPageSelectionModal"
      :selected-pages="normalizedSelectedPages"
      @update:model-value="showPageSelectionModal = $event"
      @confirm="handlePageSelectionConfirm"
    />
  </aside>
</template>

<style scoped>
/* Sidebar shell */
.settings-sidebar {
  /* owner tokens: settings-sidebar */
  --settings-sidebar-apply-actions-border-default: rgba(255, 255, 255, .24);
  --settings-sidebar-apply-actions-border-strong: #d7e2f2;
  --settings-sidebar-apply-actions-border-muted: #e3ebf6;
  --settings-sidebar-apply-actions-border-subtle: #d4deed;
  --settings-sidebar-apply-actions-border-hover: #94b5e5;
  --settings-sidebar-apply-actions-border-active: #f3cccc;
  --settings-sidebar-apply-actions-border-focus: #d8e3f1;
  --settings-sidebar-apply-actions-border-selected: #d3e1f6;
  --settings-sidebar-apply-actions-border-danger: #ffcaca;
  --settings-sidebar-apply-actions-shadow-default: rgba(22, 37, 58, .16);
  --settings-sidebar-apply-actions-shadow-raised: rgba(62, 169, 74, .24);
  --settings-sidebar-apply-actions-surface-base: #4b89d0;
  --settings-sidebar-apply-actions-surface-raised: #316fb6;
  --settings-sidebar-apply-actions-surface-muted: #c2c9d4;
  --settings-sidebar-apply-actions-surface-subtle: #3f7bc4;
  --settings-sidebar-apply-actions-surface-hover: #2b64a9;
  --settings-sidebar-apply-actions-surface-active: #285d99;
  --settings-sidebar-apply-actions-surface-selected: #2a64a5;
  --settings-sidebar-apply-actions-surface-overlay: #224f82;
  --settings-sidebar-apply-actions-surface-inverse: #f4f8fd;
  --settings-sidebar-apply-actions-surface-contrast: #e9f2ff;
  --settings-sidebar-apply-actions-surface-tint: #4a82ce;
  --settings-sidebar-apply-actions-surface-soft: #f8fbff;
  --settings-sidebar-apply-actions-surface-strong: #3ea94a;
  --settings-sidebar-apply-actions-surface-stronger: #e8f0fd;
  --settings-sidebar-apply-actions-surface-highlight: #ffe7e7;
  --settings-sidebar-apply-actions-surface-highlight-strong: #58ba54;
  --settings-sidebar-apply-actions-text-primary: #6f8099;
  --settings-sidebar-apply-actions-text-secondary: #405473;
  --settings-sidebar-apply-actions-text-muted: #2b5f9d;
  --settings-sidebar-apply-actions-text-subtle: #5d7090;
  --settings-sidebar-apply-actions-text-supporting: #21579c;
  --settings-sidebar-apply-actions-text-disabled: #6f809a;
  --settings-sidebar-apply-actions-text-inverse: #304464;
  --settings-sidebar-apply-actions-text-brand: #b73535;
  --settings-sidebar-apply-actions-text-danger: #4b5f80;
  --settings-sidebar-apply-actions-text-warning: #2d4568;
  --settings-sidebar-apply-actions-text-success: #9f2b2b;
  --settings-sidebar-shell-border-default: #dbe4ef;
  --settings-sidebar-shell-border-strong: #e2e9f2;
  --settings-sidebar-shell-border-muted: #d8e3f1;
  --settings-sidebar-shell-border-subtle: #dfe8f4;
  --settings-sidebar-shell-border-hover: #d3deed;
  --settings-sidebar-shell-border-active: #94b5e5;
  --settings-sidebar-shell-border-focus: #cfdcec;
  --settings-sidebar-shell-border-selected: #d2e2fa;
  --settings-sidebar-shell-border-danger: #d7e2ef;
  --settings-sidebar-shell-shadow-default: rgba(28, 45, 72, .07);
  --settings-sidebar-shell-surface-base: #eef3f9;
  --settings-sidebar-shell-surface-raised: #c7d5e7;
  --settings-sidebar-shell-surface-muted: #f5f8fd;
  --settings-sidebar-shell-surface-subtle: #f4f8fd;
  --settings-sidebar-shell-surface-hover: #4a82ce;
  --settings-sidebar-shell-surface-active: #e9f2ff;
  --settings-sidebar-shell-surface-selected: #edf4ff;
  --settings-sidebar-shell-text-primary: #c7d5e7;
  --settings-sidebar-shell-text-secondary: #eef3f9;
  --settings-sidebar-shell-text-muted: #20314f;
  --settings-sidebar-shell-text-subtle: #d4deeb;
  --settings-sidebar-shell-text-supporting: #24a87a;
  --settings-sidebar-shell-text-disabled: #dc9a2f;
  --settings-sidebar-shell-text-inverse: #273959;
  --settings-sidebar-shell-text-brand: #7d8ba4;
  --settings-sidebar-shell-text-danger: #2f3d56;
  --settings-sidebar-shell-text-warning: #5b6f8e;
  --settings-sidebar-shell-text-success: #21579c;
  --settings-sidebar-shell-text-info: #3a6ea7;
  --settings-sidebar-workflow-border-default: #d8e3f1;
  --settings-sidebar-workflow-border-strong: #bfd0e5;
  --settings-sidebar-workflow-shadow-default: rgba(54, 151, 64, .28);
  --settings-sidebar-workflow-shadow-raised: rgba(214, 66, 66, .24);
  --settings-sidebar-workflow-shadow-floating: rgba(191, 52, 52, .28);
  --settings-sidebar-workflow-surface-base: #d64242;
  --settings-sidebar-workflow-surface-raised: #bf3434;
  --settings-sidebar-workflow-surface-muted: #c1c8d1;
  --settings-sidebar-workflow-surface-subtle: #f8fbff;
  --settings-sidebar-workflow-surface-hover: #eef2f6;
  --settings-sidebar-workflow-surface-active: #eef4fb;
  --settings-sidebar-workflow-surface-selected: #6c7784;
  --settings-sidebar-workflow-surface-overlay: #c2c9d4;
  --settings-sidebar-workflow-surface-inverse: #5a6572;
  --settings-sidebar-workflow-text-primary: #5c6f8f;
  --settings-sidebar-workflow-text-secondary: #273959;
  --settings-sidebar-workflow-text-muted: #62748f;
  --settings-sidebar-workflow-text-subtle: #2f4b71;
  --settings-sidebar-workflow-text-supporting: #8b97a7;

  width: 100%;
  height: 100%;
  overflow-y: auto;
  padding: 10px 20px 20px;
  display: flex;
  flex-direction: column;
  direction: rtl;
  z-index: var(--z-fixed-sidebar);
  scrollbar-width: thin;
  scrollbar-color: var(--settings-sidebar-shell-text-primary) var(--settings-sidebar-shell-text-secondary);
}

.settings-sidebar > * {
  direction: ltr;
}

.settings-sidebar::-webkit-scrollbar {
  width: 8px;
}

.settings-sidebar::-webkit-scrollbar-track {
  background: var(--settings-sidebar-shell-surface-base);
  border-radius: 999px;
}

.settings-sidebar::-webkit-scrollbar-thumb {
  background: var(--settings-sidebar-shell-surface-raised);
  border-radius: 999px;
}

.settings-card {
  background: var(--color-surface-base);
  border: 1px solid var(--settings-sidebar-shell-border-default);
  border-radius: 14px;
  box-shadow: 0 8px 20px var(--settings-sidebar-shell-shadow-default);
  padding: 18px;
  margin-bottom: 14px;
}

.sidebar-title {
  margin: 0 0 14px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--settings-sidebar-shell-border-strong);
  color: var(--settings-sidebar-shell-text-muted);
  font-size: 24px;
  font-weight: 700;
  text-align: center;
}

.settings-panel {
  margin: 0 0 12px;
  padding: 12px;
  border: 1px solid var(--settings-sidebar-shell-border-muted);
  border-radius: 12px;
  background: var(--settings-sidebar-shell-surface-muted);
}

@media (--breakpoint-sidebar-height-compact) {
  .sidebar-title {
    font-size: 22px;
  }
}

@media (--breakpoint-md-down) {
  .settings-sidebar {
    order: 2;
    width: 100%;
    height: auto;
    max-height: none;
    margin-top: 0;
    overflow: visible;
    padding: 0;
    direction: ltr;
    z-index: auto;
  }

  .settings-card {
    padding: 14px 16px 32px;
    margin-bottom: 0;
  }

  .sidebar-title {
    font-size: 20px;
  }
}

.settings-form {
  display: flex;
  flex-direction: column;
}

.setting-group {
  --settings-sidebar-group-divider-color: var(--settings-sidebar-shell-text-subtle);

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

.group-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 12px;
  padding: 0 0 10px;
  border-bottom: 1px solid var(--settings-sidebar-shell-border-subtle);
}

.setting-group-typography {
  --settings-sidebar-group-divider-color: var(--settings-sidebar-shell-text-subtle);
}

.setting-group-color {
  --settings-sidebar-group-divider-color: var(--settings-sidebar-shell-text-supporting);
}

.setting-group-stroke {
  --settings-sidebar-group-divider-color: var(--settings-sidebar-shell-text-disabled);
}

.setting-group-color + .setting-group-stroke {
  margin-top: 8px;
}

.group-title {
  margin: 0;
  color: var(--settings-sidebar-shell-text-inverse);
  font-size: 14px;
  font-weight: 700;
  line-height: 1.2;
}

.group-note {
  color: var(--settings-sidebar-shell-text-brand);
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
  color: var(--settings-sidebar-shell-text-danger);
  font-size: 13px;
  font-weight: 600;
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
  padding: 5px 10px;
  border: 1px solid var(--settings-sidebar-shell-border-hover);
  border-radius: 999px;
  background: var(--settings-sidebar-shell-surface-subtle);
  color: var(--settings-sidebar-shell-text-warning);
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  user-select: none;
}

.toggle-pill input[type='checkbox'] {
  width: 14px;
  height: 14px;
  margin: 0;
  accent-color: var(--settings-sidebar-shell-surface-hover);
  cursor: pointer;
}

.toggle-pill:has(input:checked) {
  border-color: var(--settings-sidebar-shell-border-active);
  background: var(--settings-sidebar-shell-surface-active);
  color: var(--settings-sidebar-shell-text-success);
}

.auto-fontsize-toggle {
  margin-top: 2px;
}

.color-input {
  width: 58px;
  height: 34px;
  padding: 2px;
  border: 1px solid var(--settings-sidebar-shell-border-focus);
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
  color: var(--settings-sidebar-shell-text-info);
  font-size: 12px;
  line-height: 1.35;
  padding: 6px 8px;
  border: 1px solid var(--settings-sidebar-shell-border-selected);
  border-radius: 8px;
  background: var(--settings-sidebar-shell-surface-selected);
}

.stroke-options {
  margin-top: 8px;
  padding: 8px 0 0;
  border-top: 1px dashed var(--settings-sidebar-shell-border-danger);
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
  color: var(--settings-sidebar-apply-actions-text-primary);
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
  opacity: 0;
  max-height: 0;
  overflow: hidden;
}

.slide-fade-enter-to,
.slide-fade-leave-from {
  opacity: 1;
  max-height: 70px;
}

.stroke-slide-enter-active {
  transition: all 0.3s ease-out;
}

.stroke-slide-leave-active {
  transition: all 0.2s ease-in;
}

.stroke-slide-enter-from,
.stroke-slide-leave-to {
  opacity: 0;
  max-height: 0;
  overflow: hidden;
}

.stroke-slide-enter-to,
.stroke-slide-leave-from {
  opacity: 1;
  max-height: 220px;
}

.settings-sidebar__apply-group {
  display: flex;
  align-items: stretch;
  position: relative;
  margin-top: 8px;
  width: 100%;
  height: 38px;
}

.settings-sidebar__apply-group .settings-sidebar__apply-button {
  flex: 1;
  min-width: 0;
  margin: 0;
  border: none;
  border-radius: 8px 0 0 8px;
  background: linear-gradient(135deg, var(--settings-sidebar-apply-actions-surface-base) 0%, var(--settings-sidebar-apply-actions-surface-raised) 100%);
  color: var(--color-text-inverse);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.settings-sidebar__apply-group .settings-sidebar__apply-button:disabled {
  background: var(--settings-sidebar-apply-actions-surface-muted);
  cursor: not-allowed;
}

.settings-sidebar__apply-group .settings-sidebar__apply-button:hover:not(:disabled) {
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

.settings-gear-btn:hover {
  background: linear-gradient(135deg, var(--settings-sidebar-apply-actions-surface-selected) 0%, var(--settings-sidebar-apply-actions-surface-overlay) 100%);
}

.apply-options-dropdown {
  position: absolute;
  inset: auto 0 calc(100% + 6px) 0;
  padding: 10px;
  border: 1px solid var(--settings-sidebar-apply-actions-border-strong);
  border-radius: 10px;
  background: var(--color-surface-base);
  box-shadow: 0 12px 24px var(--settings-sidebar-apply-actions-shadow-default);
  max-height: 260px;
  overflow-y: auto;
  z-index: var(--z-overlay);
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

.apply-option input[type='checkbox'] {
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

.page-selection-form {
  gap: 8px;
}

.range-header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.page-selection-toggle-compact {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border: 1px solid var(--settings-sidebar-apply-actions-border-subtle);
  border-radius: 999px;
  background: var(--settings-sidebar-apply-actions-surface-inverse);
  color: var(--settings-sidebar-apply-actions-text-subtle);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}

.page-selection-toggle-compact:has(input:checked) {
  border-color: var(--settings-sidebar-apply-actions-border-hover);
  background: var(--settings-sidebar-apply-actions-surface-contrast);
  color: var(--settings-sidebar-apply-actions-text-supporting);
}

.page-selection-toggle-compact input[type='checkbox'] {
  width: 14px;
  height: 14px;
  margin: 0;
  accent-color: var(--settings-sidebar-apply-actions-surface-tint);
}

.total-count {
  color: var(--settings-sidebar-apply-actions-text-disabled);
  font-size: 12px;
  font-weight: 500;
}

.page-selection-note {
  color: var(--settings-sidebar-apply-actions-text-primary);
  font-size: 12px;
}

.page-selection-summary-block {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 4px 0 0;
}

.page-selection-summary-value {
  color: var(--settings-sidebar-apply-actions-text-inverse);
  font-size: 13px;
  line-height: 1.5;
  word-break: break-word;
}

.page-selection-open-btn {
  align-self: stretch;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  padding: 0 14px;
}

.page-selection-error {
  color: var(--settings-sidebar-apply-actions-text-brand);
  font-size: 12px;
  font-weight: 600;
  margin-top: 2px;
  padding: 6px 10px;
  border: 1px solid var(--settings-sidebar-apply-actions-border-active);
  border-radius: 8px;
  background: var(--color-surface-slate-soft);
  text-align: center;
}

.action-buttons {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 14px;
}

.workflow-controls {
  padding: 12px;
  border: 1px solid var(--settings-sidebar-apply-actions-border-focus);
  border-radius: 12px;
  background: var(--settings-sidebar-apply-actions-surface-soft);
}

.workflow-controls .settings-sidebar__field {
  margin-bottom: 0;
}

.workflow-controls .settings-sidebar__field label {
  margin-bottom: 6px;
}

.remember-workflow-mode-toggle {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-top: 10px;
  margin-bottom: 0;
  color: var(--settings-sidebar-apply-actions-text-danger);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}

.remember-workflow-mode-toggle input {
  width: 16px;
  height: 16px;
  accent-color: var(--settings-sidebar-apply-actions-surface-strong);
}

.workflow-meta {
  display: flex;
  gap: 8px;
  align-items: center;
}

.workflow-chip {
  display: inline-flex;
  align-items: center;
  height: 24px;
  padding: 0 9px;
  border: 1px solid var(--settings-sidebar-apply-actions-border-selected);
  border-radius: 999px;
  background: var(--settings-sidebar-apply-actions-surface-stronger);
  color: var(--settings-sidebar-apply-actions-text-warning);
  font-size: 12px;
  font-weight: 600;
}

.workflow-chip.danger-chip {
  border-color: var(--settings-sidebar-apply-actions-border-danger);
  background: var(--settings-sidebar-apply-actions-surface-highlight);
  color: var(--settings-sidebar-apply-actions-text-success);
}

.workflow-run-button {
  min-height: 54px;
  border: none;
  border-radius: 10px;
  background: linear-gradient(135deg, var(--settings-sidebar-apply-actions-surface-strong) 0%, var(--settings-sidebar-apply-actions-surface-highlight-strong) 100%);
  box-shadow: 0 8px 16px var(--settings-sidebar-apply-actions-shadow-raised);
  color: var(--color-text-inverse);
  font-size: 16px;
  font-weight: 700;
  cursor: pointer;
  transition:
    transform 0.2s ease,
    box-shadow 0.2s ease;
}

@media (--breakpoint-md-down) {
  .workflow-controls {
    margin-top: 8px;
  }
}

.workflow-run-button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 10px 18px var(--settings-sidebar-workflow-shadow-default);
}

.workflow-run-button.danger-button {
  background: linear-gradient(135deg, var(--settings-sidebar-workflow-surface-base) 0%, var(--settings-sidebar-workflow-surface-raised) 100%);
  box-shadow: 0 8px 16px var(--settings-sidebar-workflow-shadow-raised);
}

.workflow-run-button.danger-button:hover:not(:disabled) {
  box-shadow: 0 10px 18px var(--settings-sidebar-workflow-shadow-floating);
}

.workflow-run-button:disabled {
  background: var(--settings-sidebar-workflow-surface-muted);
  box-shadow: none;
  cursor: not-allowed;
}

.workflow-description {
  color: var(--settings-sidebar-workflow-text-primary);
  font-size: 13px;
  line-height: 1.45;
}

.book-constraints-panel {
  margin-top: 14px;
  padding: 12px;
  border: 1px solid var(--settings-sidebar-workflow-border-default);
  border-radius: 12px;
  background: var(--settings-sidebar-workflow-surface-subtle);
}

.book-constraints-title {
  color: var(--settings-sidebar-workflow-text-secondary);
  font-size: 15px;
  font-weight: 700;
}

.book-constraints-hint {
  margin-top: 6px;
  color: var(--settings-sidebar-workflow-text-muted);
  font-size: 12px;
  line-height: 1.4;
}

.book-constraints-actions {
  display: flex;
  gap: 10px;
  margin-top: 12px;
}

.book-constraints-actions .settings-button {
  flex: 1;
}

.secondary-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 38px;
  padding: 0 14px;
  border: 1px solid var(--settings-sidebar-workflow-border-strong);
  border-radius: 8px;
  background: var(--color-surface-plain);
  color: var(--settings-sidebar-workflow-text-subtle);
  font-size: 13px;
  font-weight: 600;
}

.secondary-button:disabled {
  background: var(--settings-sidebar-workflow-surface-hover);
  color: var(--settings-sidebar-workflow-text-supporting);
  cursor: not-allowed;
}

.secondary-button:hover:not(:disabled) {
  background: var(--settings-sidebar-workflow-surface-active);
}

.book-constraints-disabled-note {
  margin-top: 8px;
  color: var(--settings-sidebar-workflow-text-supporting);
  font-size: 12px;
}

.navigation-buttons {
  display: flex;
  gap: 10px;
  margin-top: 16px;
}

.navigation-buttons button {
  flex: 1;
  min-height: 38px;
  border: none;
  border-radius: 8px;
  background: var(--settings-sidebar-workflow-surface-selected);
  color: var(--color-text-inverse);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.navigation-buttons button:disabled {
  background: var(--settings-sidebar-workflow-surface-overlay);
  cursor: not-allowed;
}

.navigation-buttons button:hover:not(:disabled) {
  background: var(--settings-sidebar-workflow-surface-inverse);
}
</style>
