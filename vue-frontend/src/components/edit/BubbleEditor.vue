<template>
  <div class="edit-panel-content">
    <div class="text-column original-text-column text-block">
      <div class="text-column-header">
        <span class="column-title">漫画原文</span>
        <UiButton
          variant="toolbar"
          class="re-ocr-btn"
          :class="{ 'is-loading': isOcrLoading }"
          :disabled="isOcrLoading"
          @click="handleOcrRecognize"
          title="重新OCR此气泡"
        >
          <span class="button-icon">🔄</span>
        </UiButton>
      </div>
      <UiTextarea
        ref="originalTextInput"
        v-model="localOriginalText"
        class="text-editor original-editor"
        placeholder="OCR识别的日语原文..."
        spellcheck="false"
        @input="handleOriginalTextChange"
      />
      <div class="text-actions">
        <UiButton variant="toolbar" class="text-action-btn copy-btn" @click="copyOriginalText">📋 复制</UiButton>
        <UiButton
          variant="toolbar"
          class="text-action-btn keyboard-toggle-btn"
          @click="toggleJpKeyboard"
          title="显示/隐藏50音键盘"
        >
          ⌨️ 50音
        </UiButton>
      </div>

      <JapaneseKeyboard
        :visible="showJpKeyboard"
        :default-target="jpKeyboardTarget"
        @close="showJpKeyboard = false"
        @insert="handleKanaInsert"
        @delete="handleKanaDelete"
      />
    </div>

    <div class="text-column translated-text-column text-block">
      <div class="text-column-header">
        <span class="column-title">译文</span>
        <UiButton
          variant="toolbar"
          class="re-translate-btn"
          :class="{ 'is-loading': isTranslateLoading }"
          :disabled="isTranslateLoading"
          @click="handleReTranslate"
          title="重新翻译此气泡"
        >
          <span class="button-icon">🔄</span>
        </UiButton>
      </div>
      <UiTextarea
        ref="translatedTextInput"
        v-model="localTranslatedText"
        class="text-editor translated-editor"
        placeholder="翻译后的中文..."
        spellcheck="false"
        @input="handleTextChange"
      />
      <div class="text-actions">
        <UiButton variant="toolbar" class="text-action-btn copy-btn" @click="copyTranslatedText">📋 复制</UiButton>
      </div>
    </div>

    <div class="style-settings-section text-block">
      <div class="office-toolbar">
        <div class="toolbar-row toolbar-row-top">
          <div class="combo-control font-control">
            <label>字体</label>
            <CustomSelect
              v-model="localFontFamily"
              :groups="fontSelectGroups"
              title="字体"
              @change="handleFontFamilyChange"
            />
          </div>
          <div class="combo-control size-control">
            <label>字号</label>
            <div class="size-input-wrap">
              <UiInput
                type="number"
                v-model.number="localFontSize"
                class="toolbar-fontsize-input"
                :min="FONT_SIZE_MIN"
                :max="FONT_SIZE_MAX"
                :step="FONT_SIZE_STEP"
                title="字号"
                @change="handleFontSizeChange"
              />
              <div class="toolbar-fontsize-btns">
                <UiButton variant="toolbar" class="toolbar-fontsize-btn" @click="increaseFontSize" title="增大字号">
                  A+
                </UiButton>
                <UiButton variant="toolbar" class="toolbar-fontsize-btn" @click="decreaseFontSize" title="减小字号">
                  A-
                </UiButton>
              </div>
            </div>
          </div>
        </div>

        <div class="toolbar-row toolbar-row-actions">
          <div class="toolbar-icon-group" aria-label="排版方向">
            <UiButton
              variant="toolbar"
              class="toolbar-btn"
              :data-active="localTextDirection === 'vertical'"
              @click="setTextDirection('vertical')"
              title="竖向排版"
            >
              <svg viewBox="0 0 16 16" width="16" height="16">
                <path
                  d="M8 2v12M8 2L5 5M8 2l3 3"
                  stroke="currentColor"
                  stroke-width="1.5"
                  fill="none"
                />
              </svg>
            </UiButton>
            <UiButton
              variant="toolbar"
              class="toolbar-btn"
              :data-active="localTextDirection === 'horizontal'"
              @click="setTextDirection('horizontal')"
              title="横向排版"
            >
              <svg viewBox="0 0 16 16" width="16" height="16">
                <path
                  d="M2 8h12M14 8l-3-3M14 8l-3 3"
                  stroke="currentColor"
                  stroke-width="1.5"
                  fill="none"
                />
              </svg>
            </UiButton>
          </div>

          <div class="toolbar-divider vertical"></div>

          <div class="toolbar-color-group">
            <div class="toolbar-color-picker" title="文字颜色">
              <UiButton variant="toolbar" class="toolbar-btn toolbar-color-btn" @click="triggerTextColorPicker">
                <svg viewBox="0 0 16 16" width="16" height="16">
                  <text x="3" y="11" font-size="10" font-weight="bold" fill="currentColor">A</text>
                </svg>
                <span class="color-indicator" :style="{ background: localTextColor }"></span>
              </UiButton>
              <UiInput
                ref="textColorInput"
                type="color"
                v-model="localTextColor"
                class="hidden-color-input"
                @input="handleTextColorChange"
                @change="handleTextColorChange"
              />
            </div>
          </div>

          <div class="toolbar-divider vertical"></div>

          <!-- 背景修复方式选择器 -->
          <div class="toolbar-inpaint-group" title="背景修复方式">
            <CustomSelect
              v-model="localInpaintMethod"
              :options="inpaintMethodOptions"
              @change="handleInpaintMethodChange"
            />

            <div
              class="toolbar-color-picker toolbar-solid-color-options"
              :class="{ hidden: localInpaintMethod !== 'solid' }"
            >
              <UiButton variant="toolbar" class="toolbar-btn toolbar-color-btn" @click="triggerFillColorPicker">
                <svg viewBox="0 0 16 16" width="16" height="16">
                  <rect
                    x="2"
                    y="2"
                    width="12"
                    height="12"
                    rx="2"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="1.2"
                  />
                  <rect x="4" y="4" width="8" height="8" rx="1" fill="currentColor" opacity="0.3" />
                </svg>
                <span class="color-indicator" :style="{ background: localFillColor }"></span>
              </UiButton>
              <UiInput
                ref="fillColorInput"
                type="color"
                v-model="localFillColor"
                class="hidden-color-input"
                @change="handleFillColorChange"
              />
            </div>
          </div>

          <div class="toolbar-divider vertical"></div>

          <div class="toolbar-stroke-cluster">
            <UiButton
              variant="toolbar"
              class="toolbar-btn"
              :data-active="localStrokeEnabled"
              @click="toggleStroke"
              title="文字描边"
            >
              <svg viewBox="0 0 16 16" width="16" height="16">
                <text
                  x="3"
                  y="12"
                  font-size="11"
                  font-weight="bold"
                  stroke="currentColor"
                  stroke-width="2"
                  fill="none"
                >
                  A
                </text>
                <text x="3" y="12" font-size="11" font-weight="bold" fill="currentColor">A</text>
              </svg>
            </UiButton>

            <div
              class="toolbar-color-picker toolbar-stroke-options"
              :class="{ hidden: !localStrokeEnabled }"
              title="描边颜色"
            >
              <UiButton variant="toolbar" class="toolbar-btn toolbar-color-btn" @click="triggerStrokeColorPicker">
                <svg viewBox="0 0 16 16" width="16" height="16">
                  <circle cx="8" cy="8" r="5" fill="none" stroke="currentColor" stroke-width="2" />
                </svg>
                <span class="color-indicator" :style="{ background: localStrokeColor }"></span>
              </UiButton>
              <UiInput
                ref="strokeColorInput"
                type="color"
                v-model="localStrokeColor"
                class="hidden-color-input"
                @change="handleStrokeColorChange"
              />
            </div>

            <div
              class="toolbar-stroke-width toolbar-stroke-options"
              :class="{ hidden: !localStrokeEnabled }"
              title="描边宽度"
            >
              <UiInput
                type="number"
                v-model.number="localStrokeWidth"
                class="toolbar-mini-input"
                min="0"
                max="10"
                @change="handleStrokeWidthChange"
              />
              <span class="toolbar-unit">px</span>
            </div>
          </div>
        </div>

        <div class="toolbar-row toolbar-row-typography">
          <div class="combo-control linespacing-control">
            <label>行间距</label>
            <UiInput
              type="number"
              v-model.number="localLineSpacing"
              class="toolbar-mini-input linespacing-input"
              min="0.5"
              max="3"
              step="0.1"
              title="行间距倍数（0.5 - 3.0）"
              @change="handleLineSpacingChange"
            />
          </div>

          <div class="toolbar-divider vertical"></div>

          <div class="toolbar-icon-group" aria-label="对齐方式" title="横排=水平对齐，竖排=列内字符对齐">
            <UiButton
              variant="toolbar"
              class="toolbar-btn"
              :data-active="localTextAlign === 'start'"
              @click="setTextAlign('start')"
              :title="localTextDirection === 'vertical' ? '顶部对齐' : '左对齐'"
            >
              <svg viewBox="0 0 16 16" width="16" height="16">
                <path d="M2 4h12M2 8h8M2 12h10" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linecap="round" />
              </svg>
            </UiButton>
            <UiButton
              variant="toolbar"
              class="toolbar-btn"
              :data-active="localTextAlign === 'center'"
              @click="setTextAlign('center')"
              title="居中对齐"
            >
              <svg viewBox="0 0 16 16" width="16" height="16">
                <path d="M2 4h12M4 8h8M3 12h10" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linecap="round" />
              </svg>
            </UiButton>
            <UiButton
              variant="toolbar"
              class="toolbar-btn"
              :data-active="localTextAlign === 'end'"
              @click="setTextAlign('end')"
              :title="localTextDirection === 'vertical' ? '底部对齐' : '右对齐'"
            >
              <svg viewBox="0 0 16 16" width="16" height="16">
                <path d="M2 4h12M6 8h8M4 12h10" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linecap="round" />
              </svg>
            </UiButton>
          </div>
        </div>

        <div class="toolbar-row toolbar-row-bottom">
          <div class="toolbar-rotation-group" title="旋转角度">
            <UiButton variant="toolbar" class="toolbar-btn" @click="rotateLeft" title="逆时针旋转">
              <svg viewBox="0 0 16 16" width="16" height="16">
                <path
                  d="M2 8a6 6 0 1 1 1.5 4"
                  stroke="currentColor"
                  stroke-width="1.5"
                  fill="none"
                />
                <path d="M2 5v3.5h3.5" stroke="currentColor" stroke-width="1.5" fill="none" />
              </svg>
            </UiButton>
            <UiInput
              type="number"
              v-model.number="localRotationAngle"
              class="toolbar-mini-input toolbar-rotation-input"
              min="-180"
              max="180"
              step="5"
              @change="handleRotationChange"
            />
            <span class="toolbar-unit">°</span>
            <UiButton variant="toolbar" class="toolbar-btn" @click="rotateRight" title="顺时针旋转">
              <svg viewBox="0 0 16 16" width="16" height="16">
                <path
                  d="M14 8a6 6 0 1 0-1.5 4"
                  stroke="currentColor"
                  stroke-width="1.5"
                  fill="none"
                />
                <path d="M14 5v3.5h-3.5" stroke="currentColor" stroke-width="1.5" fill="none" />
              </svg>
            </UiButton>
            <UiButton variant="toolbar" class="toolbar-btn toolbar-small-btn" @click="resetRotation" title="重置旋转">
              0
            </UiButton>
          </div>

          <div class="toolbar-divider vertical"></div>

          <div class="toolbar-position-group" title="位置调整">
            <UiButton variant="toolbar" class="toolbar-btn" @click="moveLeft" title="左移">
              <svg viewBox="0 0 16 16" width="14" height="14">
                <path d="M10 3L5 8l5 5" stroke="currentColor" stroke-width="1.5" fill="none" />
              </svg>
            </UiButton>
            <UiButton variant="toolbar" class="toolbar-btn" @click="moveRight" title="右移">
              <svg viewBox="0 0 16 16" width="14" height="14">
                <path d="M6 3l5 5-5 5" stroke="currentColor" stroke-width="1.5" fill="none" />
              </svg>
            </UiButton>
            <UiButton variant="toolbar" class="toolbar-btn" @click="moveUp" title="上移">
              <svg viewBox="0 0 16 16" width="14" height="14">
                <path d="M3 10l5-5 5 5" stroke="currentColor" stroke-width="1.5" fill="none" />
              </svg>
            </UiButton>
            <UiButton variant="toolbar" class="toolbar-btn" @click="moveDown" title="下移">
              <svg viewBox="0 0 16 16" width="14" height="14">
                <path d="M3 6l5 5 5-5" stroke="currentColor" stroke-width="1.5" fill="none" />
              </svg>
            </UiButton>
            <span class="toolbar-position-value">
              <span>{{ positionX }}</span>,<span>{{ positionY }}</span>
            </span>
            <UiButton variant="toolbar" class="toolbar-btn toolbar-small-btn" @click="resetPosition" title="重置位置">
              ⌂
            </UiButton>
          </div>
        </div>
      </div>

      <details class="fontsize-presets-panel">
        <summary>字号预设</summary>
        <div class="font-size-presets">
          <UiButton
            variant="toolbar"
            v-for="preset in FONT_SIZE_PRESETS"
            :key="preset"
            class="preset-btn"
            :class="{ active: localFontSize === preset }"
            @click="setFontSize(preset)"
          >
            {{ preset }}
          </UiButton>
        </div>
      </details>

      <div class="edit-action-buttons">
        <UiButton variant="toolbar" class="btn-apply-all" @click="applyToAll">样式同步到本页全部气泡</UiButton>
        <UiButton variant="toolbar" class="btn-reset" @click="resetBubbleEdit">重置</UiButton>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">

import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'

import UiButton from '@/components/ui/UiButton.vue'
import JapaneseKeyboard from './JapaneseKeyboard.vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import { useBubbleEditor, type BubbleEditorEmit, type BubbleEditorProps } from './useBubbleEditor'

const props = defineProps<BubbleEditorProps>()
const emit = defineEmits<BubbleEditorEmit>()

const {
  FONT_SIZE_PRESETS,
  FONT_SIZE_MIN,
  FONT_SIZE_MAX,
  FONT_SIZE_STEP,
  localOriginalText,
  localTranslatedText,
  localFontSize,
  localFontFamily,
  localTextDirection,
  localTextColor,
  localFillColor,
  localStrokeEnabled,
  localStrokeColor,
  localStrokeWidth,
  localRotationAngle,
  localInpaintMethod,
  localLineSpacing,
  localTextAlign,
  originalTextInput,
  translatedTextInput,
  textColorInput,
  fillColorInput,
  strokeColorInput,
  showJpKeyboard,
  jpKeyboardTarget,
  positionX,
  positionY,
  fontSelectGroups,
  inpaintMethodOptions,
  handleOriginalTextChange,
  handleTextChange,
  copyOriginalText,
  copyTranslatedText,
  handleFontSizeChange,
  setFontSize,
  increaseFontSize,
  decreaseFontSize,
  handleFontFamilyChange,
  setTextDirection,
  triggerTextColorPicker,
  handleTextColorChange,
  triggerFillColorPicker,
  handleFillColorChange,
  triggerStrokeColorPicker,
  handleStrokeColorChange,
  toggleStroke,
  handleStrokeWidthChange,
  handleInpaintMethodChange,
  handleLineSpacingChange,
  setTextAlign,
  handleRotationChange,
  rotateLeft,
  rotateRight,
  resetRotation,
  moveLeft,
  moveRight,
  moveUp,
  moveDown,
  resetPosition,
  applyToAll,
  resetBubbleEdit,
  handleOcrRecognize,
  handleReTranslate,
  toggleJpKeyboard,
  handleKanaInsert,
  handleKanaDelete,
} = useBubbleEditor(props, emit)
</script>

<style scoped>
.edit-panel-content {
  --bubble-editor-text-column-divider: #e9ecef;
  --bubble-editor-column-title-text: #495057;
  --bubble-editor-translated-title-text: #27ae60;
  --bubble-editor-text-action-background: #f8f9fa;
  --bubble-editor-text-action-hover-border: #adb5bd;
  --bubble-editor-original-text-background: #f8f8f8;
  --bubble-editor-translated-text-background: #f8fff8;
  --bubble-editor-style-panel-background: #f5f6fb;
  --bubble-editor-style-panel-border: rgba(82, 92, 105, .12);
  --bubble-editor-input-border: #cfd6e4;
  --bubble-editor-input-border-focus: #5b73f2;
  --bubble-editor-input-text: #1f2430;
  --bubble-editor-textarea-focus-ring: rgba(52, 152, 219, .15);
  --bubble-editor-toolbar-border: rgba(96, 110, 140, .22);
  --bubble-editor-toolbar-row-border: rgba(226, 232, 240, .9);
  --bubble-editor-toolbar-row-start: #fbfcff;
  --bubble-editor-toolbar-row-end: #f4f6ff;
  --bubble-editor-toolbar-label: #57607c;
  --bubble-editor-toolbar-divider: rgba(15, 23, 42, .08);
  --bubble-editor-toolbar-shadow: rgba(15, 23, 42, .12);
  --bubble-editor-font-button-background: #f2f4ff;
  --bubble-editor-font-button-hover-background: #dfe4ff;
  --bubble-editor-font-button-border: #d0d7ea;
  --bubble-editor-font-button-hover-border: #9aaefc;
  --bubble-editor-font-button-text: #2f46c8;
  --bubble-editor-font-button-hover-text: #1d34a8;
  --bubble-editor-tool-button-border: rgba(119, 130, 161, .35);
  --bubble-editor-tool-button-hover-border: #7d96ff;
  --bubble-editor-tool-button-active-border: #5670ff;
  --bubble-editor-tool-button-text: #3b3f4f;
  --bubble-editor-tool-button-hover-text: #2b4bff;
  --bubble-editor-tool-button-active-text: #3040c2;
  --bubble-editor-tool-button-inner-shadow: rgba(0, 0, 0, .03);
  --bubble-editor-tool-button-hover-shadow: rgba(107, 125, 255, .25);
  --bubble-editor-tool-button-active-highlight: rgba(255, 255, 255, .7);
  --bubble-editor-tool-button-active-background-start: #e8edff;
  --bubble-editor-tool-button-active-background-end: #d9e2ff;
  --bubble-editor-color-swatch-border: rgba(0, 0, 0, .2);
  --bubble-editor-size-input-focus-ring: rgba(88, 125, 255, .15);
  --bubble-editor-mini-input-focus-ring: rgba(88, 125, 255, .2);
  --bubble-editor-toolbar-unit-text: #596071;
  --bubble-editor-position-chip-text: #4a4f63;
  --bubble-editor-position-chip-background: #eef1ff;
  --bubble-editor-apply-all-button-background-end: #5dade2;
  --bubble-editor-apply-all-button-shadow: rgba(52, 152, 219, .3);

  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 20px;
  padding: 15px;
  overflow: auto;
  min-height: 0;
  background: var(--color-surface-card, var(--color-surface-base));
}

/* 文本块 */
.text-block {
  display: flex;
  flex-direction: column;
  gap: 10px;
  width: 100%;
}

/* 文本列头部 */
.text-column-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--color-border-muted, var(--bubble-editor-text-column-divider));
}

.column-title {
  font-weight: 600;
  font-size: 14px;
  color: var(--color-text-strong, var(--bubble-editor-column-title-text));
}

.original-text-column .column-title {
  color: var(--color-text-danger-strong);
}

.translated-text-column .column-title {
  color: var(--bubble-editor-translated-title-text);
}

.re-ocr-btn,
.re-translate-btn {
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 4px;
  background: var(--color-surface-app, var(--bubble-editor-text-action-background));
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.re-ocr-btn:hover,
.re-translate-btn:hover {
  background: var(--color-surface-accent);
  color: var(--color-text-inverse);
}

.re-ocr-btn.is-loading,
.re-translate-btn.is-loading {
  opacity: 0.7;
  cursor: wait;
  pointer-events: none;
}

.re-ocr-btn.is-loading .button-icon,
.re-translate-btn.is-loading .button-icon {
  display: inline-block;
  animation: spin-icon 1s linear infinite;
}

/* 文本编辑器 */
.text-editor {
  flex: 1;
  width: 100%;
  min-height: 60px;
  padding: 12px;
  border: 2px solid var(--color-border-muted, var(--bubble-editor-text-column-divider));
  border-radius: 8px;
  font-size: 15px;
  line-height: 1.6;
  resize: none;
  transition:
    border-color 0.2s,
    box-shadow 0.2s;
  font-family: inherit;
}

.text-editor:focus {
  outline: none;
  border-color: var(--color-border-accent);
  box-shadow: 0 0 0 3px var(--bubble-editor-textarea-focus-ring);
}

.original-editor {
  background: var(--bubble-editor-original-text-background);
  font-family: var(--font-jp);
}

.translated-editor {
  background: var(--bubble-editor-translated-text-background);
}

.text-actions {
  display: flex;
  gap: 8px;
  margin-top: 8px;
  justify-content: flex-end;
}

.text-action-btn {
  padding: 6px 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 4px;
  background: var(--color-surface-card);
  cursor: pointer;
  font-size: 12px;
  transition: all 0.15s;
}

.text-action-btn:hover {
  background: var(--color-surface-app, var(--bubble-editor-text-action-background));
  border-color: var(--bubble-editor-text-action-hover-border);
}

.keyboard-toggle-btn {
  background: var(--color-surface-app, var(--bubble-editor-text-action-background));
}

.style-settings-section {
  width: 100%;
  padding: 16px;
  background: var(--bubble-editor-style-panel-background);
  border-radius: 10px;
  border: 1px solid var(--bubble-editor-style-panel-border);
  overflow-y: auto;
}

.office-toolbar {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 14px;
  background: var(--color-surface-base);
  border: 1px solid var(--bubble-editor-toolbar-border);
  border-radius: 12px;
  box-shadow: 0 10px 24px var(--bubble-editor-toolbar-shadow);
}

.toolbar-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.toolbar-row-top .combo-control {
  flex: 1;
  min-width: 160px;
}

.toolbar-row-actions,
.toolbar-row-typography,
.toolbar-row-bottom {
  gap: 8px;
  padding: 8px 10px;
  border: 1px solid var(--bubble-editor-toolbar-row-border);
  border-radius: 10px;
  background: linear-gradient(180deg, var(--bubble-editor-toolbar-row-start) 0%, var(--bubble-editor-toolbar-row-end) 100%);
}

.linespacing-input {
  width: 64px;
}

.combo-control {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 11px;
  color: var(--bubble-editor-toolbar-label);
}

.combo-control label {
  font-weight: 600;
  letter-spacing: 0;
}

.size-input-wrap {
  display: flex;
  align-items: center;
  gap: 6px;
}

.toolbar-divider {
  width: 1px;
  height: 26px;
  background: var(--bubble-editor-toolbar-divider);
}

.toolbar-divider.vertical {
  height: 34px;
  margin: 0 2px;
}

.toolbar-icon-group,
.toolbar-color-group,
.toolbar-stroke-cluster {
  display: flex;
  align-items: center;
  gap: 6px;
}

.toolbar-fontsize-input {
  width: 60px;
  height: 36px;
  border: 1px solid var(--bubble-editor-input-border);
  border-radius: 8px;
  padding: 0 8px;
  font-size: 14px;
  text-align: center;
  background: var(--color-surface-base);
  color: var(--bubble-editor-input-text);
}

.toolbar-fontsize-input:focus {
  outline: none;
  border-color: var(--bubble-editor-input-border-focus);
  box-shadow: 0 0 0 2px var(--bubble-editor-size-input-focus-ring);
}

.toolbar-fontsize-btns {
  display: flex;
  gap: 6px;
}

.toolbar-fontsize-btn {
  min-width: 50px;
  height: 34px;
  border: 1px solid var(--bubble-editor-font-button-border);
  border-radius: 8px;
  background: var(--bubble-editor-font-button-background);
  color: var(--bubble-editor-font-button-text);
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  transition: all 0.15s;
}

.toolbar-fontsize-btn:hover {
  background: var(--bubble-editor-font-button-hover-background);
  border-color: var(--bubble-editor-font-button-hover-border);
  color: var(--bubble-editor-font-button-hover-text);
}

.toolbar-btn {
  width: 34px;
  height: 34px;
  border: 1px solid var(--bubble-editor-tool-button-border);
  border-radius: 8px;
  background: var(--color-surface-base);
  color: var(--bubble-editor-tool-button-text);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.12s;
  padding: 0;
  box-shadow: inset 0 -1px 0 var(--bubble-editor-tool-button-inner-shadow);
}

.toolbar-btn:hover {
  border-color: var(--bubble-editor-tool-button-hover-border);
  color: var(--bubble-editor-tool-button-hover-text);
  box-shadow: 0 2px 8px var(--bubble-editor-tool-button-hover-shadow);
}

.toolbar-btn[data-active='true'],
.toolbar-btn.active {
  background: linear-gradient(135deg, var(--bubble-editor-tool-button-active-background-start), var(--bubble-editor-tool-button-active-background-end));
  border-color: var(--bubble-editor-tool-button-active-border);
  color: var(--bubble-editor-tool-button-active-text);
  box-shadow: inset 0 1px 0 var(--bubble-editor-tool-button-active-highlight);
}

.toolbar-btn:active {
  transform: scale(0.95);
}

.toolbar-btn svg {
  pointer-events: none;
}

.toolbar-small-btn {
  width: 24px;
  height: 24px;
  font-size: 11px;
  font-weight: 600;
}

.toolbar-color-picker {
  position: relative;
  display: inline-flex;
}

.toolbar-color-btn {
  flex-direction: column;
  gap: 4px;
  height: 34px;
  padding: 4px;
}

.color-indicator {
  width: 26px;
  height: 6px;
  border-radius: 999px;
  border: 1px solid var(--bubble-editor-color-swatch-border);
}

.hidden-color-input {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}

.toolbar-stroke-options {
  transition: opacity 0.2s;
}

.toolbar-stroke-options.hidden {
  opacity: 0.4;
  pointer-events: none;
}

.toolbar-inpaint-group {
  display: flex;
  align-items: center;
  gap: 6px;
}

.toolbar-solid-color-options {
  transition:
    opacity 0.2s,
    visibility 0.2s;
}

.toolbar-solid-color-options.hidden {
  opacity: 0;
  visibility: hidden;
  pointer-events: none;
}

.toolbar-stroke-width {
  display: flex;
  align-items: center;
  gap: 4px;
}

.toolbar-mini-input {
  width: 46px;
  height: 32px;
  border: 1px solid var(--bubble-editor-input-border);
  border-radius: 6px;
  padding: 0 6px;
  font-size: 12px;
  text-align: center;
  background: var(--color-surface-base);
  color: var(--bubble-editor-input-text);
}

.toolbar-mini-input:focus {
  outline: none;
  border-color: var(--bubble-editor-input-border-focus);
  box-shadow: 0 0 0 2px var(--bubble-editor-mini-input-focus-ring);
}

.toolbar-unit {
  font-size: 11px;
  color: var(--bubble-editor-toolbar-unit-text);
}

.toolbar-rotation-group {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.toolbar-rotation-input {
  width: 58px;
}

.toolbar-position-group {
  display: flex;
  align-items: center;
  gap: 6px;
}

.toolbar-position-value {
  font-size: 12px;
  color: var(--bubble-editor-position-chip-text);
  min-width: 48px;
  text-align: center;
  padding: 0 6px;
  border-radius: 6px;
  background: var(--bubble-editor-position-chip-background);
}

.fontsize-presets-panel {
  margin-top: 12px;
  border-top: 1px solid var(--color-border-muted, var(--color-border-default));
  padding-top: 12px;
}

.fontsize-presets-panel summary {
  cursor: pointer;
  font-size: 13px;
  color: var(--color-text-strong, var(--bubble-editor-column-title-text));
  font-weight: 500;
  padding: 4px 0;
}

.font-size-presets {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 10px;
}

.preset-btn {
  padding: 6px 12px;
  background: var(--bubble-editor-font-button-background);
  border: 1px solid var(--bubble-editor-font-button-border);
  border-radius: 6px;
  color: var(--bubble-editor-font-button-text);
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s;
}

.preset-btn:hover {
  background: var(--bubble-editor-font-button-hover-background);
  border-color: var(--bubble-editor-font-button-hover-border);
}

.preset-btn.active {
  background: linear-gradient(135deg, var(--bubble-editor-tool-button-active-background-start), var(--bubble-editor-tool-button-active-background-end));
  border-color: var(--bubble-editor-tool-button-active-border);
  color: var(--bubble-editor-tool-button-active-text);
}

.edit-action-buttons {
  display: flex;
  gap: 10px;
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid var(--color-border-muted, var(--color-border-default));
}

.btn-apply-all,
.btn-reset {
  flex: 1;
  padding: 10px 16px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-apply-all {
  background: linear-gradient(135deg, var(--color-surface-accent) 0%, var(--bubble-editor-apply-all-button-background-end) 100%);
  border: none;
  color: var(--color-text-inverse);
}

.btn-apply-all:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px var(--bubble-editor-apply-all-button-shadow);
}

.btn-reset {
  background: var(--color-surface-card, var(--color-surface-base));
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  color: var(--color-text-strong, var(--bubble-editor-column-title-text));
}

.btn-reset:hover {
  background: var(--color-surface-app, var(--bubble-editor-text-action-background));
  border-color: var(--bubble-editor-text-action-hover-border);
}
</style>
