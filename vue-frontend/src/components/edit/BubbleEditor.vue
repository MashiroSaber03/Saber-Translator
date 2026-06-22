<!--
  气泡编辑器组件
  编辑单个气泡的文本、字体、颜色等属性
  编辑器使用 Office 风格浅色主题
  - 支持原文和译文编辑
  - 支持日语软键盘输入
  - 支持单气泡重新OCR识别和翻译
  - 支持样式设置（字号、字体、颜色、描边等）
  - 支持修复方式选择
-->
<template>
  <div class="edit-panel-content">
    <!-- 始终显示编辑面板，不显示"请选择气泡"提示 -->
    <!-- 原文编辑区 -->
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
        <UiButton variant="toolbar" class="copy-btn" @click="copyOriginalText">📋 复制</UiButton>
        <UiButton variant="toolbar" class="keyboard-toggle-btn" @click="toggleJpKeyboard" title="显示/隐藏50音键盘">
          ⌨️ 50音
        </UiButton>
      </div>

      <!-- 50音软键盘 -->
      <JapaneseKeyboard
        :visible="showJpKeyboard"
        :default-target="jpKeyboardTarget"
        @close="showJpKeyboard = false"
        @insert="handleKanaInsert"
        @delete="handleKanaDelete"
      />
    </div>

    <!-- 译文编辑区 -->
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
        <UiButton variant="toolbar" class="copy-btn" @click="copyTranslatedText">📋 复制</UiButton>
      </div>
    </div>

    <!-- 样式设置区 -->
    <div class="style-settings-section text-block">
      <!-- Office风格文字设置工具栏 -->
      <div class="office-toolbar">
        <!-- 第一行：字体 + 字号 -->
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

        <!-- 第二行：样式工具按钮 -->
        <div class="toolbar-row toolbar-row-actions">
          <!-- 排版方向 -->
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

          <!-- 文字颜色 -->
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

            <!-- 纯色填充时的颜色选择器 -->
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

          <!-- 描边设置 -->
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

        <!-- 行间距 + 对齐 -->
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

        <!-- 第三行：旋转 + 位置 -->
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

      <!-- 字号预设快捷按钮（可折叠） -->
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

      <!-- 操作按钮 -->
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

<style scoped>/* ============ 编辑面板内容 - 使用浅色主题 ============ */

.edit-panel-content {
  /* owner tokens: bubble-editor */
  --bubble-editor-border-default: #d0d7ea;
  --bubble-editor-border-strong: #9aaefc;
  --bubble-editor-border-muted: rgba(119, 130, 161, .35);
  --bubble-editor-border-subtle: #7d96ff;
  --bubble-editor-border-hover: #5670ff;
  --bubble-editor-border-active: rgba(0, 0, 0, .2);
  --bubble-editor-shadow-default: rgba(88, 125, 255, .15);
  --bubble-editor-shadow-raised: rgba(0, 0, 0, .03);
  --bubble-editor-shadow-floating: rgba(107, 125, 255, .25);
  --bubble-editor-shadow-strong: rgba(255, 255, 255, .7);
  --bubble-editor-shadow-soft: rgba(88, 125, 255, .2);
  --bubble-editor-shadow-focus: rgba(52, 152, 219, .3);
  --bubble-editor-surface-base: #f2f4ff;
  --bubble-editor-surface-raised: #dfe4ff;
  --bubble-editor-surface-muted: #e8edff;
  --bubble-editor-surface-subtle: #d9e2ff;
  --bubble-editor-surface-hover: #eef1ff;
  --bubble-editor-surface-active: #5dade2;
  --bubble-editor-text-primary: #2f46c8;
  --bubble-editor-text-secondary: #1d34a8;
  --bubble-editor-text-muted: #3b3f4f;
  --bubble-editor-text-subtle: #2b4bff;
  --bubble-editor-text-supporting: #3040c2;
  --bubble-editor-text-disabled: #596071;
  --bubble-editor-text-inverse: #4a4f63;

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
  border-bottom: 2px solid var(--color-border-muted, var(--color-edit-panel-divider));
}

.column-title {
  font-weight: 600;
  font-size: 14px;
  color: var(--color-text-strong, var(--color-edit-panel-text));
}

.original-text-column .column-title {
  color: var(--color-text-danger-strong);
}

.translated-text-column .column-title {
  color: var(--color-edit-panel-success);
}

/* 重新OCR/翻译按钮 */
.re-ocr-btn,
.re-translate-btn {
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 4px;
  background: var(--color-surface-app, var(--color-edit-control-bg));
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.re-ocr-btn:hover,
.re-translate-btn:hover {
  background: var(--color-surface-accent);
  color: var(--color-text-inverse);
}

/* Loading 状态 */
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
  border: 2px solid var(--color-border-muted, var(--color-edit-panel-divider));
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
  box-shadow: 0 0 0 3px var(--shadow-edit-focus-blue);
}

.original-editor {
  background: var(--color-surface-editor-original);
  font-family: var(--font-jp);
}

.translated-editor {
  background: var(--color-edit-translated-bg);
}

/* 文本操作按钮 */
.text-actions {
  display: flex;
  gap: 8px;
  margin-top: 8px;
  justify-content: flex-end;
}

.text-actions button {
  padding: 6px 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 4px;
  background: var(--color-surface-card, white);
  cursor: pointer;
  font-size: 12px;
  transition: all 0.15s;
}

.text-actions button:hover {
  background: var(--color-surface-app, var(--color-edit-control-bg));
  border-color: var(--color-edit-muted-border-hover);
}

.keyboard-toggle-btn {
  background: var(--color-surface-app, var(--color-edit-control-bg));
}

/* ============ 样式设置区 ============ */

.style-settings-section {
  width: 100%;
  padding: 16px;
  background: var(--color-edit-style-bg);
  border-radius: 10px;
  border: 1px solid var(--color-edit-style-border);
  overflow-y: auto;
}

/* ============ Office风格工具栏 ============ */

.office-toolbar {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 14px;
  background: var(--color-surface-base);
  border: 1px solid var(--color-edit-toolbar-border);
  border-radius: 12px;
  box-shadow: 0 10px 24px var(--shadow-edit-toolbar);
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
  border: 1px solid var(--color-edit-toolbar-row-border);
  border-radius: 10px;
  background: linear-gradient(180deg, var(--color-edit-toolbar-row-start) 0%, var(--color-edit-toolbar-row-end) 100%);
}

.linespacing-input {
  width: 64px;
}

.combo-control {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 11px;
  color: var(--color-edit-toolbar-label);
}

.combo-control label {
  font-weight: 600;
  letter-spacing: 0.2px;
}

.size-input-wrap {
  display: flex;
  align-items: center;
  gap: 6px;
}

.toolbar-divider {
  width: 1px;
  height: 26px;
  background: var(--color-edit-toolbar-divider);
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

/* 字体选择器 */
.toolbar-font-select {
  min-width: 160px;
  height: 36px;
  padding: 0 10px;
  border: 1px solid var(--color-edit-input-border);
  border-radius: 8px;
  font-size: 13px;
  background: var(--color-surface-base);
  color: var(--color-edit-input-text);
  cursor: pointer;
  transition:
    border-color 0.15s,
    box-shadow 0.15s;
}

.toolbar-font-select:hover {
  border-color: var(--color-edit-input-border-hover);
}

.toolbar-font-select:focus {
  outline: none;
  border-color: var(--color-edit-input-border-focus);
  box-shadow: 0 0 0 2px var(--shadow-edit-input-focus);
}

/* 字号输入 */
.toolbar-fontsize-input {
  width: 60px;
  height: 36px;
  border: 1px solid var(--color-edit-input-border);
  border-radius: 8px;
  padding: 0 8px;
  font-size: 14px;
  text-align: center;
  background: var(--color-surface-base);
  color: var(--color-edit-input-text);
}

.toolbar-fontsize-input:focus {
  outline: none;
  border-color: var(--color-edit-input-border-focus);
  box-shadow: 0 0 0 2px var(--bubble-editor-shadow-default);
}

.toolbar-fontsize-btns {
  display: flex;
  gap: 6px;
}

.toolbar-fontsize-btn {
  min-width: 50px;
  height: 34px;
  border: 1px solid var(--bubble-editor-border-default);
  border-radius: 8px;
  background: var(--bubble-editor-surface-base);
  color: var(--bubble-editor-text-primary);
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  transition: all 0.15s;
}

.toolbar-fontsize-btn:hover {
  background: var(--bubble-editor-surface-raised);
  border-color: var(--bubble-editor-border-strong);
  color: var(--bubble-editor-text-secondary);
}

/* 工具栏按钮 */
.toolbar-btn {
  width: 34px;
  height: 34px;
  border: 1px solid var(--bubble-editor-border-muted);
  border-radius: 8px;
  background: var(--color-surface-base);
  color: var(--bubble-editor-text-muted);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.12s;
  padding: 0;
  box-shadow: inset 0 -1px 0 var(--bubble-editor-shadow-raised);
}

.toolbar-btn:hover {
  border-color: var(--bubble-editor-border-subtle);
  color: var(--bubble-editor-text-subtle);
  box-shadow: 0 2px 8px var(--bubble-editor-shadow-floating);
}

.toolbar-btn[data-active='true'],
.toolbar-btn.active {
  background: linear-gradient(135deg, var(--bubble-editor-surface-muted), var(--bubble-editor-surface-subtle));
  border-color: var(--bubble-editor-border-hover);
  color: var(--bubble-editor-text-supporting);
  box-shadow: inset 0 1px 0 var(--bubble-editor-shadow-strong);
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

/* 颜色选择器 */
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
  border: 1px solid var(--bubble-editor-border-active);
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

/* 描边选项 */
.toolbar-stroke-options {
  transition: opacity 0.2s;
}

.toolbar-stroke-options.hidden {
  opacity: 0.4;
  pointer-events: none;
}

/* 背景修复方式选择器 */
.toolbar-inpaint-group {
  display: flex;
  align-items: center;
  gap: 6px;
}

.toolbar-inpaint-select {
  height: 34px;
  padding: 0 10px;
  border: 1px solid var(--color-edit-input-border);
  border-radius: 8px;
  font-size: 12px;
  background: var(--color-surface-base);
  color: var(--color-edit-input-text);
  cursor: pointer;
  transition:
    border-color 0.15s,
    box-shadow 0.15s;
}

.toolbar-inpaint-select:hover {
  border-color: var(--color-edit-input-border-hover);
}

.toolbar-inpaint-select:focus {
  outline: none;
  border-color: var(--color-edit-input-border-focus);
  box-shadow: 0 0 0 2px var(--shadow-edit-input-focus);
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
  border: 1px solid var(--color-edit-input-border);
  border-radius: 6px;
  padding: 0 6px;
  font-size: 12px;
  text-align: center;
  background: var(--color-surface-base);
  color: var(--color-edit-input-text);
}

.toolbar-mini-input:focus {
  outline: none;
  border-color: var(--color-edit-input-border-focus);
  box-shadow: 0 0 0 2px var(--bubble-editor-shadow-soft);
}

.toolbar-unit {
  font-size: 11px;
  color: var(--bubble-editor-text-disabled);
}

/* 旋转控制组 */
.toolbar-rotation-group {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.toolbar-rotation-input {
  width: 58px;
}

/* 位置控制组 */
.toolbar-position-group {
  display: flex;
  align-items: center;
  gap: 6px;
}

.toolbar-position-value {
  font-size: 12px;
  color: var(--bubble-editor-text-inverse);
  min-width: 48px;
  text-align: center;
  padding: 0 6px;
  border-radius: 6px;
  background: var(--bubble-editor-surface-hover);
}

/* 字号预设面板 */
.fontsize-presets-panel {
  margin-top: 12px;
  border-top: 1px solid var(--color-border-muted, var(--color-border-default));
  padding-top: 12px;
}

.fontsize-presets-panel summary {
  cursor: pointer;
  font-size: 13px;
  color: var(--color-text-strong, var(--color-edit-panel-text));
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
  background: var(--bubble-editor-surface-base);
  border: 1px solid var(--bubble-editor-border-default);
  border-radius: 6px;
  color: var(--bubble-editor-text-primary);
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s;
}

.preset-btn:hover {
  background: var(--bubble-editor-surface-raised);
  border-color: var(--bubble-editor-border-strong);
}

.preset-btn.active {
  background: linear-gradient(135deg, var(--bubble-editor-surface-muted), var(--bubble-editor-surface-subtle));
  border-color: var(--bubble-editor-border-hover);
  color: var(--bubble-editor-text-supporting);
}

/* 操作按钮 */
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
  background: linear-gradient(135deg, var(--color-surface-accent) 0%, var(--bubble-editor-surface-active) 100%);
  border: none;
  color: white;
}

.btn-apply-all:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px var(--bubble-editor-shadow-focus);
}

.btn-reset {
  background: var(--color-surface-card, var(--color-surface-base));
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  color: var(--color-text-strong, var(--color-edit-panel-text));
}

.btn-reset:hover {
  background: var(--color-surface-app, var(--color-edit-control-bg));
  border-color: var(--color-edit-muted-border-hover);
}
</style>
