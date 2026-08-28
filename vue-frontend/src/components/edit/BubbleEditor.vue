<template>
  <fieldset
    v-if="bubble"
    class="bubble-editor"
    :disabled="disabled"
    :aria-busy="disabled ? 'true' : undefined"
  >
    <div class="bubble-editor__text-panel bubble-editor__text-panel--original">
      <div class="bubble-editor__text-panel-header">
        <span class="bubble-editor__text-panel-title">漫画原文</span>
        <UiIconButton
          variant="soft"
          size="xs"
          class="bubble-editor__refresh-action"
          :class="{ 'bubble-editor__refresh-action--loading': isOcrLoading }"
          :disabled="isOcrLoading"
          label="重新OCR此气泡"
          title="重新OCR此气泡"
          @click="handleOcrRecognize"
        >
          <span class="bubble-editor__refresh-icon bubble-editor__emoji-icon" aria-hidden="true">🔄</span>
        </UiIconButton>
      </div>
      <UiTextarea
        ref="originalTextInput"
        :model-value="localOriginalText"
        :rows="2"
        class="bubble-editor__textarea bubble-editor__textarea--original"
        placeholder="OCR识别的日语原文..."
        spellcheck="false"
        @update:model-value="handleOriginalTextChange"
      />
      <div class="bubble-editor__text-actions">
        <UiButton variant="toolbar" class="bubble-editor__text-action bubble-editor__text-action--copy" @click="copyOriginalText">
          <span class="bubble-editor__emoji-icon" aria-hidden="true">📋</span>
          <span>复制</span>
        </UiButton>
        <UiButton
          variant="toolbar"
          class="bubble-editor__text-action bubble-editor__text-action--keyboard"
          @click="toggleJpKeyboard"
          title="显示/隐藏50音键盘"
        >
          <span class="bubble-editor__emoji-icon" aria-hidden="true">⌨️</span>
          <span>50音</span>
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

    <div class="bubble-editor__text-panel bubble-editor__text-panel--translated">
      <div class="bubble-editor__text-panel-header">
        <span class="bubble-editor__text-panel-title">译文</span>
        <UiIconButton
          variant="soft"
          size="xs"
          class="bubble-editor__refresh-action"
          :class="{ 'bubble-editor__refresh-action--loading': isTranslateLoading }"
          :disabled="isTranslateLoading"
          label="重新翻译此气泡"
          title="重新翻译此气泡"
          @click="handleReTranslate"
        >
          <span class="bubble-editor__refresh-icon bubble-editor__emoji-icon" aria-hidden="true">🔄</span>
        </UiIconButton>
      </div>
      <UiTextarea
        ref="translatedTextInput"
        :model-value="localTranslatedText"
        :rows="2"
        class="bubble-editor__textarea bubble-editor__textarea--translated"
        placeholder="翻译后的中文..."
        spellcheck="false"
        @update:model-value="handleTextChange"
      />
      <div class="bubble-editor__text-actions">
        <UiButton variant="toolbar" class="bubble-editor__text-action bubble-editor__text-action--copy" @click="copyTranslatedText">
          <span class="bubble-editor__emoji-icon" aria-hidden="true">📋</span>
          <span>复制</span>
        </UiButton>
      </div>
    </div>

    <div class="bubble-editor__style-section">
      <div class="bubble-editor__toolbar">
        <div class="bubble-editor__toolbar-row bubble-editor__toolbar-row--top">
          <UiField class="bubble-editor__toolbar-field bubble-editor__toolbar-field--font" variant="editor" label="字体" control-id="bubbleFontFamily">
            <UiCombobox
              input-id="bubbleFontFamily"
              aria-label="字体"
              v-model="localFontFamily"
              :groups="fontSelectGroups"
              title="字体"
              @change="handleFontFamilyChange"
            />
          </UiField>
          <UiField
            class="bubble-editor__toolbar-field bubble-editor__toolbar-field--size"
            variant="editor"
            label="字号"
            control-id="bubbleFontSize"
          >
            <UiNumberField
              input-id="bubbleFontSize"
              v-model="localFontSize"
              class="bubble-editor__number-field bubble-editor__number-field--font"
              variant="editor"
              :min="FONT_SIZE_MIN"
              :step="FONT_SIZE_STEP"
              controls
              controls-placement="after"
              decrement-text="A-"
              increment-text="A+"
              aria-label="字号"
              title="字号"
              decrement-label="减小字号"
              increment-label="增大字号"
              @change="handleFontSizeChange"
            />
          </UiField>
        </div>

        <div class="bubble-editor__toolbar-row bubble-editor__toolbar-row--actions">
          <div class="bubble-editor__toolbar-icon-group" aria-label="排版方向">
            <UiIconButton
              variant="soft"
              size="sm"
              class="bubble-editor__toolbar-action"
              :active="localTextDirection === 'vertical'"
              :pressed="localTextDirection === 'vertical'"
              label="竖向排版"
              title="竖向排版"
              @click="setTextDirection('vertical')"
            >
              <UiIcon name="arrow-up" size="16" />
            </UiIconButton>
            <UiIconButton
              variant="soft"
              size="sm"
              class="bubble-editor__toolbar-action"
              :active="localTextDirection === 'horizontal'"
              :pressed="localTextDirection === 'horizontal'"
              label="横向排版"
              title="横向排版"
              @click="setTextDirection('horizontal')"
            >
              <UiIcon name="arrow-right" size="16" />
            </UiIconButton>
          </div>

          <div class="bubble-editor__toolbar-divider bubble-editor__toolbar-divider--vertical"></div>

          <div class="bubble-editor__toolbar-color-group">
            <div class="bubble-editor__toolbar-color-picker" title="文字颜色">
              <UiIconButton variant="soft" size="sm" class="bubble-editor__toolbar-action bubble-editor__toolbar-color-action" label="文字颜色" title="文字颜色" @click="triggerTextColorPicker">
                <UiIcon name="type" size="16" />
                <span class="bubble-editor__color-indicator" :style="{ background: localTextColor }"></span>
              </UiIconButton>
              <UiColorInput
                ref="textColorInput"
                :model-value="localTextColor"
                hidden
                aria-label="文字颜色"
                title="文字颜色"
                @update:model-value="handleTextColorChange"
              />
            </div>
          </div>

          <div class="bubble-editor__toolbar-divider bubble-editor__toolbar-divider--vertical"></div>

          <div class="bubble-editor__toolbar-inpaint-group" title="背景修复方式">
            <UiSelect
              v-model="localInpaintMethod"
              :options="inpaintMethodOptions"
              aria-label="背景修复方式"
              @change="handleInpaintMethodChange"
            />

            <div
              v-if="localInpaintMethod === 'solid'"
              class="bubble-editor__toolbar-color-picker bubble-editor__toolbar-solid-color-options"
            >
              <UiIconButton variant="soft" size="sm" class="bubble-editor__toolbar-action bubble-editor__toolbar-color-action" label="背景填充颜色" title="背景填充颜色" @click="triggerFillColorPicker">
                <UiIcon name="square" size="16" />
                <span class="bubble-editor__color-indicator" :style="{ background: localFillColor }"></span>
              </UiIconButton>
              <UiColorInput
                ref="fillColorInput"
                :model-value="localFillColor"
                hidden
                aria-label="背景填充颜色"
                title="背景填充颜色"
                @update:model-value="handleFillColorChange"
              />
            </div>
          </div>

          <div class="bubble-editor__toolbar-divider bubble-editor__toolbar-divider--vertical"></div>

          <div class="bubble-editor__toolbar-stroke-cluster">
            <UiIconButton
              variant="soft"
              size="sm"
              class="bubble-editor__toolbar-action"
              :active="localStrokeEnabled"
              :pressed="localStrokeEnabled"
              label="文字描边"
              title="文字描边"
              @click="toggleStroke"
            >
              <UiIcon name="case-sensitive" size="16" />
            </UiIconButton>

            <div
              v-if="localStrokeEnabled"
              class="bubble-editor__toolbar-color-picker bubble-editor__toolbar-stroke-options"
              title="描边颜色"
            >
              <UiIconButton variant="soft" size="sm" class="bubble-editor__toolbar-action bubble-editor__toolbar-color-action" label="描边颜色" title="描边颜色" @click="triggerStrokeColorPicker">
                <UiIcon name="circle" size="16" />
                <span class="bubble-editor__color-indicator" :style="{ background: localStrokeColor }"></span>
              </UiIconButton>
              <UiColorInput
                ref="strokeColorInput"
                :model-value="localStrokeColor"
                hidden
                aria-label="描边颜色"
                title="描边颜色"
                @update:model-value="handleStrokeColorChange"
              />
            </div>

            <div
              v-if="localStrokeEnabled"
              class="bubble-editor__toolbar-stroke-width bubble-editor__toolbar-stroke-options"
              title="描边宽度"
            >
              <UiNumberField
                v-model="localStrokeWidth"
                class="bubble-editor__number-field bubble-editor__number-field--compact"
                variant="editor"
                :min="0"
                aria-label="描边宽度"
                @change="handleStrokeWidthChange"
              />
              <span class="bubble-editor__toolbar-unit">px</span>
            </div>
          </div>
        </div>

        <div class="bubble-editor__toolbar-row bubble-editor__toolbar-row--typography">
          <UiField
            class="bubble-editor__toolbar-field bubble-editor__toolbar-field--line-spacing"
            variant="editor"
            label="行间距"
            control-id="bubbleLineSpacing"
          >
            <UiNumberField
              input-id="bubbleLineSpacing"
              v-model="localLineSpacing"
              class="bubble-editor__number-field bubble-editor__number-field--compact"
              variant="editor"
              :min="0.1"
              :step="0.1"
              aria-label="行间距"
              title="行间距倍数（必须大于 0）"
              @change="handleLineSpacingChange"
            />
          </UiField>

          <div class="bubble-editor__toolbar-divider bubble-editor__toolbar-divider--vertical"></div>

          <div class="bubble-editor__toolbar-icon-group" aria-label="行内对齐" title="横排控制每行左右位置，竖排控制每列上下位置">
            <UiIconButton
              variant="soft"
              size="sm"
              class="bubble-editor__toolbar-action"
              :active="localInlineAlign === 'start'"
              :pressed="localInlineAlign === 'start'"
              :label="localTextDirection === 'vertical' ? '顶部对齐' : '左对齐'"
              :title="localTextDirection === 'vertical' ? '顶部对齐' : '左对齐'"
              @click="setInlineAlign('start')"
            >
              <UiIcon :name="localTextDirection === 'vertical' ? 'align-vertical-start' : 'align-left'" size="16" />
            </UiIconButton>
            <UiIconButton
              variant="soft"
              size="sm"
              class="bubble-editor__toolbar-action"
              :active="localInlineAlign === 'center'"
              :pressed="localInlineAlign === 'center'"
              label="行内居中"
              title="行内居中"
              @click="setInlineAlign('center')"
            >
              <UiIcon :name="localTextDirection === 'vertical' ? 'align-vertical-center' : 'align-center'" size="16" />
            </UiIconButton>
            <UiIconButton
              variant="soft"
              size="sm"
              class="bubble-editor__toolbar-action"
              :active="localInlineAlign === 'end'"
              :pressed="localInlineAlign === 'end'"
              :label="localTextDirection === 'vertical' ? '底部对齐' : '右对齐'"
              :title="localTextDirection === 'vertical' ? '底部对齐' : '右对齐'"
              @click="setInlineAlign('end')"
            >
              <UiIcon :name="localTextDirection === 'vertical' ? 'align-vertical-end' : 'align-right'" size="16" />
            </UiIconButton>
          </div>

          <div class="bubble-editor__toolbar-divider bubble-editor__toolbar-divider--vertical"></div>

          <div class="bubble-editor__toolbar-icon-group" aria-label="文本块对齐" title="控制整个文本块在气泡内的位置">
            <UiIconButton
              variant="soft"
              size="sm"
              class="bubble-editor__toolbar-action"
              :active="localBlockAlign === 'start'"
              :pressed="localBlockAlign === 'start'"
              :label="localTextDirection === 'vertical' ? '文本块靠右' : '文本块靠上'"
              :title="localTextDirection === 'vertical' ? '文本块靠右' : '文本块靠上'"
              @click="setBlockAlign('start')"
            >
              <UiIcon :name="localTextDirection === 'vertical' ? 'align-horizontal-end' : 'align-vertical-start'" size="16" />
            </UiIconButton>
            <UiIconButton
              variant="soft"
              size="sm"
              class="bubble-editor__toolbar-action"
              :active="localBlockAlign === 'center'"
              :pressed="localBlockAlign === 'center'"
              label="文本块居中"
              title="文本块居中"
              @click="setBlockAlign('center')"
            >
              <UiIcon :name="localTextDirection === 'vertical' ? 'align-horizontal-center' : 'align-vertical-center'" size="16" />
            </UiIconButton>
            <UiIconButton
              variant="soft"
              size="sm"
              class="bubble-editor__toolbar-action"
              :active="localBlockAlign === 'end'"
              :pressed="localBlockAlign === 'end'"
              :label="localTextDirection === 'vertical' ? '文本块靠左' : '文本块靠下'"
              :title="localTextDirection === 'vertical' ? '文本块靠左' : '文本块靠下'"
              @click="setBlockAlign('end')"
            >
              <UiIcon :name="localTextDirection === 'vertical' ? 'align-horizontal-start' : 'align-vertical-end'" size="16" />
            </UiIconButton>
          </div>
        </div>

        <div class="bubble-editor__toolbar-row bubble-editor__toolbar-row--bottom">
          <div class="bubble-editor__toolbar-rotation-group" title="旋转角度">
            <UiIconButton variant="soft" size="sm" class="bubble-editor__toolbar-action" label="逆时针旋转" title="逆时针旋转" @click="rotateLeft">
              <UiIcon name="rotate-ccw" size="16" />
            </UiIconButton>
            <UiNumberField
              v-model="localRotationAngle"
              class="bubble-editor__number-field bubble-editor__number-field--rotation"
              variant="editor"
              :min="-180"
              :max="180"
              :step="5"
              aria-label="旋转角度"
              @change="handleRotationChange"
            />
            <span class="bubble-editor__toolbar-unit">°</span>
            <UiIconButton variant="soft" size="sm" class="bubble-editor__toolbar-action" label="顺时针旋转" title="顺时针旋转" @click="rotateRight">
              <UiIcon name="rotate-cw" size="16" />
            </UiIconButton>
            <UiIconButton variant="soft" size="xs" class="bubble-editor__toolbar-action" label="重置旋转" title="重置旋转" @click="resetRotation">
              0
            </UiIconButton>
          </div>

          <div class="bubble-editor__toolbar-divider bubble-editor__toolbar-divider--vertical"></div>

          <div class="bubble-editor__toolbar-position-group" title="位置调整">
            <UiIconButton variant="soft" size="sm" class="bubble-editor__toolbar-action" label="左移" title="左移" @click="moveLeft">
              <UiIcon name="arrow-left" size="14" />
            </UiIconButton>
            <UiIconButton variant="soft" size="sm" class="bubble-editor__toolbar-action" label="右移" title="右移" @click="moveRight">
              <UiIcon name="arrow-right" size="14" />
            </UiIconButton>
            <UiIconButton variant="soft" size="sm" class="bubble-editor__toolbar-action" label="上移" title="上移" @click="moveUp">
              <UiIcon name="arrow-up" size="14" />
            </UiIconButton>
            <UiIconButton variant="soft" size="sm" class="bubble-editor__toolbar-action" label="下移" title="下移" @click="moveDown">
              <UiIcon name="arrow-down" size="14" />
            </UiIconButton>
            <span class="bubble-editor__toolbar-position-value">
              <span>{{ positionX }}</span>,<span>{{ positionY }}</span>
            </span>
            <UiIconButton variant="soft" size="xs" class="bubble-editor__toolbar-action" label="重置位置" title="重置位置" @click="resetPosition">
              <UiIcon name="home" size="14" />
            </UiIconButton>
          </div>
        </div>
      </div>

      <details class="bubble-editor__font-size-presets-panel">
        <summary class="bubble-editor__font-size-presets-title">字号预设</summary>
        <div class="bubble-editor__font-size-presets">
          <UiButton
            variant="toolbar"
            v-for="preset in FONT_SIZE_PRESETS"
            :key="preset"
            class="bubble-editor__font-size-preset"
            :class="{ 'bubble-editor__font-size-preset--active': localFontSize === preset }"
            :aria-pressed="localFontSize === preset"
            @click="setFontSize(preset)"
          >
            {{ preset }}
          </UiButton>
        </div>
      </details>

      <div class="bubble-editor__footer-actions">
        <UiButton
          variant="primary"
          tone="success"
          size="sm"
          block
          class="bubble-editor__footer-action"
          @click="applyToAll"
        >
          样式同步到本页全部气泡
        </UiButton>
      </div>
    </div>
  </fieldset>
  <div v-else class="bubble-editor bubble-editor--empty">
    <UiIcon name="mouse-pointer" size="24" aria-hidden="true" />
    <span>请选择一个气泡进行编辑</span>
  </div>
</template>

<script setup lang="ts">

import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiColorInput from '@/components/ui/UiColorInput.vue'

import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import JapaneseKeyboard from './JapaneseKeyboard.vue'
import UiCombobox from '@/components/ui/UiCombobox.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import { useBubbleEditor, type BubbleEditorEmit, type BubbleEditorProps } from './useBubbleEditor'

const props = defineProps<BubbleEditorProps>()
const emit = defineEmits<BubbleEditorEmit>()

const {
  FONT_SIZE_PRESETS,
  FONT_SIZE_MIN,
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
  localInlineAlign,
  localBlockAlign,
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
  setInlineAlign,
  setBlockAlign,
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
  handleOcrRecognize,
  handleReTranslate,
  toggleJpKeyboard,
  handleKanaInsert,
  handleKanaDelete,
} = useBubbleEditor(props, emit)
</script>

<style scoped>
.bubble-editor {
  --bubble-editor-translated-title-text: var(--color-surface-success);
  --bubble-editor-text-action-hover-border: var(--color-text-disabled);
  --bubble-editor-original-text-background: var(--color-surface-quiet);
  --bubble-editor-translated-text-background: color-mix(in srgb, var(--color-status-success) 6%, var(--color-surface-base));
  --bubble-editor-style-panel-background: color-mix(in srgb, var(--color-action-brand) 5%, var(--color-surface-app));
  --bubble-editor-style-panel-border: color-mix(in srgb, var(--color-text-heading) 12%, transparent);
  --bubble-editor-textarea-focus-ring: color-mix(in srgb, var(--color-action-primary) 15%, transparent);
  --bubble-editor-toolbar-border: color-mix(in srgb, var(--color-text-heading) 22%, transparent);
  --bubble-editor-toolbar-row-border: color-mix(in srgb, var(--color-border-muted) 90%, transparent);
  --bubble-editor-toolbar-row-start: var(--color-surface-base);
  --bubble-editor-toolbar-row-end: color-mix(in srgb, var(--color-action-brand) 4%, var(--color-surface-base));
  --bubble-editor-toolbar-label: var(--color-text-secondary);
  --bubble-editor-toolbar-divider: color-mix(in srgb, var(--color-overlay-backdrop-solid) 8%, transparent);
  --bubble-editor-toolbar-shadow: color-mix(in srgb, var(--color-overlay-backdrop-solid) 12%, transparent);
  --bubble-editor-font-button-background: color-mix(in srgb, var(--color-action-brand) 8%, var(--color-surface-base));
  --bubble-editor-font-button-hover-background: color-mix(in srgb, var(--color-action-brand) 16%, var(--color-surface-base));
  --bubble-editor-font-button-border: var(--color-border-muted);
  --bubble-editor-font-button-hover-border: color-mix(in srgb, var(--color-action-brand) 45%, var(--color-border-muted));
  --bubble-editor-font-button-text: var(--color-text-brand);
  --bubble-editor-tool-button-active-border: var(--color-action-brand);
  --bubble-editor-tool-button-active-text: var(--color-action-brand-strong);
  --bubble-editor-tool-button-active-background-start: color-mix(in srgb, var(--color-action-brand) 12%, var(--color-surface-base));
  --bubble-editor-tool-button-active-background-end: color-mix(in srgb, var(--color-action-brand) 18%, var(--color-surface-base));
  --bubble-editor-color-swatch-border: color-mix(in srgb, var(--color-overlay-backdrop-solid) 20%, transparent);
  --bubble-editor-toolbar-unit-text: var(--color-text-secondary);
  --bubble-editor-position-chip-text: var(--color-text-default);
  --bubble-editor-position-chip-background: color-mix(in srgb, var(--color-action-brand) 10%, var(--color-surface-base));

  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 20px;
  padding: 15px;
  overflow: auto;
  min-height: 0;
  background: var(--color-surface-card);
  min-width: 0;
  margin: 0;
  border: 0;
}

.bubble-editor:disabled {
  cursor: wait;
}

.bubble-editor--empty {
  align-items: center;
  justify-content: center;
  color: var(--color-text-muted);
  font-size: 13px;
}

.bubble-editor__text-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  width: 100%;
}

.bubble-editor__text-panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--color-border-muted);
}

.bubble-editor__text-panel-title {
  font-weight: 600;
  font-size: 14px;
  color: var(--color-text-strong);
}

.bubble-editor__text-panel--original .bubble-editor__text-panel-title {
  color: var(--color-text-danger-strong);
}

.bubble-editor__text-panel--translated .bubble-editor__text-panel-title {
  color: var(--bubble-editor-translated-title-text);
}

.bubble-editor__refresh-action--loading .bubble-editor__refresh-icon {
  display: inline-block;
  animation: spin-icon 1s linear infinite;
}

.bubble-editor__emoji-icon {
  font-size: 14px;
  line-height: 1;
}

.bubble-editor__textarea {
  flex: 1;
  width: 100%;
  min-height: 60px;
  padding: 12px;
  border: 2px solid var(--color-border-muted);
  border-radius: 8px;
  font-size: 15px;
  line-height: 1.6;
  resize: none;
  transition:
    border-color 0.2s,
    box-shadow 0.2s;
  font-family: inherit;
}

.bubble-editor__textarea:focus {
  outline: none;
  border-color: var(--color-border-accent);
  box-shadow: 0 0 0 3px var(--bubble-editor-textarea-focus-ring);
}

.bubble-editor__textarea--original {
  background: var(--bubble-editor-original-text-background);
  font-family: var(--font-jp);
}

.bubble-editor__textarea--translated {
  background: var(--bubble-editor-translated-text-background);
}

.bubble-editor__text-actions {
  display: flex;
  gap: 8px;
  margin-top: 8px;
  justify-content: flex-end;
}

.bubble-editor__text-action {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border: 1px solid var(--color-border-muted);
  border-radius: 4px;
  background: var(--color-surface-card);
  cursor: pointer;
  font-size: 12px;
  transition: all 0.15s;
}

.bubble-editor__text-action:hover {
  background: var(--color-surface-app);
  border-color: var(--bubble-editor-text-action-hover-border);
}

.bubble-editor__text-action--keyboard {
  background: var(--color-surface-app);
}

.bubble-editor__style-section {
  width: 100%;
  padding: 16px;
  background: var(--bubble-editor-style-panel-background);
  border-radius: 10px;
  border: 1px solid var(--bubble-editor-style-panel-border);
  overflow-y: auto;
}

.bubble-editor__toolbar {
  --ui-icon-button-active-background: linear-gradient(135deg, var(--bubble-editor-tool-button-active-background-start), var(--bubble-editor-tool-button-active-background-end));
  --ui-icon-button-active-border: var(--bubble-editor-tool-button-active-border);
  --ui-icon-button-active-color: var(--bubble-editor-tool-button-active-text);
  --ui-icon-button-active-hover-background: linear-gradient(135deg, var(--bubble-editor-tool-button-active-background-start), var(--bubble-editor-tool-button-active-background-end));
  --ui-icon-button-active-hover-border: var(--bubble-editor-tool-button-active-border);

  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 14px;
  background: var(--color-surface-base);
  border: 1px solid var(--bubble-editor-toolbar-border);
  border-radius: 12px;
  box-shadow: 0 10px 24px var(--bubble-editor-toolbar-shadow);
}

.bubble-editor__toolbar-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.bubble-editor__toolbar-row--top .bubble-editor__toolbar-field {
  flex: 1;
  min-width: 160px;
}

.bubble-editor__toolbar-row--actions,
.bubble-editor__toolbar-row--typography,
.bubble-editor__toolbar-row--bottom {
  gap: 8px;
  padding: 8px 10px;
  border: 1px solid var(--bubble-editor-toolbar-row-border);
  border-radius: 10px;
  background: linear-gradient(180deg, var(--bubble-editor-toolbar-row-start) 0%, var(--bubble-editor-toolbar-row-end) 100%);
}

.bubble-editor__toolbar-field {
  --ui-field-editor-label-color: var(--bubble-editor-toolbar-label);
  --ui-field-editor-label-font-size: 11px;

  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 11px;
  color: var(--bubble-editor-toolbar-label);
}

.bubble-editor__toolbar-divider {
  width: 1px;
  height: 26px;
  background: var(--bubble-editor-toolbar-divider);
}

.bubble-editor__toolbar-divider--vertical {
  height: 34px;
  margin: 0 2px;
}

.bubble-editor__toolbar-icon-group,
.bubble-editor__toolbar-color-group,
.bubble-editor__toolbar-stroke-cluster {
  display: flex;
  align-items: center;
  gap: 6px;
}

.bubble-editor__toolbar-color-picker {
  position: relative;
  display: inline-flex;
}

.bubble-editor__toolbar-color-action {
  flex-direction: column;
  gap: 4px;
}

.bubble-editor__color-indicator {
  width: 26px;
  height: 6px;
  border-radius: 999px;
  border: 1px solid var(--bubble-editor-color-swatch-border);
}

.bubble-editor__toolbar-inpaint-group {
  display: flex;
  align-items: center;
  gap: 6px;
}

.bubble-editor__toolbar-stroke-width {
  display: flex;
  align-items: center;
  gap: 4px;
}

.bubble-editor__number-field--font {
  --ui-number-field-input-width: 60px;
  --ui-number-field-control-width: 50px;
  --ui-number-field-control-height: 34px;
}

.bubble-editor__number-field--compact {
  --ui-number-field-input-width: 48px;
}

.bubble-editor__number-field--rotation {
  --ui-number-field-input-width: 58px;
}

.bubble-editor__toolbar-unit {
  font-size: 11px;
  color: var(--bubble-editor-toolbar-unit-text);
}

.bubble-editor__toolbar-rotation-group {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.bubble-editor__toolbar-position-group {
  display: flex;
  align-items: center;
  gap: 6px;
}

.bubble-editor__toolbar-position-value {
  font-size: 12px;
  color: var(--bubble-editor-position-chip-text);
  min-width: 48px;
  text-align: center;
  padding: 0 6px;
  border-radius: 6px;
  background: var(--bubble-editor-position-chip-background);
}

.bubble-editor__font-size-presets-panel {
  margin-top: 12px;
  border-top: 1px solid var(--color-border-muted);
  padding-top: 12px;
}

.bubble-editor__font-size-presets-title {
  cursor: pointer;
  font-size: 13px;
  color: var(--color-text-strong);
  font-weight: 500;
  padding: 4px 0;
}

.bubble-editor__font-size-presets {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 10px;
}

.bubble-editor__font-size-preset {
  padding: 6px 12px;
  background: var(--bubble-editor-font-button-background);
  border: 1px solid var(--bubble-editor-font-button-border);
  border-radius: 6px;
  color: var(--bubble-editor-font-button-text);
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s;
}

.bubble-editor__font-size-preset:hover {
  background: var(--bubble-editor-font-button-hover-background);
  border-color: var(--bubble-editor-font-button-hover-border);
}

.bubble-editor__font-size-preset--active {
  background: linear-gradient(135deg, var(--bubble-editor-tool-button-active-background-start), var(--bubble-editor-tool-button-active-background-end));
  border-color: var(--bubble-editor-tool-button-active-border);
  color: var(--bubble-editor-tool-button-active-text);
}

.bubble-editor__footer-actions {
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid var(--color-border-muted);
}

.bubble-editor__footer-action {
  min-height: 40px;

  --ui-button-primary-background: linear-gradient(135deg, var(--color-action-primary) 0%, var(--color-action-primary-soft) 100%);
  --ui-button-primary-hover-background: var(--ui-button-primary-background);
  --ui-button-primary-hover-shadow: 0 4px 12px color-mix(in srgb, var(--color-action-primary) 30%, transparent);
}
</style>
