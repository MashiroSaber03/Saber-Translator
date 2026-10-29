<template>
  <fieldset
    v-if="bubble"
    class="bubble-editor"
    :disabled="disabled"
    :aria-busy="disabled ? 'true' : undefined"
  >
    <div class="bubble-editor__text-panel bubble-editor__text-panel--original">
      <div class="bubble-editor__text-panel-header">
        <UiButton variant="toolbar" size="sm" class="bubble-editor__original-toggle bubble-editor__text-panel-title" :aria-expanded="showOriginalText" @click="showOriginalText = !showOriginalText">
          <UiIcon :name="showOriginalText ? 'chevron-down' : 'chevron-right'" size="14" />漫画原文
        </UiButton>
        <UiIconButton
          variant="soft"
          size="xs"
          class="bubble-editor__refresh-action"
          :class="{ 'bubble-editor__refresh-action--loading': isOcrLoading }"
          :disabled="isOcrLoading"
          label="重新OCR此气泡"
          title="重新OCR此气泡"
          @click="showOriginalText = true; handleOcrRecognize()"
        >
          <UiIcon class="bubble-editor__refresh-icon" name="refresh" size="14" />
        </UiIconButton>
      </div>
      <div v-show="showOriginalText" class="bubble-editor__original-content">
        <UiTextarea
          ref="originalTextInput"
          :model-value="localOriginalText"
          :rows="2"
          class="bubble-editor__textarea bubble-editor__textarea--original"
          placeholder="识别的漫画原文…"
          spellcheck="false"
          @update:model-value="handleOriginalTextChange"
        />
        <div class="bubble-editor__text-actions">
          <UiButton variant="toolbar" class="bubble-editor__text-action bubble-editor__text-action--copy" @click="copyOriginalText">
            <UiIcon name="copy" size="14" />
            <span>复制</span>
          </UiButton>
          <UiButton
            variant="toolbar"
            class="bubble-editor__text-action bubble-editor__text-action--keyboard"
            @click="toggleJpKeyboard"
            title="显示/隐藏50音键盘"
          >
            <UiIcon name="keyboard" size="14" />
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
          <UiIcon class="bubble-editor__refresh-icon" name="refresh" size="14" />
        </UiIconButton>
      </div>
      <RichTextInput
        ref="translatedTextInput"
        :bubble="bubble"
        :fonts="editorFonts"
        :disabled="disabled"
        @update="handleRichChange"
        @selection="handleRichSelection"
      />
      <div class="bubble-editor__text-actions bubble-editor__text-actions--format">
        <span class="bubble-editor__text-scope" title="选中文字可修改局部样式；点击画布后修改整框文字">{{ textScope }}</span>
        <div class="bubble-editor__format-actions">
          <UiIconButton variant="soft" size="xs" label="加粗" title="加粗选中文字 (Ctrl+B)" :active="bold" :pressed="bold" @click="toggleTextStyle('faux_bold')"><strong>B</strong></UiIconButton>
          <UiIconButton variant="soft" size="xs" label="斜体" title="倾斜选中文字 (Ctrl+I)" :active="italic" :pressed="italic" @click="toggleTextStyle('faux_italic')"><i>I</i></UiIconButton>
          <UiIconButton variant="soft" size="xs" label="下划线" title="选中文字添加下划线 (Ctrl+U)" :active="underline" :pressed="underline" @click="toggleTextStyle('underline')"><u>U</u></UiIconButton>
          <UiIconButton variant="soft" size="xs" label="清除局部样式" title="恢复当前框的字体、字号和颜色，清除选字的局部修饰；保留文字和段落" @click="clearFormatting"><UiIcon name="eraser" size="15" /></UiIconButton>
          <UiButton variant="toolbar" class="bubble-editor__text-action bubble-editor__text-action--copy" @click="copyTranslatedText">
            <UiIcon name="copy" size="14" />
            <span>{{ copyLabel }}</span>
          </UiButton>
        </div>
      </div>
    </div>

    <div class="bubble-editor__style-section">
      <h3 class="bubble-editor__section-title">文字样式</h3>
      <div class="bubble-editor__toolbar">
        <div class="bubble-editor__toolbar-row bubble-editor__toolbar-row--top">
          <UiField class="bubble-editor__toolbar-field" variant="editor" label="字体" control-id="bubbleFontFamily">
            <UiCombobox
              fit
              input-id="bubbleFontFamily"
              aria-label="字体"
              v-model="localFontFamily"
              :groups="fontSelectGroups"
              title="字体"
              @change="handleFontFamilyChange"
              placeholder="多种字体"
            />
          </UiField>
          <UiField
            class="bubble-editor__toolbar-field"
            variant="editor"
            label="字号"
            control-id="bubbleFontSize"
          >
            <template #label-actions>
              <UiIconButton variant="plain" size="xs" label="适配字号" title="按当前文本框重新计算整框字号；保留局部样式，不改变修复范围" @click="emit('fitText')"><UiIcon name="scan-line" size="14" /></UiIconButton>
            </template>
            <UiNumberField
              input-id="bubbleFontSize"
              v-model="localFontSize"
              nullable
              placeholder="混合"
              class="bubble-editor__number-field"
              variant="editor"
              :min="FONT_SIZE_MIN"
              :step="FONT_SIZE_STEP"
              :controls="localFontSize !== null"
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
          <UiField variant="editor" label="方向">
            <div class="bubble-editor__toolbar-icon-group" aria-label="排版方向">
              <UiIconButton
                variant="soft"
                size="md"
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
                size="md"
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
          </UiField>

          <UiField variant="editor" label="文字颜色">
            <div class="bubble-editor__toolbar-color-picker" title="文字颜色">
              <UiIconButton variant="soft" size="md" class="bubble-editor__toolbar-action bubble-editor__toolbar-color-action" label="文字颜色" title="文字颜色" aria-haspopup="dialog" :aria-expanded="colorPopover?.field === 'textColor'" @click="openColorPopover('textColor', $event)">
                <UiIcon name="type" size="16" />
                <span class="bubble-editor__color-indicator" :style="{ background: localTextColor }"></span>
              </UiIconButton>
            </div>
          </UiField>

          <UiField variant="editor" label="描边" class="bubble-editor__stroke-field">
            <div class="bubble-editor__toolbar-stroke-cluster">
              <UiIconButton
                variant="soft"
                size="md"
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
                <UiIconButton variant="soft" size="md" class="bubble-editor__toolbar-action bubble-editor__toolbar-color-action" label="描边颜色" title="描边颜色" aria-haspopup="dialog" :aria-expanded="colorPopover?.field === 'strokeColor'" @click="openColorPopover('strokeColor', $event)">
                  <UiIcon name="circle" size="16" />
                  <span class="bubble-editor__color-indicator" :style="{ background: localStrokeColor }"></span>
                </UiIconButton>
              </div>

              <div
                v-if="localStrokeEnabled"
                class="bubble-editor__toolbar-stroke-width bubble-editor__toolbar-stroke-options"
                title="描边宽度"
              >
                <UiNumberField
                  v-model="localStrokeWidth"
                  :spin-step="1"
                  class="bubble-editor__number-field"
                  variant="editor"
                  :min="0"
                  :step="0.1"
                  aria-label="描边宽度"
                  @change="handleStrokeWidthChange"
                />
                <span class="bubble-editor__toolbar-unit">px</span>
              </div>
            </div>
          </UiField>
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
              class="bubble-editor__number-field"
              variant="editor"
              :min="0.1"
              :step="0.1"
              aria-label="行间距"
              title="行间距倍数（必须大于 0）"
              @change="handleLineSpacingChange"
            />
          </UiField>

          <UiField variant="editor" label="行内对齐">
            <div class="bubble-editor__toolbar-icon-group" aria-label="行内对齐" title="横排控制每行左右位置，竖排控制每列上下位置">
              <UiIconButton
                variant="soft"
                size="md"
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
                size="md"
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
                size="md"
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
          </UiField>

          <UiField variant="editor" label="整块对齐">
            <div class="bubble-editor__toolbar-icon-group" aria-label="文本块对齐" title="控制整个文本块在气泡内的位置">
              <UiIconButton
                variant="soft"
                size="md"
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
                size="md"
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
                size="md"
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
          </UiField>
        </div>

        <div class="bubble-editor__character-layout">
          <div class="bubble-editor__metrics-header"><span>字符调整</span><UiIconButton variant="plain" size="xs" label="重置字距和字形" title="恢复正常字距和文字宽高；保留字体、字号与颜色" @click="formatText({ tracking: 0, horizontal_scale: 1, vertical_scale: 1 })"><UiIcon name="rotate-ccw" size="14" /></UiIconButton></div>
          <div class="bubble-editor__metrics">
            <UiField v-for="metric in characterMetrics" :key="metric.field" class="bubble-editor__toolbar-field" variant="editor" :label="`${metric.label} %`" :control-id="`bubble-${metric.field}`">
              <UiNumberField :input-id="`bubble-${metric.field}`" :model-value="metric.value" :empty-value="metric.initial / metric.factor" nullable placeholder="混合" class="bubble-editor__number-field" variant="editor" :min="metric.min" :max="metric.max" :step="1" :spin-step="1" :aria-label="metric.label" :title="metric.title" @change="setCharacterMetric(metric.field, $event)" />
            </UiField>
          </div>

          <div class="bubble-editor__toolbar-row bubble-editor__toolbar-row--bottom">
            <UiField variant="editor" label="旋转">
              <div class="bubble-editor__toolbar-rotation-group" title="旋转角度">
                <UiIconButton variant="soft" size="md" class="bubble-editor__toolbar-action" label="逆时针旋转" title="逆时针旋转" @click="rotateLeft">
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
                <UiIconButton variant="soft" size="md" class="bubble-editor__toolbar-action" label="顺时针旋转" title="顺时针旋转" @click="rotateRight">
                  <UiIcon name="rotate-cw" size="16" />
                </UiIconButton>
                <UiIconButton variant="soft" size="xs" class="bubble-editor__toolbar-action" label="重置旋转" title="重置旋转" @click="resetRotation">
                  0
                </UiIconButton>
              </div>
            </UiField>

            <UiField variant="editor" label="位置微调">
              <div class="bubble-editor__toolbar-position-group" title="位置调整">
                <UiIconButton variant="soft" size="md" class="bubble-editor__toolbar-action" label="左移" title="左移" @click="moveLeft">
                  <UiIcon name="arrow-left" size="14" />
                </UiIconButton>
                <UiIconButton variant="soft" size="md" class="bubble-editor__toolbar-action" label="右移" title="右移" @click="moveRight">
                  <UiIcon name="arrow-right" size="14" />
                </UiIconButton>
                <UiIconButton variant="soft" size="md" class="bubble-editor__toolbar-action" label="上移" title="上移" @click="moveUp">
                  <UiIcon name="arrow-up" size="14" />
                </UiIconButton>
                <UiIconButton variant="soft" size="md" class="bubble-editor__toolbar-action" label="下移" title="下移" @click="moveDown">
                  <UiIcon name="arrow-down" size="14" />
                </UiIconButton>
                <span class="bubble-editor__toolbar-position-value">
                  <span>{{ positionX }}</span>,<span>{{ positionY }}</span>
                </span>
              </div>
            </UiField>
          </div>
        </div>
      </div>

      <div class="bubble-editor__repair-row">
        <UiField variant="editor" label="背景修复" control-id="bubbleInpaintMethod">
          <div class="bubble-editor__toolbar-inpaint-group">
            <UiSelect id="bubbleInpaintMethod" v-model="localInpaintMethod" :options="inpaintMethodOptions" aria-label="背景修复方式" @change="handleInpaintMethodChange" />
            <div v-if="localInpaintMethod === 'solid'" class="bubble-editor__toolbar-color-picker bubble-editor__toolbar-solid-color-options">
              <UiIconButton variant="soft" size="md" class="bubble-editor__toolbar-color-action" label="背景填充颜色" title="背景填充颜色" aria-haspopup="dialog" :aria-expanded="colorPopover?.field === 'fillColor'" @click="openColorPopover('fillColor', $event)">
                <UiIcon name="square" size="16" /><span class="bubble-editor__color-indicator" :style="{ background: localFillColor }"></span>
              </UiIconButton>
            </div>
          </div>
        </UiField>
      </div>

      <details class="bubble-editor__advanced">
        <summary class="bubble-editor__disclosure"><UiIcon name="chevron-right" size="14" class="bubble-editor__disclosure-chevron" />常用样式</summary>
        <div class="bubble-editor__favorites" title="收藏保存在当前浏览器；应用后的样式随页面保存">
          <div class="bubble-editor__favorite-row">
            <UiSelect v-model="favoriteSelected" :options="favorites.map(item => ({ label: item.name, value: item.name }))" aria-label="常用样式" placeholder="选择收藏的样式" />
            <UiButton size="sm" variant="secondary" :disabled="!favoriteSelected" @click="applyStyleFavorite">套用样式</UiButton>
            <UiIconButton size="md" variant="soft" label="删除样式收藏" :disabled="!favoriteSelected" @click="removeFavorite"><UiIcon name="trash" size="14" /></UiIconButton>
          </div>
          <div class="bubble-editor__favorite-row">
            <UiInput v-model="favoriteName" size="sm" maxlength="30" aria-label="样式名称" placeholder="如：对白、旁白、注释" @keydown.enter.prevent="saveStyleFavorite" />
            <UiButton size="sm" variant="secondary" :disabled="!canCopyStyle || !favoriteName.trim()" @click="saveStyleFavorite">收藏当前样式</UiButton>
          </div>
        </div>
      </details>
      <details class="bubble-editor__advanced">
        <summary class="bubble-editor__disclosure"><UiIcon name="chevron-right" size="14" class="bubble-editor__disclosure-chevron" />字号预设</summary>
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
        <UiButton variant="secondary" size="sm" :disabled="!canCopyStyle" :title="canCopyStyle ? '复制当前文字样式及排版，不复制内容和修复设置' : '文字包含多种样式，请先选中同一样式的文字'" @click="copyStyle">复制样式</UiButton>
        <UiButton variant="secondary" size="sm" aria-label="粘贴样式" :disabled="!copiedTextStyle" :title="`${textScope}：粘贴文字样式`" @click="pasteStyle">{{ pasteLabel }}</UiButton>
        <UiButton
          variant="secondary"
          size="sm"
          block
          class="bubble-editor__footer-action"
          :disabled="!canCopyStyle"
          title="将当前统一的文字样式应用到本页全部气泡；多种样式时请先选中文字"
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
  <EditColorPopover
    v-if="colorPopover"
    :key="colorPopover.field"
    :field="colorPopover.field"
    :anchor="colorPopover.anchor"
    :color="activeColorValue"
    @apply="applyColor"
    @pick="requestImageColor"
    @close="closeColorPopover"
  />
</template>

<script setup lang="ts">
import { ref } from 'vue'

import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import RichTextInput from './RichTextInput.vue'
import { nativeColor } from './richTextDocument'
import EditColorPopover from './EditColorPopover.vue'

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
const showOriginalText = ref(false)

const {
  favorites, favoriteName, favoriteSelected, removeFavorite, saveStyleFavorite, applyStyleFavorite, clearFormatting,
  copiedTextStyle,
  canCopyStyle,
  copyStyle,
  pasteStyle,
  editorFonts,
  textScope,
  pasteLabel,
  copyLabel,
  bold,
  italic,
  underline,
  characterMetrics,
  setCharacterMetric,
  handleRichSelection,
  handleRichChange,
  toggleTextStyle,
  formatText,
  FONT_SIZE_PRESETS,
  FONT_SIZE_MIN,
  FONT_SIZE_STEP,
  localOriginalText,
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
  colorPopover,
  activeColorValue,
  closeColorPopover,
  applyColor,
  requestImageColor,
  showJpKeyboard,
  jpKeyboardTarget,
  positionX,
  positionY,
  fontSelectGroups,
  inpaintMethodOptions,
  handleOriginalTextChange,
  copyOriginalText,
  copyTranslatedText,
  handleFontSizeChange,
  setFontSize,
  handleFontFamilyChange,
  setTextDirection,
  openColorPopover,
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
  applyToAll,
  handleOcrRecognize,
  handleReTranslate,
  toggleJpKeyboard,
  handleKanaInsert,
  handleKanaDelete,
} = useBubbleEditor(props, emit)

defineExpose({ focusText: () => translatedTextInput.value?.focus(), applyPickedColor(field: string, value: string): boolean {
  if (field !== 'textColor') return false
  formatText({ color: nativeColor(value) })
  return true
} })
</script>

<style scoped>
.bubble-editor {
  --ui-input-sm-min-height: 36px;
  --ui-input-sm-padding: 7px 10px;
  --ui-input-sm-font-size: 13px;
  --ui-selector-control-min-height: 36px;
  --ui-selector-control-padding: 0 10px;
  --ui-selector-control-font-size: 13px;
  --ui-field-editor-header-min-height: 24px;
  --ui-field-editor-label-font-size: 12px;
  --ui-number-field-width: 100%;
  --ui-number-field-input-width: 100%;
  --ui-number-field-control-width: 30px;
  --ui-number-field-control-height: 36px;
  --ui-icon-button-active-background: color-mix(in srgb, var(--color-action-brand) 12%, var(--color-surface-card));
  --ui-icon-button-active-border: var(--color-action-brand);
  --ui-icon-button-active-color: var(--color-text-brand);
  --ui-icon-button-active-hover-background: color-mix(in srgb, var(--color-action-brand) 18%, var(--color-surface-card));
  --ui-icon-button-active-hover-border: var(--color-action-brand);

  display: flex;
  flex: 1;
  flex-direction: column;
  gap: 16px;
  min-width: 0;
  min-height: 0;
  margin: 0;
  padding: 12px 14px 16px;
  overflow: auto;
  container: bubble-editor / inline-size;
  border: 0;
  background: var(--color-surface-card);
}

.bubble-editor:disabled { cursor: wait; }
.bubble-editor:disabled .bubble-editor__text-scope { color: var(--color-text-disabled); }
.bubble-editor--empty { align-items: center; justify-content: center; color: var(--color-text-muted); font-size: 13px; }
.bubble-editor__text-panel { display: flex; flex-direction: column; gap: 8px; width: 100%; }
.bubble-editor__text-panel-header { display: flex; align-items: center; justify-content: space-between; min-height: 28px; }

.bubble-editor__text-panel-title,
.bubble-editor__section-title { margin: 0; color: var(--color-text-strong); font-size: 13px; font-weight: 600; }
.bubble-editor__original-toggle { display: inline-flex; align-items: center; gap: 5px; padding: 4px 0; color: var(--color-text-danger-strong); }
.bubble-editor__refresh-action--loading .bubble-editor__refresh-icon { animation: spin-icon 1s linear infinite; }
.bubble-editor__textarea { width: 100%; font-family: var(--font-jp); }
.bubble-editor__text-actions { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: 6px; margin-top: 6px; }
.bubble-editor__text-actions--format { justify-content: space-between; margin-top: 0; padding: 4px 0 12px; border-bottom: 1px solid var(--color-border-muted); }
.bubble-editor__format-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-left: auto; }
.bubble-editor__text-scope { color: var(--color-text-secondary); font-size: 11px; line-height: 24px; white-space: nowrap; }
.bubble-editor__text-action { display: inline-flex; align-items: center; gap: 5px; padding: 5px 8px; font-size: 12px; }

.bubble-editor__style-section { display: flex; flex: 0 0 auto; flex-direction: column; gap: 12px; width: 100%; }
.bubble-editor__toolbar { display: grid; gap: 16px; min-width: 0; }
.bubble-editor__toolbar-row { display: grid; align-items: start; gap: 12px; min-width: 0; }
.bubble-editor__toolbar-row--top { grid-template-columns: minmax(0, 1fr) minmax(180px, 0.6fr); }

.bubble-editor__toolbar-row--actions,
.bubble-editor__toolbar-row--typography { padding-top: 12px; border-top: 1px solid var(--color-border-muted); }
.bubble-editor__toolbar-row--actions { grid-template-columns: minmax(76px, 1fr) minmax(52px, 0.7fr) minmax(152px, 1.6fr); }
.bubble-editor__toolbar-row--typography { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.bubble-editor__toolbar-field { min-width: 0; }

.bubble-editor__toolbar-icon-group,
.bubble-editor__toolbar-stroke-cluster,
.bubble-editor__toolbar-inpaint-group,
.bubble-editor__toolbar-stroke-width,
.bubble-editor__toolbar-rotation-group,
.bubble-editor__toolbar-position-group { display: flex; align-items: center; gap: 4px; min-width: 0; }
.bubble-editor__toolbar-color-picker { display: inline-flex; }
.bubble-editor__toolbar-color-action { flex-direction: column; gap: 2px; }
.bubble-editor__color-indicator { width: 18px; height: 4px; border: 1px solid var(--color-border-default); border-radius: 3px; }
.bubble-editor__toolbar-stroke-width { flex: 1; }
.bubble-editor__number-field--rotation { max-width: 96px; }
.bubble-editor__toolbar-unit { color: var(--color-text-secondary); font-size: 11px; }
.bubble-editor__toolbar-position-value { margin-left: auto; padding-left: 4px; color: var(--color-text-muted); font-size: 11px; font-variant-numeric: tabular-nums; white-space: nowrap; }
.bubble-editor__repair-row { padding-top: 16px; border-top: 1px solid var(--color-border-muted); }
.bubble-editor__toolbar-inpaint-group { gap: 8px; }

.bubble-editor__advanced { min-width: 0; border-top: 1px solid var(--color-border-muted); }
.bubble-editor__disclosure { display: flex; align-items: center; gap: 7px; min-height: 36px; padding: 6px 0; border-radius: 6px; color: var(--color-text-strong); font-size: 12px; font-weight: 600; list-style: none; cursor: pointer; user-select: none; }
.bubble-editor__disclosure::-webkit-details-marker { display: none; }
.bubble-editor__disclosure:hover { background: var(--color-surface-interactive-hover); }
.bubble-editor__disclosure:focus-visible { outline: 2px solid var(--color-border-brand); outline-offset: -2px; }
.bubble-editor__disclosure-chevron { color: var(--color-text-muted); transition: rotate 0.15s; }
.bubble-editor__advanced[open] > .bubble-editor__disclosure .bubble-editor__disclosure-chevron { rotate: 90deg; }
.bubble-editor__character-layout { display: grid; gap: 12px; min-width: 0; padding-top: 12px; border-top: 1px solid var(--color-border-muted); }
.bubble-editor__metrics-header { display: flex; align-items: center; justify-content: space-between; min-height: 24px; color: var(--color-text-strong); font-size: 12px; font-weight: 600; }
.bubble-editor__metrics { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.bubble-editor__toolbar-row--bottom { grid-template-columns: repeat(auto-fit, minmax(min(100%, 228px), 1fr)); padding-top: 12px; border-top: 1px solid var(--color-border-muted); }
.bubble-editor__favorites { display: grid; gap: 10px; padding-top: 8px; }
.bubble-editor__favorite-row { display: flex; align-items: center; gap: 6px; min-width: 0; }
.bubble-editor__favorite-row > :first-child { flex: 1; min-width: 0; }
.bubble-editor__favorite-row > :not(:first-child) { flex-shrink: 0; }
.bubble-editor__font-size-presets { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 6px; padding-top: 8px; }
.bubble-editor__font-size-preset { padding: 8px 6px; border: 1px solid var(--color-border-muted); border-radius: 6px; background: var(--color-surface-input); color: var(--color-text-secondary); font-size: 12px; font-variant-numeric: tabular-nums; }
.bubble-editor__font-size-preset:hover { border-color: var(--color-border-brand); color: var(--color-text-brand); }
.bubble-editor__font-size-preset--active { border-color: var(--color-border-brand); background: var(--ui-icon-button-active-background); color: var(--color-text-brand); }
.bubble-editor__footer-actions { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; padding-top: 12px; border-top: 1px solid var(--color-border-muted); }
.bubble-editor__footer-action { grid-column: 1 / -1; }

@container bubble-editor (max-width: 360px) {
  .bubble-editor__toolbar-row--top { grid-template-columns: minmax(0, 1fr); }

  .bubble-editor__toolbar-row--actions,
  .bubble-editor__toolbar-row--typography { grid-template-columns: repeat(2, minmax(0, 1fr)); }

  .bubble-editor__stroke-field,
  .bubble-editor__toolbar-field--line-spacing { grid-column: 1 / -1; }
}
</style>
