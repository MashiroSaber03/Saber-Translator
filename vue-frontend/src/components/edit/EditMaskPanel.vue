<script setup lang="ts">
import { computed } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiColorInput from '@/components/ui/UiColorInput.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import type { UiIconName } from '@/components/ui/iconRegistry'
import { usePublicUserAccess } from '@/composables/usePublicUserAccess'
import { inpaintMethodOptions } from '@/utils/textStyleForm'
import type { InpaintMethod } from '@/types/bubble'
import type { MaskTool } from '@/composables/edit/useMaskEditor'
import type { PageMasks } from '@/api/v2/masks'
import { maskLayerOptions, type MaskLayer, type MaskVisibility } from '@/utils/maskPreview'

const props = defineProps<{
  tool: MaskTool | null
  visibility: MaskVisibility
  preview: PageMasks | null
  busy: boolean
  loading: boolean
  saving: boolean
  applying: boolean
  dirty: boolean
  error: string
  inpaintMethod: InpaintMethod
  fillColor: string
  brushSize: number
}>()
const emit = defineEmits<{
  tool: [value: MaskTool | null]
  visibility: [value: MaskVisibility]
  apply: []
  close: []
  retry: []
  discard: []
  method: [value: InpaintMethod]
  fillColor: [value: string]
}>()
const access = usePublicUserAccess()
const repairMethods = computed(() => access.modelOptions(inpaintMethodOptions, {
  lama_mpe: 'lama_mpe', litelama: 'litelama', lama_manga: 'lama_manga',
}))
const tools: { value: MaskTool; label: string; icon: UiIconName; hint: string }[] = [
  { value: 'repair', label: '增加修复区域 (R)', icon: 'brush', hint: '修复区域笔刷：按住 R，或点击启用；再次点击退出' },
  { value: 'preserve', label: '保留原图区域 (U)', icon: 'eraser', hint: '保留原图笔刷：按住 U，或点击启用；再次点击退出' },
  { value: 'clear', label: '清除手动标记', icon: 'rotate-ccw', hint: '清除手动标记，恢复跟随自动；再次点击退出' },
]
const anyVisible = computed(() => maskLayerOptions.some(item => props.visibility[item.value] &&
  (item.value !== 'raw' || props.preview?.preciseMask)))
const processingHint = computed(() => {
  if (!props.preview) return ''
  if (!props.preview.preciseMask) return '没有精确文字范围，自动区域使用文本框'
  const settings = props.preview.processing
  return `框扩大 ${settings.boxExpandRatio}% · 膨胀 ${settings.dilateSize}px${settings.closeGaps ? ' · 闭合空隙' : ''}`
})
function toggleLayer(layer: MaskLayer): void {
  emit('visibility', { ...props.visibility, [layer]: !props.visibility[layer] })
}
function layerDescription(layer: MaskLayer): string {
  const processing = props.preview?.processing
  switch (layer) {
    case 'raw': return props.preview?.preciseMask ? '检测器输出的完整文字范围，灰色斜纹' : '当前页没有精确文字范围'
    case 'boxes': return `黄色虚线和淡黄色底色：文本框范围，扩大 ${processing?.boxExpandRatio ?? 0}%`
    case 'automatic': return props.preview?.preciseMask
      ? `裁切后膨胀 ${processing?.dilateSize ?? 0}px${processing?.closeGaps ? '，并闭合细小空隙' : ''}；膨胀可略超出裁切边界`
      : '使用文本框区域，不进行膨胀'
    case 'repair': return '绿色：强制修复，可画在文本框外'
    case 'preserve': return '蓝色斜纹：强制保留，覆盖自动区域；手动标记后画覆盖前画'
    case 'final': return '红色半透明：最终实际修复的范围'
  }
}
</script>

<template>
  <div class="edit-mask-panel" aria-label="修复范围编辑">
    <div class="edit-mask-panel__toolbar">
      <div class="edit-mask-panel__group" aria-label="编辑模式">
        <UiButton
          variant="toolbar" class="edit-mask-panel__control" aria-label="返回文本框编辑"
          title="返回文本框编辑" :disabled="busy" @click="$emit('close')"
        >
          <UiIcon name="square" size="14" />文本框
        </UiButton>
        <span class="edit-mask-panel__title"><UiIcon name="target" size="14" />修复范围</span>
      </div>
      <div role="group" aria-label="修复范围画笔" class="edit-mask-panel__group">
        <UiIconButton
          v-for="item in tools" :key="item.value" variant="inverse" size="sm"
          class="edit-mask-panel__control edit-mask-panel__brush" :class="`edit-mask-panel__brush--${item.value}`"
          :label="item.label" :title="item.hint" :pressed="tool === item.value"
          :disabled="busy" @click="$emit('tool', item.value)"
        >
          <UiIcon :name="item.icon" size="16" />
        </UiIconButton>
      </div>
      <div
        role="group" aria-label="查看修复范围图层" class="edit-mask-panel__layers"
        title="原始文字按文本框裁切，再处理自动区域，最后叠加手动标记。开关只控制显示，不改变修复结果。"
      >
        <span
          v-for="item in maskLayerOptions" :key="item.value" class="edit-mask-panel__layer"
          :style="{ '--mask-layer-color': `rgb(${item.color.join(' ')})` }"
        >
          <span v-if="item.operator" class="edit-mask-panel__operator" aria-hidden="true">{{ item.operator }}</span>
          <UiButton
            variant="toolbar" class="edit-mask-panel__control edit-mask-panel__layer-toggle"
            :aria-label="`显示${item.label}`" :title="`${item.text}：${layerDescription(item.value)}。点击切换显示。`"
            :aria-pressed="visibility[item.value] && (item.value !== 'raw' || !!preview?.preciseMask)"
            :disabled="item.value === 'raw' && !preview?.preciseMask" @click="toggleLayer(item.value)"
          >
            <i class="edit-mask-panel__swatch" :class="`edit-mask-panel__swatch--${item.value}`" aria-hidden="true"></i>
            {{ item.text }}
          </UiButton>
        </span>
      </div>
      <div class="edit-mask-panel__method">
        <div class="edit-mask-panel__select">
          <UiSelect
            aria-label="修复方式" teleport-to=".edit-mask-panel" size="sm"
            :options="repairMethods" :model-value="inpaintMethod" :disabled="busy"
            @update:model-value="$emit('method', $event as InpaintMethod)"
          />
        </div>
        <UiColorInput
          v-if="inpaintMethod === 'solid'" aria-label="填充颜色"
          :model-value="fillColor" :disabled="busy" class="edit-mask-panel__color"
          @update:model-value="$emit('fillColor', $event)"
        />
      </div>
      <UiButton
        variant="toolbar" class="edit-mask-panel__control edit-mask-panel__apply"
        aria-label="按当前范围修复" title="按当前范围修复 (Ctrl+Enter)"
        :disabled="busy || dirty || !!error" @click="$emit('apply')"
      >
        <UiIcon :name="applying ? 'loading' : 'wand-sparkles'" size="14" />
        {{ applying ? '正在修复…' : '按范围修复' }}
      </UiButton>
    </div>
    <div class="edit-mask-panel__status" role="status">
      <span v-if="error" class="edit-mask-panel__error"><UiIcon name="alert-triangle" size="12" />{{ error }}</span>
      <template v-else>
        <span class="edit-mask-panel__state"><UiIcon :name="saving || loading || applying ? 'loading' : 'check'" size="12" />{{ saving ? '正在保存…' : applying ? '正在修复…' : loading ? '正在更新预览…' : '预览已就绪' }}</span>
        <span>{{ tool ? `${brushSize}px · 滚轮调笔刷 · Esc / 再次点击退出 · 松开自动保存` : '滚轮缩放 · 拖动移动 · 按住 R / U 使用笔刷' }}</span>
        <span class="edit-mask-panel__processing">{{ processingHint }} · {{ anyVisible ? '图层开关仅控制显示' : '覆盖层已隐藏，修复范围不变' }}</span>
      </template>
      <UiButton v-if="error" variant="toolbar" class="edit-mask-panel__control" :disabled="busy" @click="$emit('retry')">
        {{ dirty ? '重试保存' : '重新加载预览' }}
      </UiButton>
      <UiButton v-if="error && dirty" variant="toolbar" class="edit-mask-panel__control" :disabled="busy" @click="$emit('discard')">
        放弃未保存修改
      </UiButton>
    </div>
  </div>
</template>

<style scoped>
.edit-mask-panel {
  --edit-mask-panel-muted: color-mix(in srgb, var(--color-text-inverse) 70%, transparent);
  --edit-mask-panel-accent: var(--color-action-brand);
  --ui-select-background: var(--color-overlay-scrim-subtle);
  --ui-select-color: var(--color-text-inverse);
  --ui-select-border: 1px solid var(--color-overlay-inverse-muted);
  --ui-selector-arrow-text: var(--edit-mask-panel-muted);
  --ui-selector-control-text: var(--color-text-inverse);
  --ui-selector-dropdown-background: var(--color-surface-inverse-panel);
  --ui-selector-dropdown-border: var(--color-overlay-inverse-muted);
  --ui-selector-option-hover-background: var(--color-overlay-inverse-subtle);
  --ui-selector-option-selected-background: color-mix(in srgb, var(--color-action-brand) 30%, transparent);
  --ui-selector-option-selected-text: var(--color-text-inverse);

  flex-shrink: 0;
  padding: 6px 12px;
  background: var(--color-overlay-scrim-subtle);
  color: var(--color-text-inverse);
  font-size: 12px;
  color-scheme: dark;
}

.edit-mask-panel__toolbar,
.edit-mask-panel__group,
.edit-mask-panel__title,
.edit-mask-panel__layers,
.edit-mask-panel__layer,
.edit-mask-panel__method,
.edit-mask-panel__status,
.edit-mask-panel__state {
  display: flex;
  align-items: center;
  gap: 6px;
}

.edit-mask-panel__toolbar { flex-wrap: wrap; gap: 6px 12px; }
.edit-mask-panel__group, .edit-mask-panel__method { flex-shrink: 0; }
.edit-mask-panel__title { padding: 5px 6px; color: var(--color-text-inverse); font-weight: 600; }
.edit-mask-panel__title svg { color: var(--color-action-brand); }
.edit-mask-panel__layers { flex-wrap: wrap; gap: 4px; }
.edit-mask-panel__layer { gap: 4px; }
.edit-mask-panel__operator { color: var(--edit-mask-panel-muted); font-size: 12px; }

.edit-mask-panel__control {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  min-height: 30px;
  padding: 4px 8px;
  border: 1px solid var(--color-overlay-inverse-muted);
  border-radius: 6px;
  background: var(--color-overlay-inverse-subtle);
  color: var(--color-text-inverse);
  font-size: 12px;
  white-space: nowrap;
}

.edit-mask-panel__control:hover:not(:disabled) { background: var(--color-overlay-inverse-soft); }
.edit-mask-panel__control:disabled { opacity: 0.45; }
.edit-mask-panel__brush { padding: 0; }
.edit-mask-panel__brush--repair { --edit-mask-panel-accent: var(--color-status-success); }
.edit-mask-panel__brush--preserve { --edit-mask-panel-accent: var(--color-status-info-bright); }

.edit-mask-panel__brush[aria-pressed='true'] {
  background: color-mix(in srgb, var(--edit-mask-panel-accent) 22%, transparent);
  border-color: var(--edit-mask-panel-accent);
  color: color-mix(in srgb, var(--edit-mask-panel-accent) 60%, var(--color-text-inverse));
}

.edit-mask-panel__layer-toggle { background: transparent; }
.edit-mask-panel__layer-toggle[aria-pressed='true'] { background: color-mix(in srgb, var(--mask-layer-color) 18%, transparent); border-color: color-mix(in srgb, var(--mask-layer-color) 65%, transparent); }
.edit-mask-panel__swatch { width: 8px; height: 8px; border-radius: 2px; background: var(--mask-layer-color); flex-shrink: 0; }
.edit-mask-panel__swatch--raw, .edit-mask-panel__swatch--preserve { background: repeating-linear-gradient(135deg, var(--mask-layer-color) 0 2px, transparent 2px 4px); border: 1px solid var(--mask-layer-color); }
.edit-mask-panel__swatch--boxes { background: transparent; border: 1px dashed var(--mask-layer-color); }
.edit-mask-panel__swatch--final { background: transparent; border: 1px solid var(--mask-layer-color); }
.edit-mask-panel__swatch--final { border-radius: 50%; }
.edit-mask-panel__select { width: 140px; }
.edit-mask-panel__color { --ui-colorpicker-width: 30px; --ui-colorpicker-height: 30px; --ui-colorpicker-background: var(--color-overlay-scrim-subtle); --ui-colorpicker-border: 1px solid var(--color-overlay-inverse-muted); }
.edit-mask-panel__apply { margin-left: auto; background: var(--color-action-brand); border-color: transparent; }
.edit-mask-panel__apply:hover:not(:disabled) { background: var(--color-action-brand-strong); }
.edit-mask-panel__status { flex-wrap: wrap; gap: 4px 12px; margin-top: 4px; min-height: 16px; color: var(--edit-mask-panel-muted); font-size: 11px; }
.edit-mask-panel__state { flex-shrink: 0; color: var(--color-status-success); }
.edit-mask-panel__error { display: inline-flex; align-items: center; gap: 6px; color: var(--color-status-error-bright); }
.edit-mask-panel__processing { margin-left: auto; }
</style>
