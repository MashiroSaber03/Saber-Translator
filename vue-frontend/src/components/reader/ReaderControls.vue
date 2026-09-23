<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiField from '@/components/ui/UiField.vue'
import UiColorSwatchGroup from '@/components/ui/UiColorSwatchGroup.vue'
import {
  READER_LAYOUTS,
  READER_FITS,
  READER_BG_COLOR_PRESETS,
  type ReaderSettings,
  type ReaderLayout,
  type ReaderFit,
} from './readerSettings'

const props = defineProps<{
  settings: ReaderSettings
  page: number
  total: number
  range: string
  offset: boolean
  canPrev: boolean
  canNext: boolean
  hasPrevChapter: boolean
  hasNextChapter: boolean
  chapterId: string
  chapters: { id: string; title: string }[]
}>()
const emit = defineEmits<{
  settingsChange: [settings: ReaderSettings]
  navigate: [delta: number]
  jump: [page: number]
  chapter: [id: string]
  offset: []
  restart: []
  close: []
}>()
const pageInput = ref<number | null>(props.page)
watch(
  () => props.page,
  value => {
    pageInput.value = value
  }
)
const chapterOptions = computed(() => props.chapters.map(c => ({ value: c.id, label: c.title })))
const chapterIndex = computed(() => props.chapters.findIndex(c => c.id === props.chapterId))
const layoutLabel = computed(
  () => READER_LAYOUTS.find(item => item.value === props.settings.layout)!.label
)
const fitLabel = computed(
  () => READER_FITS.find(item => item.value === props.settings.fits[props.settings.layout])!.label
)
function update(patch: Partial<ReaderSettings>) {
  emit('settingsChange', { ...props.settings, ...patch })
}
function setLayout(layout: ReaderLayout) {
  update({ layout })
}
function setFit(fit: ReaderFit) {
  update({ fits: { ...props.settings.fits, [props.settings.layout]: fit } })
}
function cycleLayout() {
  setLayout(
    READER_LAYOUTS[
      (READER_LAYOUTS.findIndex(x => x.value === props.settings.layout) + 1) % READER_LAYOUTS.length
    ]!.value
  )
}
function cycleFit() {
  setFit(
    READER_FITS[
      (READER_FITS.findIndex(x => x.value === props.settings.fits[props.settings.layout]) + 1) %
        READER_FITS.length
    ]!.value
  )
}
function chapter(delta: number) {
  const item = props.chapters[chapterIndex.value + delta]
  if (item) emit('chapter', item.id)
}
function jump() {
  if (pageInput.value !== null && Number.isFinite(pageInput.value)) emit('jump', pageInput.value)
}
</script>

<template>
  <aside class="reader-controls" aria-label="阅读设置">
    <div class="reader-controls__heading">
      <strong>阅读设置</strong><UiButton variant="inverse" size="sm" aria-label="收起阅读设置" @click="emit('close')">
        关闭
      </UiButton>
    </div>
    <section class="reader-controls__section">
      <UiField class="reader-controls__field" label="页面" tone="inverse">
        <form class="reader-controls__navigation" @submit.prevent="jump">
          <UiButton
            variant="inverse"
            size="sm"
            :disabled="!canPrev"
            aria-label="上一页"
            @click="emit('navigate', -1)"
          >
            ‹
          </UiButton>
          <UiNumberField
            v-model="pageInput"
            class="reader-controls__page-input"
            aria-label="跳转页码"
            :min="1"
            :max="Math.max(1, total)"
            :disabled="!total"
          />
          <UiButton type="submit" variant="inverse" size="sm" :disabled="!total">跳转</UiButton>
          <UiButton
            variant="inverse"
            size="sm"
            :disabled="!canNext"
            aria-label="下一页"
            @click="emit('navigate', 1)"
          >
            ›
          </UiButton>
        </form>
        <span class="reader-controls__hint">{{ range }} / {{ total }} 页</span>
      </UiField>
      <UiField class="reader-controls__field" label="章节" tone="inverse">
        <UiSelect
          :model-value="chapterId"
          :options="chapterOptions"
          teleport-to=".reader-page"
          aria-label="选择章节"
          @update:model-value="emit('chapter', String($event))"
        />
        <div class="reader-controls__pair">
          <UiButton variant="inverse" size="sm" :disabled="!hasPrevChapter" @click="chapter(-1)">
            上一章
          </UiButton>
          <UiButton variant="inverse" size="sm" :disabled="!hasNextChapter" @click="chapter(1)">
            下一章
          </UiButton>
        </div>
      </UiField>
    </section>
    <section class="reader-controls__section">
      <UiField class="reader-controls__field" label="阅读模式" tone="inverse">
        <UiButton variant="inverse" block aria-label="循环切换阅读模式" @click="cycleLayout">
          {{ layoutLabel }} · 切换
        </UiButton>
        <details>
          <summary>选择模式</summary>
          <div class="reader-controls__options">
            <UiButton
              v-for="item in READER_LAYOUTS"
              :key="item.value"
              :variant="settings.layout === item.value ? 'primary' : 'inverse'"
              :aria-pressed="settings.layout === item.value"
              size="sm"
              @click="setLayout(item.value)"
            >
              {{ item.label }}
            </UiButton>
          </div>
        </details>
      </UiField>
      <UiButton
        v-if="settings.layout === 'double'"
        :variant="offset ? 'primary' : 'inverse'"
        :aria-pressed="offset"
        block
        @click="emit('offset')"
      >
        双页错开一页
      </UiButton>
      <UiField class="reader-controls__field" label="图片适配" tone="inverse">
        <UiButton variant="inverse" block aria-label="循环切换图片适配" @click="cycleFit">
          {{ fitLabel }} · 切换
        </UiButton>
        <details>
          <summary>选择适配方式</summary>
          <div class="reader-controls__options">
            <UiButton
              v-for="item in READER_FITS"
              :key="item.value"
              :variant="settings.fits[settings.layout] === item.value ? 'primary' : 'inverse'"
              :aria-pressed="settings.fits[settings.layout] === item.value"
              size="sm"
              @click="setFit(item.value)"
            >
              {{ item.label }}
            </UiButton>
          </div>
        </details>
      </UiField>
      <UiField class="reader-controls__field" label="阅读方向" tone="inverse">
        <div class="reader-controls__pair">
          <UiButton
            :variant="settings.direction === 'ltr' ? 'primary' : 'inverse'"
            size="sm"
            :aria-pressed="settings.direction === 'ltr'"
            @click="update({ direction: 'ltr' })"
          >
            从左向右
          </UiButton>
          <UiButton
            :variant="settings.direction === 'rtl' ? 'primary' : 'inverse'"
            size="sm"
            :aria-pressed="settings.direction === 'rtl'"
            @click="update({ direction: 'rtl' })"
          >
            从右向左
          </UiButton>
        </div>
      </UiField>
      <UiField class="reader-controls__field" label="底部进度条" tone="inverse">
        <div class="reader-controls__options reader-controls__options--three">
          <UiButton
            v-for="item in [
              { value: 'normal', label: '普通' },
              { value: 'pages', label: '页码分段' },
              { value: 'hidden', label: '隐藏' },
            ] as const"
            :key="item.value"
            :variant="settings.progress === item.value ? 'primary' : 'inverse'"
            size="sm"
            :aria-pressed="settings.progress === item.value"
            @click="update({ progress: item.value })"
          >
            {{ item.label }}
          </UiButton>
        </div>
      </UiField>
    </section>
    <details class="reader-controls__section">
      <summary>外观与快捷键</summary>
      <UiField label="背景颜色" tone="inverse">
        <UiColorSwatchGroup
          :model-value="settings.bgColor"
          :options="READER_BG_COLOR_PRESETS"
          aria-label="阅读背景颜色"
          @change="update({ bgColor: $event })"
        />
      </UiField>
      <template v-if="settings.layout === 'vertical' || settings.layout === 'horizontal'">
        <UiField
          v-if="settings.fits[settings.layout] === 'width' || settings.fits[settings.layout] === 'screen'"
          :label="`图片宽度上限 ${settings.imageWidth}%`"
          tone="inverse"
        >
          <UiInput
            type="range"
            aria-label="图片宽度上限"
            :min="50"
            :max="100"
            :model-value="settings.imageWidth"
            @update:model-value="update({ imageWidth: Number($event) })"
          />
        </UiField>
        <UiField :label="`图片间距 ${settings.imageGap}px`" tone="inverse">
          <UiInput
            type="range"
            aria-label="图片间距"
            :min="0"
            :max="50"
            :model-value="settings.imageGap"
            @update:model-value="update({ imageGap: Number($event) })"
          />
        </UiField>
      </template>
      <p class="reader-controls__hint">
        ← → 翻页 · Home / End 首尾页<br />M 菜单 · F 全屏 · I 适配 · O 双页偏移<br />滚轮只滚动图片，不自动跳章。
      </p>
    </details>
    <UiButton variant="inverse" block :disabled="!total" @click="emit('restart')">
      从头阅读
    </UiButton>
  </aside>
</template>
<style scoped>
.reader-controls {
  width: 288px;
  flex: 0 0 288px;
  min-height: 0;
  overflow-y: auto;
  padding: 20px 16px 48px;
  background: var(--color-surface-inverse-raised);
  color: var(--color-text-inverse);
  border-left: 1px solid var(--color-overlay-inverse-subtle);
}

.reader-controls__heading {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.reader-controls__section {
  display: grid;
  gap: 16px;
  border-bottom: 1px solid var(--color-overlay-inverse-subtle);
  padding-bottom: 20px;
  margin-bottom: 20px;
}

.reader-controls__navigation {
  display: flex;
  align-items: center;
  gap: 4px;
}

.reader-controls__page-input {
  min-width: 0;
  flex: 1;
}

.reader-controls__field {
  margin-bottom: 0;
}

.reader-controls__pair,
.reader-controls__options {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px;
  margin-top: 8px;
}

.reader-controls__options--three {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.reader-controls__hint {
  display: block;
  font-size: 12px;
  color: var(--color-text-inverse);
  opacity: 0.7;
  line-height: 1.8;
  margin-top: 8px;
}

.reader-controls summary {
  cursor: pointer;
  font-size: 12px;
  padding: 10px 0;
  opacity: 0.85;
}

.reader-controls__options {
  --ui-button-padding: 8px 4px;
  --ui-button-font-size: 12px;
}

@media (--breakpoint-md-down) {
  .reader-controls {
    width: 100%;
    flex-basis: 100%;
  }
}
</style>
