<script setup lang="ts">
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiColorSwatchGroup from '@/components/ui/UiColorSwatchGroup.vue'
import UiField from '@/components/ui/UiField.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import OverlayLayer from '@/components/ui/OverlayLayer.vue'
import { nextTick, ref, onMounted, onUnmounted, watch } from 'vue'
import { useDialogLifecycle } from '@/composables/useDialogLifecycle'
import {
  DEFAULT_READER_SETTINGS,
  READER_BG_COLOR_PRESETS,
  loadReaderSettings,
  saveReaderSettings,
  type ReaderSettings,
} from './readerSettings'

const props = defineProps<{
  hasPrevChapter: boolean
  hasNextChapter: boolean
  showChapterNav: boolean
  settingsRequestId?: number
}>()

const emit = defineEmits<{
  (e: 'navigateChapter', direction: 'prev' | 'next'): void
  (e: 'settingsChange', settings: ReaderSettings): void
}>()

const settings = ref<ReaderSettings>({ ...DEFAULT_READER_SETTINGS })
const isSettingsPanelOpen = ref(false)
const settingsDialogRef = ref<HTMLElement | null>(null)
const showScrollTopBtn = ref(false)
const bgColorPresets = READER_BG_COLOR_PRESETS
let scrollContainer: HTMLElement | null = null

function bindScrollContainer() {
  scrollContainer?.removeEventListener('scroll', handleScroll)
  scrollContainer = document.querySelector<HTMLElement>('.reader-canvas__stream')
  scrollContainer?.addEventListener('scroll', handleScroll, { passive: true })
  handleScroll()
}

function openSettings() {
  isSettingsPanelOpen.value = true
}

function closeSettings() {
  isSettingsPanelOpen.value = false
}

function scrollToTop() {
  scrollContainer?.scrollTo({ top: 0, behavior: 'smooth' })
}

function handleScroll() {
  showScrollTopBtn.value = (scrollContainer?.scrollTop ?? 0) > 500
}

function handleKeydown(e: KeyboardEvent) {
  if (isSettingsPanelOpen.value) return
  switch (e.key) {
    case 'ArrowLeft':
      if (props.hasPrevChapter) {
        emit('navigateChapter', 'prev')
      }
      break
    case 'ArrowRight':
      if (props.hasNextChapter) {
        emit('navigateChapter', 'next')
      }
      break
    case 'Home':
      scrollContainer?.scrollTo({ top: 0, behavior: 'smooth' })
      break
    case 'End':
      scrollContainer?.scrollTo({
        top: scrollContainer.scrollHeight,
        behavior: 'smooth',
      })
      break
  }
}

useDialogLifecycle({
  open: isSettingsPanelOpen,
  container: settingsDialogRef,
  close: closeSettings,
})

function loadSettings() {
  const storedSettings = loadReaderSettings()
  if (storedSettings) {
    settings.value = storedSettings
  }
  publishSettings()
}

function saveSettings() {
  saveReaderSettings(settings.value)
}

function publishSettings() {
  emit('settingsChange', { ...settings.value })
}

function updateImageWidth(value: number) {
  settings.value.imageWidth = value
  publishSettings()
  saveSettings()
}

function updateImageGap(value: number) {
  settings.value.imageGap = value
  publishSettings()
  saveSettings()
}

function updateBgColor(color: string) {
  settings.value.bgColor = color
  publishSettings()
  saveSettings()
}

function navigateChapter(direction: 'prev' | 'next') {
  emit('navigateChapter', direction)
}

watch(
  () => props.settingsRequestId,
  (requestId, previousRequestId) => {
    if (requestId !== undefined && requestId !== previousRequestId) {
      openSettings()
    }
  }
)
watch(
  () => props.showChapterNav,
  () => {
    void nextTick(bindScrollContainer)
  }
)

onMounted(() => {
  loadSettings()

  void nextTick(bindScrollContainer)
  document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  scrollContainer?.removeEventListener('scroll', handleScroll)
  scrollContainer = null
  document.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
  <OverlayLayer v-if="showChapterNav" class="reader-controls__chapter-nav-layer" passthrough>
    <nav class="reader-controls__chapter-nav" aria-label="章节导航">
      <UiButton
        variant="inverse"
        size="md"
        class="reader-controls__nav-button"
        :disabled="!hasPrevChapter"
        @click="navigateChapter('prev')"
      >
        <span class="reader-controls__nav-icon" aria-hidden="true">◀</span>
        <span class="reader-controls__nav-text">上一章</span>
      </UiButton>
      <UiButton
        variant="inverse"
        size="md"
        class="reader-controls__nav-button"
        :disabled="!hasNextChapter"
        @click="navigateChapter('next')"
      >
        <span class="reader-controls__nav-text">下一章</span>
        <span class="reader-controls__nav-icon" aria-hidden="true">▶</span>
      </UiButton>
    </nav>
  </OverlayLayer>

  <OverlayLayer v-show="showScrollTopBtn" class="reader-controls__scroll-top-layer" passthrough>
    <UiIconButton
      variant="primary"
      size="xl"
      shape="circle"
      elevated
      class="reader-controls__scroll-top-button"
      label="回到顶部"
      @click="scrollToTop"
    >
      <span aria-hidden="true">↑</span>
    </UiIconButton>
  </OverlayLayer>

  <OverlayLayer
    v-if="isSettingsPanelOpen"
    class="reader-controls__settings-panel"
    level="popover"
    role="dialog"
    aria-modal="true"
    aria-label="阅读设置"
  >
    <div class="reader-controls__settings-overlay" @click="closeSettings"></div>
    <div ref="settingsDialogRef" class="reader-controls__settings-content" tabindex="-1">
      <div class="reader-controls__settings-header">
        <h3 class="reader-controls__settings-title">阅读设置</h3>
        <UiIconButton
          variant="inverse"
          size="sm"
          shape="circle"
          class="reader-controls__close-button"
          label="关闭阅读设置"
          @click="closeSettings"
        >
          <UiIcon name="x" size="16" />
        </UiIconButton>
      </div>
      <div class="reader-controls__settings-body">
        <UiField
          variant="settings"
          tone="inverse"
          label="图片宽度"
          control-id="imageWidthSlider"
          class="reader-controls__setting-field"
        >
          <div class="reader-controls__setting-control">
            <UiInput
              type="range"
              id="imageWidthSlider"
              class="reader-controls__range"
              min="50"
              max="100"
              :model-value="settings.imageWidth"
              @update:model-value="value => updateImageWidth(Number(value))"
            />
            <span class="reader-controls__setting-value">{{ settings.imageWidth }}%</span>
          </div>
        </UiField>

        <UiField
          variant="settings"
          tone="inverse"
          label="图片间距"
          control-id="imageGapSlider"
          class="reader-controls__setting-field"
        >
          <div class="reader-controls__setting-control">
            <UiInput
              type="range"
              id="imageGapSlider"
              class="reader-controls__range"
              min="0"
              max="50"
              :model-value="settings.imageGap"
              @update:model-value="value => updateImageGap(Number(value))"
            />
            <span class="reader-controls__setting-value">{{ settings.imageGap }}px</span>
          </div>
        </UiField>

        <UiField
          variant="settings"
          tone="inverse"
          label="背景颜色"
          class="reader-controls__setting-field"
        >
          <UiColorSwatchGroup
            :model-value="settings.bgColor"
            :options="bgColorPresets"
            aria-label="阅读背景颜色"
            @change="updateBgColor"
          />
        </UiField>
      </div>
    </div>
  </OverlayLayer>
</template>

<style scoped>
.reader-controls__chapter-nav-layer,
.reader-controls__scroll-top-layer,
.reader-controls__settings-panel {
  --reader-controls-chapter-nav-start: color-mix(
    in srgb,
    var(--color-surface-inverse) 95%,
    transparent
  );
  --reader-controls-chapter-nav-end: color-mix(
    in srgb,
    var(--color-surface-inverse) 80%,
    transparent
  );
  --reader-controls-settings-overlay-background: var(--color-overlay-scrim);
  --reader-controls-settings-panel-background: var(--color-surface-inverse-raised);
  --reader-controls-settings-panel-shadow: var(--color-overlay-scrim-subtle);
  --reader-controls-settings-divider: var(--color-overlay-inverse-subtle);
  --reader-controls-setting-label-text: color-mix(
    in srgb,
    var(--color-text-inverse) 70%,
    transparent
  );
  --reader-controls-range-track: var(--color-overlay-inverse-muted);
}

.reader-controls__chapter-nav-layer {
  display: flex;
  align-items: flex-end;
  justify-content: center;
}

.reader-controls__chapter-nav {
  height: 60px;
  width: 100%;
  background: linear-gradient(
    to top,
    var(--reader-controls-chapter-nav-start),
    var(--reader-controls-chapter-nav-end)
  );
  backdrop-filter: blur(10px);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 24px;
  padding: 0 16px;
}

.reader-controls__scroll-top-layer {
  display: flex;
  align-items: flex-end;
  justify-content: flex-end;
  padding: 0 24px 80px 0;
}

.reader-controls__scroll-top-button {
  z-index: var(--z-dropdown);
}

.reader-controls__settings-panel {
  display: block;
}

.reader-controls__settings-overlay {
  position: absolute;
  inset: 0;
  background: var(--reader-controls-settings-overlay-background);
}

.reader-controls__settings-content {
  position: absolute;
  top: 56px;
  right: 16px;
  width: 300px;
  background: var(--reader-controls-settings-panel-background);
  border-radius: 12px;
  box-shadow: 0 8px 32px var(--reader-controls-settings-panel-shadow);
  overflow: hidden;
  outline: none;
}

.reader-controls__settings-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border-bottom: 1px solid var(--reader-controls-settings-divider);
  background: var(--reader-controls-settings-panel-background);
}

.reader-controls__settings-title {
  margin: 0;
  color: var(--color-text-inverse);
  font-size: 16px;
  font-weight: 500;
}

.reader-controls__close-button {
  flex: 0 0 auto;
}

.reader-controls__settings-body {
  padding: 16px;
  background: var(--reader-controls-settings-panel-background);
}

.reader-controls__setting-field {
  --ui-field-inverse-label-color: var(--reader-controls-setting-label-text);
  --ui-field-label-font-size: 13px;
  --ui-field-label-font-weight: 400;
  --ui-field-settings-header-margin-bottom: 8px;

  margin-bottom: 20px;
}

.reader-controls__setting-field:last-child {
  margin-bottom: 0;
}

.reader-controls__setting-control {
  display: flex;
  align-items: center;
  gap: 12px;
}

.reader-controls__range {
  flex: 1;
  height: 4px;
  appearance: none;
  background: var(--reader-controls-range-track);
  border-radius: 2px;
  outline: none;
}

.reader-controls__range::-webkit-slider-thumb {
  appearance: none;
  width: 16px;
  height: 16px;
  background: var(--color-action-brand);
  border-radius: 50%;
  cursor: pointer;
}

.reader-controls__setting-value {
  color: var(--color-text-inverse);
  font-size: 13px;
  min-width: 45px;
  text-align: right;
}

@media (--breakpoint-md-down) {
  .reader-controls__settings-content {
    right: 8px;
    left: 8px;
    width: auto;
  }

  .reader-controls__scroll-top-button {
    transform: scale(0.9);
    transform-origin: right bottom;
  }
}
</style>
