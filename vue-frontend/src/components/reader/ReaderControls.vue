<script setup lang="ts">

import UiInput from '@/components/ui/UiInput.vue'

import UiButton from '@/components/ui/UiButton.vue'
import OverlayLayer from '@/components/ui/OverlayLayer.vue'
import { ref, onMounted, onUnmounted } from 'vue'

export interface ReaderSettings {
  imageWidth: number
  imageGap: number
  bgColor: string
}

interface StoredReaderSettings extends ReaderSettings {
  readerSettingsSchemaVersion: 1
}

const READER_SETTINGS_KEY = 'readerSettings'
const READER_SETTINGS_SCHEMA_VERSION = 1
const DEFAULT_READER_SETTINGS: ReaderSettings = {
  imageWidth: 100,
  imageGap: 8,
  bgColor: '#1a1a2e'
}

const props = defineProps<{
  currentPage: number
  totalPages: number
  hasPrevChapter: boolean
  hasNextChapter: boolean
  showChapterNav: boolean
}>()

const emit = defineEmits<{
  (e: 'navigateChapter', direction: 'prev' | 'next'): void
  (e: 'settingsChange', settings: ReaderSettings): void
}>()

const settings = ref<ReaderSettings>({ ...DEFAULT_READER_SETTINGS })
const isSettingsPanelOpen = ref(false)
const showScrollTopBtn = ref(false)
const bgColorPresets = [
  { color: '#1a1a2e', name: '深蓝' },
  { color: '#ffffff', name: '白色' },
  { color: '#f5f5dc', name: '米色' },
  { color: '#2d2d2d', name: '深灰' }
]
const bgColorValues = new Set(bgColorPresets.map((preset) => preset.color))

function openSettings() {
  isSettingsPanelOpen.value = true
}

function closeSettings() {
  isSettingsPanelOpen.value = false
}

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' })
}

function handleScroll() {
  const scrollTop = window.scrollY
  showScrollTopBtn.value = scrollTop > 500
}

function handleKeydown(e: KeyboardEvent) {
  switch (e.key) {
    case 'Escape':
      closeSettings()
      break
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
      window.scrollTo({ top: 0, behavior: 'smooth' })
      break
    case 'End':
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
      break
  }
}

function isNumberInRange(value: unknown, min: number, max: number): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max
}

function isStoredReaderSettings(value: unknown): value is StoredReaderSettings {
  if (!value || typeof value !== 'object') return false

  const candidate = value as Partial<StoredReaderSettings>
  return (
    candidate.readerSettingsSchemaVersion === READER_SETTINGS_SCHEMA_VERSION &&
    isNumberInRange(candidate.imageWidth, 50, 100) &&
    isNumberInRange(candidate.imageGap, 0, 50) &&
    typeof candidate.bgColor === 'string' &&
    bgColorValues.has(candidate.bgColor)
  )
}

function loadSettings() {
  const saved = localStorage.getItem(READER_SETTINGS_KEY)
  if (saved) {
    try {
      const parsed: unknown = JSON.parse(saved)
      if (isStoredReaderSettings(parsed)) {
        settings.value = {
          imageWidth: parsed.imageWidth,
          imageGap: parsed.imageGap,
          bgColor: parsed.bgColor
        }
      }
    } catch {
      settings.value = { ...DEFAULT_READER_SETTINGS }
    }
  }
  applySettings()
}

function saveSettings() {
  const payload: StoredReaderSettings = {
    readerSettingsSchemaVersion: READER_SETTINGS_SCHEMA_VERSION,
    ...settings.value
  }
  try {
    localStorage.setItem(READER_SETTINGS_KEY, JSON.stringify(payload))
  } catch {
    // 当前会话内设置已应用；持久化不可写时静默降级。
  }
}

function applySettings() {
  document.documentElement.style.setProperty('--reader-page-background', settings.value.bgColor)
  document.documentElement.style.setProperty('--reader-image-width', `${settings.value.imageWidth}%`)
  document.documentElement.style.setProperty('--reader-gap', `${settings.value.imageGap}px`)
  emit('settingsChange', settings.value)
}

function updateImageWidth(value: number) {
  settings.value.imageWidth = value
  applySettings()
  saveSettings()
}

function updateImageGap(value: number) {
  settings.value.imageGap = value
  applySettings()
  saveSettings()
}

function updateBgColor(color: string) {
  settings.value.bgColor = color
  applySettings()
  saveSettings()
}

function navigateChapter(direction: 'prev' | 'next') {
  emit('navigateChapter', direction)
}

onMounted(() => {
  loadSettings()

  window.addEventListener('scroll', handleScroll)
  document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  window.removeEventListener('scroll', handleScroll)
  document.removeEventListener('keydown', handleKeydown)

  document.documentElement.style.removeProperty('--reader-page-background')
  document.documentElement.style.removeProperty('--reader-image-width')
  document.documentElement.style.removeProperty('--reader-gap')
})

defineExpose({
  openSettings,
  closeSettings,
  settings
})
</script>

<template>
  <OverlayLayer v-if="showChapterNav" class="reader-controls__chapter-nav-layer" passthrough>
    <nav id="chapterNav" class="reader-controls__chapter-nav">
      <UiButton
        variant="toolbar"
        id="prevChapterBtn"
        class="reader-controls__nav-button"
        :disabled="!hasPrevChapter"
        @click="navigateChapter('prev')"
      >
        <span class="reader-controls__nav-icon">◀</span>
        <span class="reader-controls__nav-text">上一章</span>
      </UiButton>
      <UiButton
        variant="toolbar"
        id="nextChapterBtn"
        class="reader-controls__nav-button"
        :disabled="!hasNextChapter"
        @click="navigateChapter('next')"
      >
        <span class="reader-controls__nav-text">下一章</span>
        <span class="reader-controls__nav-icon">▶</span>
      </UiButton>
    </nav>
  </OverlayLayer>

  <OverlayLayer v-show="showScrollTopBtn" class="reader-controls__scroll-top-layer" passthrough>
    <UiButton
      variant="toolbar"
      id="scrollTopBtn"
      class="reader-controls__scroll-top-button"
      title="回到顶部"
      aria-label="回到顶部"
      @click="scrollToTop"
    >
      <span>↑</span>
    </UiButton>
  </OverlayLayer>

  <OverlayLayer v-if="isSettingsPanelOpen" id="settingsPanel" class="reader-controls__settings-panel" level="popover">
    <div class="reader-controls__settings-overlay" @click="closeSettings"></div>
    <div class="reader-controls__settings-content">
      <div class="reader-controls__settings-header">
        <h3>阅读设置</h3>
        <UiButton variant="toolbar" class="reader-controls__close-button" aria-label="关闭阅读设置" @click="closeSettings">×</UiButton>
      </div>
      <div class="reader-controls__settings-body">
        <div class="reader-controls__setting-item">
          <label>图片宽度</label>
          <div class="reader-controls__setting-control">
            <UiInput
              type="range"
              id="imageWidthSlider"
              class="reader-controls__range"
              min="50"
              max="100"
              :value="settings.imageWidth"
              @input="updateImageWidth(Number(($event.target as HTMLInputElement).value))"
            />
            <span id="imageWidthValue">{{ settings.imageWidth }}%</span>
          </div>
        </div>

        <div class="reader-controls__setting-item">
          <label>图片间距</label>
          <div class="reader-controls__setting-control">
            <UiInput
              type="range"
              id="imageGapSlider"
              class="reader-controls__range"
              min="0"
              max="50"
              :value="settings.imageGap"
              @input="updateImageGap(Number(($event.target as HTMLInputElement).value))"
            />
            <span id="imageGapValue">{{ settings.imageGap }}px</span>
          </div>
        </div>

        <div class="reader-controls__setting-item">
          <label>背景颜色</label>
          <div class="reader-controls__setting-control reader-controls__bg-options">
            <UiButton
              variant="toolbar"
              v-for="preset in bgColorPresets"
              :key="preset.color"
              class="reader-controls__bg-option"
              :class="{ active: settings.bgColor === preset.color }"
              :data-bg="preset.color"
              :style="{ background: preset.color }"
              :title="preset.name"
              :aria-label="`设置背景颜色为${preset.name}`"
              @click="updateBgColor(preset.color)"
            ></UiButton>
          </div>
        </div>
      </div>
    </div>
  </OverlayLayer>
</template>

<style scoped>
.reader-controls__chapter-nav-layer,
.reader-controls__scroll-top-layer,
.reader-controls__settings-panel {
  --reader-controls-chapter-nav-start: rgba(26, 26, 46, .95);
  --reader-controls-chapter-nav-end: rgba(26, 26, 46, .8);
  --reader-controls-button-background: rgba(255, 255, 255, .1);
  --reader-controls-button-hover-background: rgba(255, 255, 255, .2);
  --reader-controls-button-border: rgba(255, 255, 255, .2);
  --reader-controls-scroll-top-hover-shadow: rgba(102, 126, 234, .5);
  --reader-controls-settings-overlay-background: rgba(0, 0, 0, .5);
  --reader-controls-settings-panel-background: #2d2d44;
  --reader-controls-settings-panel-shadow: rgba(0, 0, 0, .3);
  --reader-controls-settings-divider: rgba(255, 255, 255, .1);
  --reader-controls-setting-label-text: rgba(255, 255, 255, .7);
  --reader-controls-swatch-active-ring: rgba(102, 126, 234, .3);
}

.reader-controls__chapter-nav-layer {
  display: flex;
  align-items: flex-end;
  justify-content: center;
}

.reader-controls__chapter-nav {
  height: 60px;
  width: 100%;
  background: linear-gradient(to top, var(--reader-controls-chapter-nav-start), var(--reader-controls-chapter-nav-end));
  backdrop-filter: blur(10px);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 24px;
  padding: 0 16px;
}

.reader-controls__nav-button {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  background: var(--reader-controls-button-background);
  border: 1px solid var(--reader-controls-button-border);
  border-radius: 8px;
  color: var(--color-text-inverse);
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
}

.reader-controls__nav-button:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.reader-controls__nav-button:hover:not(:disabled) {
  background: var(--reader-controls-button-hover-background);
}

.reader-controls__nav-icon {
  font-size: 12px;
}

.reader-controls__scroll-top-layer {
  display: flex;
  align-items: flex-end;
  justify-content: flex-end;
  padding: 0 24px 80px 0;
}

.reader-controls__scroll-top-button {
  width: 48px;
  height: 48px;
  background: var(--color-action-primary, var(--color-action-brand));
  border: none;
  border-radius: 50%;
  color: var(--color-text-inverse);
  font-size: 20px;
  cursor: pointer;
  box-shadow: 0 4px 12px var(--shadow-action-brand);
  transition: all 0.3s;
  z-index: var(--z-dropdown);
}

.reader-controls__scroll-top-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px var(--reader-controls-scroll-top-hover-shadow);
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
}

.reader-controls__settings-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border-bottom: 1px solid var(--reader-controls-settings-divider);
  background: var(--reader-controls-settings-panel-background);
}

.reader-controls__settings-header h3 {
  margin: 0;
  color: var(--color-text-inverse);
  font-size: 16px;
  font-weight: 500;
}

.reader-controls__close-button {
  width: 28px;
  height: 28px;
  background: var(--reader-controls-button-background);
  border: none;
  border-radius: 50%;
  color: var(--color-text-inverse);
  font-size: 18px;
  cursor: pointer;
  transition: background 0.2s;
}

.reader-controls__close-button:hover {
  background: var(--reader-controls-button-hover-background);
}

.reader-controls__settings-body {
  padding: 16px;
  background: var(--reader-controls-settings-panel-background);
}

.reader-controls__setting-item {
  margin-bottom: 20px;
}

.reader-controls__setting-item:last-child {
  margin-bottom: 0;
}

.reader-controls__setting-item label {
  display: block;
  color: var(--reader-controls-setting-label-text);
  font-size: 13px;
  margin-bottom: 8px;
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
  background: var(--reader-controls-button-hover-background);
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

.reader-controls__setting-control span {
  color: var(--color-text-inverse);
  font-size: 13px;
  min-width: 45px;
  text-align: right;
}

.reader-controls__bg-options {
  display: flex;
  gap: 8px;
}

.reader-controls__bg-option {
  width: 32px;
  height: 32px;
  border: 2px solid transparent;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.reader-controls__bg-option:hover {
  transform: scale(1.1);
}

.reader-controls__bg-option.active {
  border-color: var(--color-border-brand-gradient);
  box-shadow: 0 0 0 2px var(--reader-controls-swatch-active-ring);
}

@media (--breakpoint-md-down) {
  .reader-controls__settings-content {
    right: 8px;
    left: 8px;
    width: auto;
  }

  .reader-controls__nav-button {
    padding: 10px 16px;
    font-size: 13px;
  }

  .reader-controls__scroll-top-button {
    right: 16px;
    bottom: 72px;
    width: 40px;
    height: 40px;
  }
}
</style>
