<template>
  <div v-if="visible" class="kana-keyboard">
    <div class="kana-keyboard__header">
      <span class="kana-keyboard__title">50音键盘</span>
      <ProductSegmentedTabs
        class="kana-keyboard__tabs"
        :tabs="tabs"
        :active-tab="activeTab"
        aria-label="假名分类"
        layout="scroll"
        @select="handleActiveTabSelect"
      />
      <UiIconButton
        variant="inverse"
        size="xs"
        shape="circle"
        label="关闭50音键盘"
        title="关闭"
        @click="close"
      >
        <UiIcon name="x" size="14" />
      </UiIconButton>
    </div>

    <div class="kana-keyboard__options">
      <div class="kana-keyboard__mode-select">
        <ProductSegmentedTabs
          class="kana-keyboard__mode-tabs"
          :tabs="kanaModeTabs"
          :active-tab="kanaMode"
          appearance="radio"
          aria-label="假名字符类型"
          @select="handleKanaModeSelect"
        />
      </div>
      <UiField
        class="kana-keyboard__target-select"
        label="输入到："
        control-id="kanaTargetField"
        layout="inline"
      >
        <UiSelect
          id="kanaTargetField"
          :model-value="targetField"
          :options="targetFieldOptions"
          @change="handleTargetFieldChange"
        />
      </UiField>
    </div>

    <div
      class="kana-keyboard__tab-content"
      :class="{ 'kana-keyboard__tab-content--active': activeTab === 'basic' }"
    >
      <table class="kana-keyboard__table">
        <thead>
          <tr>
            <th></th>
            <th>あ段</th>
            <th>い段</th>
            <th>う段</th>
            <th>え段</th>
            <th>お段</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in basicKana" :key="row.label">
            <td class="kana-keyboard__row-label">{{ row.label }}</td>
            <td v-for="(kana, idx) in row.chars" :key="idx">
              <UiButton
                variant="toolbar"
                v-if="kana"
                class="kana-keyboard__key"
                :class="{ 'kana-keyboard__key--pressed': pressedKey === kana.h }"
                @click="insertKana(kana)"
              >
                <span class="kana-keyboard__hiragana">{{ kana.h }}</span>
                <span class="kana-keyboard__katakana">{{ kana.k }}</span>
              </UiButton>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div
      class="kana-keyboard__tab-content"
      :class="{ 'kana-keyboard__tab-content--active': activeTab === 'dakuten' }"
    >
      <table class="kana-keyboard__table">
        <thead>
          <tr>
            <th></th>
            <th>あ段</th>
            <th>い段</th>
            <th>う段</th>
            <th>え段</th>
            <th>お段</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in dakutenKana" :key="row.label">
            <td class="kana-keyboard__row-label">{{ row.label }}</td>
            <td v-for="(kana, idx) in row.chars" :key="idx">
              <UiButton
                variant="toolbar"
                v-if="kana"
                class="kana-keyboard__key"
                :class="{ 'kana-keyboard__key--pressed': pressedKey === kana.h }"
                @click="insertKana(kana)"
              >
                <span class="kana-keyboard__hiragana">{{ kana.h }}</span>
                <span class="kana-keyboard__katakana">{{ kana.k }}</span>
              </UiButton>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div
      class="kana-keyboard__tab-content"
      :class="{ 'kana-keyboard__tab-content--active': activeTab === 'combo' }"
    >
      <table class="kana-keyboard__table kana-keyboard__table--combo">
        <thead>
          <tr>
            <th></th>
            <th>ゃ</th>
            <th>ゅ</th>
            <th>ょ</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in comboKana" :key="row.label">
            <td class="kana-keyboard__row-label">{{ row.label }}</td>
            <td v-for="(kana, idx) in row.chars" :key="idx">
              <UiButton
                variant="toolbar"
                v-if="kana"
                class="kana-keyboard__key"
                :class="{ 'kana-keyboard__key--pressed': pressedKey === kana.h }"
                @click="insertKana(kana)"
              >
                <span class="kana-keyboard__hiragana">{{ kana.h }}</span>
                <span class="kana-keyboard__katakana">{{ kana.k }}</span>
              </UiButton>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div
      class="kana-keyboard__tab-content"
      :class="{ 'kana-keyboard__tab-content--active': activeTab === 'special' }"
    >
      <div class="kana-keyboard__special-grid">
        <UiButton
          variant="toolbar"
          v-for="char in specialChars"
          :key="char.char"
          class="kana-keyboard__key kana-keyboard__key--special"
          :class="{ 'kana-keyboard__key--pressed': pressedKey === char.char }"
          @click="insertSpecialChar(char.char)"
        >
          {{ char.char }}
          <span v-if="char.label" class="kana-keyboard__char-label">{{ char.label }}</span>
        </UiButton>
      </div>
    </div>

    <div class="kana-keyboard__footer">
      <UiButton class="kana-keyboard__backspace" variant="danger" size="sm" @click="deleteChar">
        <UiIcon name="arrow-left" size="14" />
        退格
      </UiButton>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onUnmounted, ref } from 'vue'
import ProductSegmentedTabs from '@/components/product/ProductSegmentedTabs.vue'
import type { ProductSegmentedTab } from '@/components/product/ProductSegmentedTabs.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiSelect from '@/components/ui/UiSelect.vue'

interface KanaChar {
  h: string
  k: string
}

interface KanaRow {
  label: string
  chars: (KanaChar | null)[]
}

interface SpecialChar {
  char: string
  label?: string
}

type KanaMode = 'hiragana' | 'katakana'
type KanaTab = 'basic' | 'dakuten' | 'combo' | 'special'

const props = withDefaults(defineProps<{
  visible?: boolean
  defaultTarget?: 'original' | 'translated'
}>(), {
  visible: false,
  defaultTarget: 'original'
})

const emit = defineEmits<{
  (e: 'close'): void
  (e: 'insert', char: string, target: 'original' | 'translated'): void
  (e: 'delete', target: 'original' | 'translated'): void
}>()

const activeTab = ref<KanaTab>('basic')
const kanaMode = ref<KanaMode>('hiragana')
const targetField = ref<'original' | 'translated'>(props.defaultTarget)
const pressedKey = ref<string | null>(null)
let pressFeedbackTimer: ReturnType<typeof setTimeout> | null = null

const targetFieldOptions = [
  { label: '原文', value: 'original' },
  { label: '译文', value: 'translated' }
]

const tabs: ProductSegmentedTab[] = [
  { id: 'basic', label: '基本' },
  { id: 'dakuten', label: '浊/半浊音' },
  { id: 'combo', label: '拗音' },
  { id: 'special', label: '特殊' }
]

const kanaModeTabs: ProductSegmentedTab[] = [
  { id: 'hiragana', label: '平假名' },
  { id: 'katakana', label: '片假名' }
]

const basicKana: KanaRow[] = [
  { label: 'あ行', chars: [{ h: 'あ', k: 'ア' }, { h: 'い', k: 'イ' }, { h: 'う', k: 'ウ' }, { h: 'え', k: 'エ' }, { h: 'お', k: 'オ' }] },
  { label: 'か行', chars: [{ h: 'か', k: 'カ' }, { h: 'き', k: 'キ' }, { h: 'く', k: 'ク' }, { h: 'け', k: 'ケ' }, { h: 'こ', k: 'コ' }] },
  { label: 'さ行', chars: [{ h: 'さ', k: 'サ' }, { h: 'し', k: 'シ' }, { h: 'す', k: 'ス' }, { h: 'せ', k: 'セ' }, { h: 'そ', k: 'ソ' }] },
  { label: 'た行', chars: [{ h: 'た', k: 'タ' }, { h: 'ち', k: 'チ' }, { h: 'つ', k: 'ツ' }, { h: 'て', k: 'テ' }, { h: 'と', k: 'ト' }] },
  { label: 'な行', chars: [{ h: 'な', k: 'ナ' }, { h: 'に', k: 'ニ' }, { h: 'ぬ', k: 'ヌ' }, { h: 'ね', k: 'ネ' }, { h: 'の', k: 'ノ' }] },
  { label: 'は行', chars: [{ h: 'は', k: 'ハ' }, { h: 'ひ', k: 'ヒ' }, { h: 'ふ', k: 'フ' }, { h: 'へ', k: 'ヘ' }, { h: 'ほ', k: 'ホ' }] },
  { label: 'ま行', chars: [{ h: 'ま', k: 'マ' }, { h: 'み', k: 'ミ' }, { h: 'む', k: 'ム' }, { h: 'め', k: 'メ' }, { h: 'も', k: 'モ' }] },
  { label: 'や行', chars: [{ h: 'や', k: 'ヤ' }, null, { h: 'ゆ', k: 'ユ' }, null, { h: 'よ', k: 'ヨ' }] },
  { label: 'ら行', chars: [{ h: 'ら', k: 'ラ' }, { h: 'り', k: 'リ' }, { h: 'る', k: 'ル' }, { h: 'れ', k: 'レ' }, { h: 'ろ', k: 'ロ' }] },
  { label: 'わ行', chars: [{ h: 'わ', k: 'ワ' }, null, null, null, { h: 'を', k: 'ヲ' }] },
  { label: 'ん', chars: [{ h: 'ん', k: 'ン' }, null, null, null, null] }
]

const dakutenKana: KanaRow[] = [
  { label: 'が行', chars: [{ h: 'が', k: 'ガ' }, { h: 'ぎ', k: 'ギ' }, { h: 'ぐ', k: 'グ' }, { h: 'げ', k: 'ゲ' }, { h: 'ご', k: 'ゴ' }] },
  { label: 'ざ行', chars: [{ h: 'ざ', k: 'ザ' }, { h: 'じ', k: 'ジ' }, { h: 'ず', k: 'ズ' }, { h: 'ぜ', k: 'ゼ' }, { h: 'ぞ', k: 'ゾ' }] },
  { label: 'だ行', chars: [{ h: 'だ', k: 'ダ' }, { h: 'ぢ', k: 'ヂ' }, { h: 'づ', k: 'ヅ' }, { h: 'で', k: 'デ' }, { h: 'ど', k: 'ド' }] },
  { label: 'ば行', chars: [{ h: 'ば', k: 'バ' }, { h: 'び', k: 'ビ' }, { h: 'ぶ', k: 'ブ' }, { h: 'べ', k: 'ベ' }, { h: 'ぼ', k: 'ボ' }] },
  { label: 'ぱ行', chars: [{ h: 'ぱ', k: 'パ' }, { h: 'ぴ', k: 'ピ' }, { h: 'ぷ', k: 'プ' }, { h: 'ぺ', k: 'ペ' }, { h: 'ぽ', k: 'ポ' }] }
]

const comboKana: KanaRow[] = [
  { label: 'きゃ行', chars: [{ h: 'きゃ', k: 'キャ' }, { h: 'きゅ', k: 'キュ' }, { h: 'きょ', k: 'キョ' }] },
  { label: 'しゃ行', chars: [{ h: 'しゃ', k: 'シャ' }, { h: 'しゅ', k: 'シュ' }, { h: 'しょ', k: 'ショ' }] },
  { label: 'ちゃ行', chars: [{ h: 'ちゃ', k: 'チャ' }, { h: 'ちゅ', k: 'チュ' }, { h: 'ちょ', k: 'チョ' }] },
  { label: 'にゃ行', chars: [{ h: 'にゃ', k: 'ニャ' }, { h: 'にゅ', k: 'ニュ' }, { h: 'にょ', k: 'ニョ' }] },
  { label: 'ひゃ行', chars: [{ h: 'ひゃ', k: 'ヒャ' }, { h: 'ひゅ', k: 'ヒュ' }, { h: 'ひょ', k: 'ヒョ' }] },
  { label: 'みゃ行', chars: [{ h: 'みゃ', k: 'ミャ' }, { h: 'みゅ', k: 'ミュ' }, { h: 'みょ', k: 'ミョ' }] },
  { label: 'りゃ行', chars: [{ h: 'りゃ', k: 'リャ' }, { h: 'りゅ', k: 'リュ' }, { h: 'りょ', k: 'リョ' }] },
  { label: 'ぎゃ行', chars: [{ h: 'ぎゃ', k: 'ギャ' }, { h: 'ぎゅ', k: 'ギュ' }, { h: 'ぎょ', k: 'ギョ' }] },
  { label: 'じゃ行', chars: [{ h: 'じゃ', k: 'ジャ' }, { h: 'じゅ', k: 'ジュ' }, { h: 'じょ', k: 'ジョ' }] },
  { label: 'びゃ行', chars: [{ h: 'びゃ', k: 'ビャ' }, { h: 'びゅ', k: 'ビュ' }, { h: 'びょ', k: 'ビョ' }] },
  { label: 'ぴゃ行', chars: [{ h: 'ぴゃ', k: 'ピャ' }, { h: 'ぴゅ', k: 'ピュ' }, { h: 'ぴょ', k: 'ピョ' }] }
]

const specialChars: SpecialChar[] = [
  { char: 'っ', label: '促音' },
  { char: 'ッ', label: '促音' },
  { char: 'ー', label: '长音' },
  { char: '〜', label: '波浪' },
  { char: '。', label: '句号' },
  { char: '、', label: '顿号' },
  { char: '「', label: '引号' },
  { char: '」', label: '引号' },
  { char: '『', label: '双引' },
  { char: '』', label: '双引' },
  { char: '…', label: '省略' },
  { char: '・', label: '中点' },
  { char: '！', label: '感叹' },
  { char: '？', label: '问号' },
  { char: '♪', label: '音符' },
  { char: '♡', label: '心形' },
  { char: '★', label: '星形' },
  { char: '☆', label: '空星' },
  { char: 'ぁ', label: '小あ' },
  { char: 'ぃ', label: '小い' },
  { char: 'ぅ', label: '小う' },
  { char: 'ぇ', label: '小え' },
  { char: 'ぉ', label: '小お' },
  { char: 'ァ', label: '小ア' },
  { char: 'ィ', label: '小イ' },
  { char: 'ゥ', label: '小ウ' },
  { char: 'ェ', label: '小エ' },
  { char: 'ォ', label: '小オ' },
  { char: 'ゃ', label: '小や' },
  { char: 'ゅ', label: '小ゆ' },
  { char: 'ょ', label: '小よ' },
  { char: 'ャ', label: '小ヤ' },
  { char: 'ュ', label: '小ユ' },
  { char: 'ョ', label: '小ヨ' }
]

function close(): void {
  emit('close')
}

function handleActiveTabSelect(tabId: string): void {
  if (tabId === 'basic' || tabId === 'dakuten' || tabId === 'combo' || tabId === 'special') {
    activeTab.value = tabId
  }
}

function handleKanaModeSelect(tabId: string): void {
  if (tabId === 'hiragana' || tabId === 'katakana') {
    kanaMode.value = tabId
  }
}

function handleTargetFieldChange(value: string | number): void {
  if (value === 'original' || value === 'translated') {
    targetField.value = value
  }
}

function flashPressedKey(key: string): void {
  if (pressFeedbackTimer) {
    clearTimeout(pressFeedbackTimer)
  }
  pressedKey.value = key
  pressFeedbackTimer = setTimeout(() => {
    pressedKey.value = null
    pressFeedbackTimer = null
  }, 100)
}

function insertKana(kana: KanaChar): void {
  const char = kanaMode.value === 'hiragana' ? kana.h : kana.k
  flashPressedKey(kana.h)

  emit('insert', char, targetField.value)
}

function insertSpecialChar(char: string): void {
  flashPressedKey(char)

  emit('insert', char, targetField.value)
}

function deleteChar(): void {
  emit('delete', targetField.value)
}

onUnmounted(() => {
  if (pressFeedbackTimer) {
    clearTimeout(pressFeedbackTimer)
  }
})
</script>

<style scoped>
.kana-keyboard {
  --japanese-keyboard-panel-shadow: color-mix(in srgb, var(--color-overlay-backdrop-solid) 15%, transparent);
  --japanese-keyboard-header-start: var(--color-status-error-bright);
  --japanese-keyboard-header-end: var(--color-status-error-bright-hover);
  --japanese-keyboard-header-button-background: var(--color-overlay-inverse-muted);
  --japanese-keyboard-kana-key-background: var(--color-surface-quiet);
  --japanese-keyboard-kana-key-hover-background: var(--color-surface-interactive-hover);
  --japanese-keyboard-kana-key-active-background: color-mix(in srgb, var(--color-status-info) 25%, var(--color-surface-base));
  --japanese-keyboard-kana-key-hover-border: var(--color-action-primary);
  --japanese-keyboard-kana-key-hover-shadow: color-mix(in srgb, var(--color-action-primary) 30%, transparent);

  background: var(--color-surface-base);
  border: 1px solid var(--color-border-default);
  border-radius: 8px;
  box-shadow: 0 4px 12px var(--japanese-keyboard-panel-shadow);
  margin-top: 10px;
  overflow: hidden;
  color: var(--color-text-default);
}

.kana-keyboard__header {
  --ui-button-icon-width: 24px;
  --ui-button-icon-height: 24px;

  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: linear-gradient(135deg, var(--japanese-keyboard-header-start) 0%, var(--japanese-keyboard-header-end) 100%);
  color: var(--color-text-inverse);
}

.kana-keyboard__title {
  font-weight: 600;
  font-size: 13px;
  color: var(--color-text-inverse);
}

.kana-keyboard__tabs {
  --product-segmented-tabs-background: transparent;
  --product-segmented-tabs-border: transparent;
  --product-segmented-tabs-padding: 0;
  --product-segmented-tabs-radius: 0;
  --product-segmented-tabs-gap: 4px;
  --product-segmented-tabs-tab-padding: 4px 10px;
  --product-segmented-tabs-tab-radius: 4px;
  --product-segmented-tabs-tab-background: var(--japanese-keyboard-header-button-background);
  --product-segmented-tabs-tab-font-size: 11px;
  --product-segmented-tabs-tab-line-height: 1.35;
  --product-segmented-tabs-active-background: var(--color-surface-base);
  --product-segmented-tabs-active-text: var(--color-text-danger-strong);
  --product-segmented-tabs-active-shadow: none;
  --product-segmented-tabs-text: var(--color-text-inverse);

  flex: 0 1 auto;
  max-width: none;
}

.kana-keyboard__options {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 8px 12px;
  background: var(--color-surface-subtle);
  border-bottom: 1px solid var(--color-border-default);
}

.kana-keyboard__mode-select,
.kana-keyboard__target-select {
  --ui-field-inline-label-color: var(--color-text-default);
  --ui-field-inline-label-font-size: 12px;
  --ui-field-inline-label-font-weight: 500;

  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--color-text-default);
  font-size: 12px;
}

.kana-keyboard__mode-tabs {
  --product-segmented-tabs-background: transparent;
  --product-segmented-tabs-border: transparent;
  --product-segmented-tabs-padding: 0;
  --product-segmented-tabs-radius: 0;
  --product-segmented-tabs-gap: 12px;
  --product-segmented-tabs-active-text: var(--color-text-default);
  --product-segmented-tabs-text: var(--color-text-default);
  --product-segmented-tabs-tab-font-size: 12px;
  --product-segmented-tabs-tab-font-weight: 400;
  --product-segmented-tabs-radio-tab-gap: 4px;
  --product-segmented-tabs-radio-border: var(--color-text-secondary);
  --product-segmented-tabs-radio-active-color: var(--color-text-danger-strong);
  --product-segmented-tabs-radio-inner-color: var(--color-surface-subtle);
}

.kana-keyboard__target-select {
  --ui-select-min-height: 40px;
  --ui-select-padding: 4px 8px;
  --ui-select-radius: 4px;
  --ui-select-font-size: 12px;

  width: 210px;
}

.kana-keyboard__tab-content {
  padding: 10px;
  max-height: 280px;
  overflow-y: auto;
  background: var(--color-surface-base);
}

.kana-keyboard__table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  background: var(--color-surface-base);
}

.kana-keyboard__table th,
.kana-keyboard__table td {
  padding: 2px;
  text-align: center;
  vertical-align: middle;
  background: var(--color-surface-base);
}

.kana-keyboard__table th {
  color: var(--color-text-secondary);
  font-weight: 500;
  font-size: 11px;
  padding-bottom: 6px;
}

.kana-keyboard__row-label {
  color: var(--color-text-subtle);
  font-size: 11px;
  font-weight: 500;
  padding-right: 8px;
  text-align: right;
  white-space: nowrap;
}

.kana-keyboard__key {
  width: 42px;
  height: 42px;
  border: 1px solid var(--color-border-subtle);
  border-radius: 6px;
  background: var(--japanese-keyboard-kana-key-background);
  color: var(--color-text-default);
  font-size: 13px;
  line-height: 1.2;
  cursor: pointer;
  transition: all 0.15s;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 2px;
}

.kana-keyboard__hiragana {
  color: var(--color-text-default);
  font-size: 13px;
  display: block;
}

.kana-keyboard__katakana {
  color: var(--color-text-secondary);
  font-size: 11px;
  display: block;
}

.kana-keyboard__key:hover {
  background: var(--japanese-keyboard-kana-key-hover-background);
  border-color: var(--japanese-keyboard-kana-key-hover-border);
  transform: translateY(-1px);
  box-shadow: 0 2px 6px var(--japanese-keyboard-kana-key-hover-shadow);
}

.kana-keyboard__key:active,
.kana-keyboard__key--pressed {
  transform: translateY(0);
  background: var(--japanese-keyboard-kana-key-active-background);
}

.kana-keyboard__special-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(60px, 1fr));
  gap: 6px;
  background: var(--color-surface-base);
}

.kana-keyboard__key--special {
  width: auto;
  height: auto;
  min-height: 32px;
  padding: 4px 8px;
  font-size: 14px;
  color: var(--color-text-default);
}

.kana-keyboard__char-label {
  font-size: 9px;
  color: var(--color-text-secondary);
  margin-top: 2px;
}

.kana-keyboard__table--combo .kana-keyboard__key {
  width: 56px;
}

.kana-keyboard__footer {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding: 8px 12px;
  background: var(--color-surface-subtle);
  border-top: 1px solid var(--color-border-default);
}

.kana-keyboard__backspace {
  --ui-button-danger-background: color-mix(in srgb, var(--color-status-error) 10%, transparent);
  --ui-button-danger-color: var(--color-text-danger-strong);
  --ui-button-danger-border: 1px solid color-mix(in srgb, var(--color-status-error) 30%, transparent);
  --ui-button-danger-shadow: none;
  --ui-button-sm-padding: 6px 16px;
  --ui-button-sm-font-size: 12px;

  border-radius: 4px;
}

</style>
