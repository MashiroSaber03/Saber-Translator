<template>
  <div v-if="visible" class="kana-keyboard">
    <div class="kana-keyboard-header">
      <span class="kana-keyboard-title">50音键盘</span>
      <div class="kana-keyboard-tabs">
        <UiButton
          variant="toolbar"
          v-for="tab in tabs"
          :key="tab.id"
          class="kana-tab"
          :class="{ active: activeTab === tab.id }"
          @click="activeTab = tab.id"
        >
          {{ tab.label }}
        </UiButton>
      </div>
      <UiButton variant="toolbar" class="kana-keyboard-close" @click="close">✕</UiButton>
    </div>

    <div class="kana-keyboard-options">
      <div class="kana-mode-select">
        <span class="kana-mode-label">字符：</span>
        <UiButton
          variant="toolbar"
          class="kana-mode-button"
          :data-active="kanaMode === 'hiragana'"
          :aria-pressed="kanaMode === 'hiragana'"
          @click="kanaMode = 'hiragana'"
        >
          平假名
        </UiButton>
        <UiButton
          variant="toolbar"
          class="kana-mode-button"
          :data-active="kanaMode === 'katakana'"
          :aria-pressed="kanaMode === 'katakana'"
          @click="kanaMode = 'katakana'"
        >
          片假名
        </UiButton>
      </div>
      <div class="kana-target-select">
        <label>输入到：</label>
        <CustomSelect
          v-model="targetField"
          :options="targetFieldOptions"
        />
      </div>
    </div>

    <div class="kana-tab-content" :class="{ active: activeTab === 'basic' }">
      <table class="kana-table">
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
            <td class="row-label">{{ row.label }}</td>
            <td v-for="(kana, idx) in row.chars" :key="idx">
              <UiButton
                variant="toolbar"
                v-if="kana"
                class="kana-key"
                :class="{ pressed: pressedKey === kana.h }"
                @click="insertKana(kana)"
              >
                <span class="kana-hiragana">{{ kana.h }}</span>
                <span class="kana-katakana">{{ kana.k }}</span>
              </UiButton>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="kana-tab-content" :class="{ active: activeTab === 'dakuten' }">
      <table class="kana-table">
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
            <td class="row-label">{{ row.label }}</td>
            <td v-for="(kana, idx) in row.chars" :key="idx">
              <UiButton
                variant="toolbar"
                v-if="kana"
                class="kana-key"
                :class="{ pressed: pressedKey === kana.h }"
                @click="insertKana(kana)"
              >
                <span class="kana-hiragana">{{ kana.h }}</span>
                <span class="kana-katakana">{{ kana.k }}</span>
              </UiButton>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="kana-tab-content" :class="{ active: activeTab === 'combo' }">
      <table class="kana-table combo-table">
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
            <td class="row-label">{{ row.label }}</td>
            <td v-for="(kana, idx) in row.chars" :key="idx">
              <UiButton
                variant="toolbar"
                v-if="kana"
                class="kana-key"
                :class="{ pressed: pressedKey === kana.h }"
                @click="insertKana(kana)"
              >
                <span class="kana-hiragana">{{ kana.h }}</span>
                <span class="kana-katakana">{{ kana.k }}</span>
              </UiButton>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="kana-tab-content" :class="{ active: activeTab === 'special' }">
      <div class="special-chars-grid">
        <UiButton
          variant="toolbar"
          v-for="char in specialChars"
          :key="char.char"
          class="kana-key special-key"
          :class="{ pressed: pressedKey === char.char }"
          @click="insertSpecialChar(char.char)"
        >
          {{ char.char }}
          <span v-if="char.label" class="char-label">{{ char.label }}</span>
        </UiButton>
      </div>
    </div>

    <div class="kana-keyboard-footer">
      <UiButton variant="toolbar" class="kana-backspace" @click="deleteChar">⌫ 退格</UiButton>
    </div>
  </div>
</template>

<script setup lang="ts">

import UiButton from '@/components/ui/UiButton.vue'
import { onUnmounted, ref } from 'vue'
import CustomSelect from '@/components/common/CustomSelect.vue'

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

const activeTab = ref<'basic' | 'dakuten' | 'combo' | 'special'>('basic')
const kanaMode = ref<'hiragana' | 'katakana'>('hiragana')
const targetField = ref<'original' | 'translated'>(props.defaultTarget)
const pressedKey = ref<string | null>(null)
let pressFeedbackTimer: ReturnType<typeof setTimeout> | null = null

const targetFieldOptions = [
  { label: '原文', value: 'original' },
  { label: '译文', value: 'translated' }
]

const tabs = [
  { id: 'basic' as const, label: '基本' },
  { id: 'dakuten' as const, label: '浊/半浊音' },
  { id: 'combo' as const, label: '拗音' },
  { id: 'special' as const, label: '特殊' }
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
  --japanese-keyboard-panel-shadow: rgba(0, 0, 0, .15);
  --japanese-keyboard-header-start: #ff6b6b;
  --japanese-keyboard-header-end: #ee5a5a;
  --japanese-keyboard-header-button-background: rgba(255, 255, 255, .2);
  --japanese-keyboard-header-button-hover-background: rgba(255, 255, 255, .4);
  --japanese-keyboard-tab-hover-background: rgba(255, 255, 255, .3);
  --japanese-keyboard-mode-button-active-background: #e74c3c;
  --japanese-keyboard-kana-key-background: #f8f9fa;
  --japanese-keyboard-kana-key-hover-background: #e3f2fd;
  --japanese-keyboard-kana-key-active-background: #bbdefb;
  --japanese-keyboard-kana-key-hover-border: #2196f3;
  --japanese-keyboard-kana-key-hover-shadow: rgba(33, 150, 243, .3);
  --japanese-keyboard-backspace-background: rgba(231, 76, 60, .1);
  --japanese-keyboard-backspace-hover-background: rgba(231, 76, 60, .2);
  --japanese-keyboard-backspace-border: rgba(231, 76, 60, .3);
  --japanese-keyboard-backspace-hover-border: rgba(231, 76, 60, .5);

  background: var(--color-surface-base);
  border: 1px solid var(--color-border-default);
  border-radius: 8px;
  box-shadow: 0 4px 12px var(--japanese-keyboard-panel-shadow);
  margin-top: 10px;
  overflow: hidden;
  color: var(--color-text-default);
}

.kana-keyboard-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: linear-gradient(135deg, var(--japanese-keyboard-header-start) 0%, var(--japanese-keyboard-header-end) 100%);
  color: var(--color-text-inverse);
}

.kana-keyboard-title {
  font-weight: 600;
  font-size: 13px;
  color: var(--color-text-inverse);
}

.kana-keyboard-tabs {
  display: flex;
  gap: 4px;
}

.kana-tab {
  padding: 4px 10px;
  border: none;
  border-radius: 4px;
  background: var(--japanese-keyboard-header-button-background);
  color: var(--color-text-inverse);
  font-size: 11px;
  cursor: pointer;
  transition: all 0.2s;
}

.kana-tab:hover {
  background: var(--japanese-keyboard-tab-hover-background);
}

.kana-tab.active {
  background: var(--color-surface-base);
  color: var(--color-text-danger-strong);
  font-weight: 600;
}

.kana-keyboard-close {
  width: 24px;
  height: 24px;
  border: none;
  border-radius: 50%;
  background: var(--japanese-keyboard-header-button-background);
  color: var(--color-text-inverse);
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.kana-keyboard-close:hover {
  background: var(--japanese-keyboard-header-button-hover-background);
}

.kana-keyboard-options {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 8px 12px;
  background: var(--color-surface-subtle);
  border-bottom: 1px solid var(--color-border-default);
}

.kana-mode-select,
.kana-target-select {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--color-text-default);
  font-size: 12px;
}

.kana-mode-label {
  font-weight: 500;
}

.kana-mode-button {
  padding: 4px 10px;
  border: 1px solid var(--color-border-subtle);
  border-radius: 999px;
  background: var(--color-surface-base);
  color: var(--color-text-default);
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s;
}

.kana-mode-button:hover {
  border-color: var(--color-border-accent);
}

.kana-mode-button[data-active='true'] {
  background: var(--japanese-keyboard-mode-button-active-background);
  border-color: var(--japanese-keyboard-mode-button-active-background);
  color: var(--color-text-inverse);
}

.kana-tab-content {
  padding: 10px;
  max-height: 280px;
  overflow-y: auto;
  background: var(--color-surface-base);
}

.kana-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  background: var(--color-surface-base);
}

.kana-table th,
.kana-table td {
  padding: 2px;
  text-align: center;
  vertical-align: middle;
  background: var(--color-surface-base);
}

.kana-table th {
  color: var(--color-text-secondary);
  font-weight: 500;
  font-size: 11px;
  padding-bottom: 6px;
}

.row-label {
  color: var(--color-text-subtle);
  font-size: 11px;
  font-weight: 500;
  padding-right: 8px;
  text-align: right;
  white-space: nowrap;
}

.kana-key {
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

.kana-hiragana {
  color: var(--color-text-default);
  font-size: 13px;
  display: block;
}

.kana-katakana {
  color: var(--color-text-secondary);
  font-size: 11px;
  display: block;
}

.kana-key:hover {
  background: var(--japanese-keyboard-kana-key-hover-background);
  border-color: var(--japanese-keyboard-kana-key-hover-border);
  transform: translateY(-1px);
  box-shadow: 0 2px 6px var(--japanese-keyboard-kana-key-hover-shadow);
}

.kana-key:active,
.kana-key.pressed {
  transform: translateY(0);
  background: var(--japanese-keyboard-kana-key-active-background);
}

.special-chars-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(60px, 1fr));
  gap: 6px;
  background: var(--color-surface-base);
}

.special-key {
  width: auto;
  height: auto;
  min-height: 32px;
  padding: 4px 8px;
  font-size: 14px;
  color: var(--color-text-default);
}

.char-label {
  font-size: 9px;
  color: var(--color-text-secondary);
  margin-top: 2px;
}

.combo-table .kana-key {
  width: 56px;
}

.kana-keyboard-footer {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding: 8px 12px;
  background: var(--color-surface-subtle);
  border-top: 1px solid var(--color-border-default);
}

.kana-backspace {
  padding: 6px 16px;
  background: var(--japanese-keyboard-backspace-background);
  border: 1px solid var(--japanese-keyboard-backspace-border);
  border-radius: 4px;
  color: var(--color-text-danger-strong);
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.kana-backspace:hover {
  background: var(--japanese-keyboard-backspace-hover-background);
  border-color: var(--japanese-keyboard-backspace-hover-border);
}
</style>
