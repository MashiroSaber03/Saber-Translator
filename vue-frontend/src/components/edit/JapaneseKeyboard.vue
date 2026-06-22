<!--
  50音软键盘组件
  提供日语假名输入功能
  - 支持平假名和片假名切换
  - 支持基本、浊音、拗音、特殊字符标签页
  - 支持向指定文本框插入字符
-->
<template>
  <div v-if="visible" class="kana-keyboard">
    <!-- 键盘头部 -->
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

    <!-- 模式和目标选择 -->
    <div class="kana-keyboard-options">
      <div class="kana-mode-select">
        <label>
          <UiInput
            type="radio"
            name="kanaMode"
            value="hiragana"
            v-model="kanaMode"
          />
          平假名
        </label>
        <label>
          <UiInput
            type="radio"
            name="kanaMode"
            value="katakana"
            v-model="kanaMode"
          />
          片假名
        </label>
      </div>
      <div class="kana-target-select">
        <label>输入到：</label>
        <CustomSelect
          v-model="targetField"
          :options="targetFieldOptions"
        />
      </div>
    </div>

    <!-- 基本50音 -->
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

    <!-- 浊音/半浊音 -->
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

    <!-- 拗音 -->
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

    <!-- 特殊字符 -->
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

    <!-- 底部工具栏 -->
    <div class="kana-keyboard-footer">
      <UiButton variant="toolbar" class="kana-backspace" @click="deleteChar">⌫ 退格</UiButton>
    </div>
  </div>
</template>

<script setup lang="ts">

import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
/**
 * 50音软键盘组件
 * 提供日语假名输入功能
 */
import { ref } from 'vue'
import CustomSelect from '@/components/common/CustomSelect.vue'

// ============================================================
// 类型定义
// ============================================================

interface KanaChar {
  h: string  // 平假名
  k: string  // 片假名
}

interface KanaRow {
  label: string
  chars: (KanaChar | null)[]
}

interface SpecialChar {
  char: string
  label?: string
}

// ============================================================
// Props 和 Emits
// ============================================================

const props = withDefaults(defineProps<{
  /** 是否显示 */
  visible?: boolean
  /** 默认目标字段 */
  defaultTarget?: 'original' | 'translated'
}>(), {
  visible: false,
  defaultTarget: 'original'
})

const emit = defineEmits<{
  /** 关闭键盘 */
  (e: 'close'): void
  /** 插入字符 */
  (e: 'insert', char: string, target: 'original' | 'translated'): void
  /** 删除字符 */
  (e: 'delete', target: 'original' | 'translated'): void
}>()

// ============================================================
// 状态
// ============================================================

/** 当前激活的标签页 */
const activeTab = ref<'basic' | 'dakuten' | 'combo' | 'special'>('basic')

/** 假名模式：平假名或片假名 */
const kanaMode = ref<'hiragana' | 'katakana'>('hiragana')

/** 目标字段 */
const targetField = ref<'original' | 'translated'>(props.defaultTarget)

/** 当前按下的键（用于视觉反馈） */
const pressedKey = ref<string | null>(null)

/** 目标字段选项（用于CustomSelect） */
const targetFieldOptions = [
  { label: '原文', value: 'original' },
  { label: '译文', value: 'translated' }
]

// ============================================================
// 标签页配置
// ============================================================

const tabs = [
  { id: 'basic' as const, label: '基本' },
  { id: 'dakuten' as const, label: '浊/半浊音' },
  { id: 'combo' as const, label: '拗音' },
  { id: 'special' as const, label: '特殊' }
]

// ============================================================
// 假名数据
// ============================================================

/** 基本50音 */
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

/** 浊音/半浊音 */
const dakutenKana: KanaRow[] = [
  { label: 'が行', chars: [{ h: 'が', k: 'ガ' }, { h: 'ぎ', k: 'ギ' }, { h: 'ぐ', k: 'グ' }, { h: 'げ', k: 'ゲ' }, { h: 'ご', k: 'ゴ' }] },
  { label: 'ざ行', chars: [{ h: 'ざ', k: 'ザ' }, { h: 'じ', k: 'ジ' }, { h: 'ず', k: 'ズ' }, { h: 'ぜ', k: 'ゼ' }, { h: 'ぞ', k: 'ゾ' }] },
  { label: 'だ行', chars: [{ h: 'だ', k: 'ダ' }, { h: 'ぢ', k: 'ヂ' }, { h: 'づ', k: 'ヅ' }, { h: 'で', k: 'デ' }, { h: 'ど', k: 'ド' }] },
  { label: 'ば行', chars: [{ h: 'ば', k: 'バ' }, { h: 'び', k: 'ビ' }, { h: 'ぶ', k: 'ブ' }, { h: 'べ', k: 'ベ' }, { h: 'ぼ', k: 'ボ' }] },
  { label: 'ぱ行', chars: [{ h: 'ぱ', k: 'パ' }, { h: 'ぴ', k: 'ピ' }, { h: 'ぷ', k: 'プ' }, { h: 'ぺ', k: 'ペ' }, { h: 'ぽ', k: 'ポ' }] }
]

/** 拗音 */
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

/** 特殊字符 */
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

// ============================================================
// 方法
// ============================================================

/**
 * 关闭键盘
 */
function close(): void {
  emit('close')
}

/**
 * 插入假名字符
 */
function insertKana(kana: KanaChar): void {
  const char = kanaMode.value === 'hiragana' ? kana.h : kana.k
  
  // 视觉反馈
  pressedKey.value = kana.h
  setTimeout(() => {
    pressedKey.value = null
  }, 100)
  
  emit('insert', char, targetField.value)
}

/**
 * 插入特殊字符
 */
function insertSpecialChar(char: string): void {
  // 视觉反馈
  pressedKey.value = char
  setTimeout(() => {
    pressedKey.value = null
  }, 100)
  
  emit('insert', char, targetField.value)
}

/**
 * 删除字符
 */
function deleteChar(): void {
  emit('delete', targetField.value)
}
</script>

<style scoped>
/* 50音键盘容器 - 使用固定颜色值 */
.kana-keyboard {
  /* owner tokens: japanese-keyboard */
  --japanese-keyboard-border-default: #2196f3;
  --japanese-keyboard-border-strong: rgba(231, 76, 60, .3);
  --japanese-keyboard-border-muted: rgba(231, 76, 60, .5);
  --japanese-keyboard-shadow-default: rgba(0, 0, 0, .15);
  --japanese-keyboard-shadow-raised: rgba(33, 150, 243, .3);
  --japanese-keyboard-surface-base: #ff6b6b;
  --japanese-keyboard-surface-raised: #ee5a5a;
  --japanese-keyboard-surface-muted: rgba(255, 255, 255, .2);
  --japanese-keyboard-surface-subtle: rgba(255, 255, 255, .3);
  --japanese-keyboard-surface-hover: rgba(255, 255, 255, .4);
  --japanese-keyboard-surface-active: #e74c3c;
  --japanese-keyboard-surface-selected: #f8f9fa;
  --japanese-keyboard-surface-overlay: #e3f2fd;
  --japanese-keyboard-surface-inverse: #bbdefb;
  --japanese-keyboard-surface-contrast: rgba(231, 76, 60, .1);
  --japanese-keyboard-surface-tint: rgba(231, 76, 60, .2);

  background: var(--color-surface-base);
  border: 1px solid var(--color-border-default);
  border-radius: 8px;
  box-shadow: 0 4px 12px var(--japanese-keyboard-shadow-default);
  margin-top: 10px;
  overflow: hidden;
  color: var(--color-text-default);
}

/* 键盘头部 - 红色渐变背景 */
.kana-keyboard-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: linear-gradient(135deg, var(--japanese-keyboard-surface-base) 0%, var(--japanese-keyboard-surface-raised) 100%);
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
  background: var(--japanese-keyboard-surface-muted);
  color: var(--color-text-inverse);
  font-size: 11px;
  cursor: pointer;
  transition: all 0.2s;
}

.kana-tab:hover {
  background: var(--japanese-keyboard-surface-subtle);
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
  background: var(--japanese-keyboard-surface-muted);
  color: var(--color-text-inverse);
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.kana-keyboard-close:hover {
  background: var(--japanese-keyboard-surface-hover);
}

/* 选项区域 */
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

.kana-mode-select label {
  display: flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
  color: var(--color-text-default);
}

.kana-mode-select input[type="radio"] {
  accent-color: var(--japanese-keyboard-surface-active);
}

.kana-target-select select {
  padding: 4px 8px;
  background: var(--color-surface-base);
  border: 1px solid var(--color-border-subtle);
  border-radius: 4px;
  color: var(--color-text-default);
  font-size: 12px;
}

/* 标签页内容区域 */
.kana-tab-content {
  padding: 10px;
  max-height: 280px;
  overflow-y: auto;
  background: var(--color-surface-base);
}

/* 假名表格 */
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

/* 假名按键 - 浅色主题 */
.kana-key {
  width: 42px;
  height: 42px;
  border: 1px solid var(--color-border-subtle);
  border-radius: 6px;
  background: var(--japanese-keyboard-surface-selected);
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

/* 假名文字样式 */
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
  background: var(--japanese-keyboard-surface-overlay);
  border-color: var(--japanese-keyboard-border-default);
  transform: translateY(-1px);
  box-shadow: 0 2px 6px var(--japanese-keyboard-shadow-raised);
}

.kana-key:active,
.kana-key.pressed {
  transform: translateY(0);
  background: var(--japanese-keyboard-surface-inverse);
}

/* 特殊字符网格 */
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

/* 拗音表格 */
.combo-table .kana-key {
  width: 56px;
}

/* 底部工具栏 */
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
  background: var(--japanese-keyboard-surface-contrast);
  border: 1px solid var(--japanese-keyboard-border-strong);
  border-radius: 4px;
  color: var(--color-text-danger-strong);
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.kana-backspace:hover {
  background: var(--japanese-keyboard-surface-tint);
  border-color: var(--japanese-keyboard-border-muted);
}
</style>
