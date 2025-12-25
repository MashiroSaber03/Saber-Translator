<!--
  50音软键盘组件
  提供日语假名输入功能
  - 支持平假名和片假名切换
  - 支持基本、浊音、拗音、特殊字符标签页
  - 支持向指定文本框插入字符
  对应原 edit_mode.js 中的 initKanaKeyboard 功能
-->
<template>
  <div v-if="visible" class="kana-keyboard">
    <!-- 键盘头部 -->
    <div class="kana-keyboard-header">
      <span class="kana-keyboard-title">50音键盘</span>
      <div class="kana-keyboard-tabs">
        <button
          v-for="tab in tabs"
          :key="tab.id"
          class="kana-tab"
          :class="{ active: activeTab === tab.id }"
          @click="activeTab = tab.id"
        >
          {{ tab.label }}
        </button>
      </div>
      <button class="kana-keyboard-close" @click="close">✕</button>
    </div>

    <!-- 模式和目标选择 -->
    <div class="kana-keyboard-options">
      <div class="kana-mode-select">
        <label>
          <input
            type="radio"
            name="kanaMode"
            value="hiragana"
            v-model="kanaMode"
          />
          平假名
        </label>
        <label>
          <input
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
              <button
                v-if="kana"
                class="kana-key"
                :class="{ pressed: pressedKey === kana.h }"
                @click="insertKana(kana)"
              >
                <span class="kana-hiragana">{{ kana.h }}</span>
                <span class="kana-katakana">{{ kana.k }}</span>
              </button>
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
              <button
                v-if="kana"
                class="kana-key"
                :class="{ pressed: pressedKey === kana.h }"
                @click="insertKana(kana)"
              >
                <span class="kana-hiragana">{{ kana.h }}</span>
                <span class="kana-katakana">{{ kana.k }}</span>
              </button>
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
              <button
                v-if="kana"
                class="kana-key"
                :class="{ pressed: pressedKey === kana.h }"
                @click="insertKana(kana)"
              >
                <span class="kana-hiragana">{{ kana.h }}</span>
                <span class="kana-katakana">{{ kana.k }}</span>
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- 特殊字符 -->
    <div class="kana-tab-content" :class="{ active: activeTab === 'special' }">
      <div class="special-chars-grid">
        <button
          v-for="char in specialChars"
          :key="char.char"
          class="kana-key special-key"
          :class="{ pressed: pressedKey === char.char }"
          @click="insertSpecialChar(char.char)"
        >
          {{ char.char }}
          <span v-if="char.label" class="char-label">{{ char.label }}</span>
        </button>
      </div>
    </div>

    <!-- 底部工具栏 -->
    <div class="kana-keyboard-footer">
      <button class="kana-backspace" @click="deleteChar">⌫ 退格</button>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 50音软键盘组件
 * 提供日语假名输入功能
 */
import { ref, computed } from 'vue'
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
  background: #ffffff !important;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  margin-top: 10px;
  overflow: hidden;
  color: #333333 !important;
}

/* 键盘头部 - 红色渐变背景 */
.kana-keyboard-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
  color: #fff;
}

.kana-keyboard-title {
  font-weight: 600;
  font-size: 13px;
  color: #fff;
}

.kana-keyboard-tabs {
  display: flex;
  gap: 4px;
}

.kana-tab {
  padding: 4px 10px;
  border: none;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.2);
  color: #fff;
  font-size: 11px;
  cursor: pointer;
  transition: all 0.2s;
}

.kana-tab:hover {
  background: rgba(255, 255, 255, 0.3);
}

.kana-tab.active {
  background: #fff;
  color: #e74c3c;
  font-weight: 600;
}

.kana-keyboard-close {
  width: 24px;
  height: 24px;
  border: none;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.2);
  color: #fff;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.kana-keyboard-close:hover {
  background: rgba(255, 255, 255, 0.4);
}

/* 选项区域 */
.kana-keyboard-options {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 8px 12px;
  background: #f5f5f5;
  border-bottom: 1px solid #e0e0e0;
}

.kana-mode-select,
.kana-target-select {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #333333 !important;
  font-size: 12px;
}

.kana-mode-select label {
  display: flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
  color: #333333 !important;
}

.kana-mode-select input[type="radio"] {
  accent-color: #e74c3c;
}

.kana-target-select select {
  padding: 4px 8px;
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 4px;
  color: #333333 !important;
  font-size: 12px;
}

/* 标签页内容区域 */
.kana-tab-content {
  padding: 10px;
  max-height: 280px;
  overflow-y: auto;
  background: #ffffff !important;
}

/* 假名表格 */
.kana-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  background: #ffffff !important;
}

.kana-table th,
.kana-table td {
  padding: 2px;
  text-align: center;
  vertical-align: middle;
  background: #ffffff !important;
}

.kana-table th {
  color: #666666 !important;
  font-weight: 500;
  font-size: 11px;
  padding-bottom: 6px;
}

.row-label {
  color: #888888 !important;
  font-size: 11px;
  font-weight: 500;
  padding-right: 8px !important;
  text-align: right !important;
  white-space: nowrap;
}

/* 假名按键 - 浅色主题 */
.kana-key {
  width: 42px;
  height: 42px;
  border: 1px solid #dddddd !important;
  border-radius: 6px;
  background: #f8f9fa !important;
  color: #333333 !important;
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
  color: #333333 !important;
  font-size: 13px;
  display: block;
}

.kana-katakana {
  color: #666666 !important;
  font-size: 11px;
  display: block;
}

.kana-key:hover {
  background: #e3f2fd !important;
  border-color: #2196f3 !important;
  transform: translateY(-1px);
  box-shadow: 0 2px 6px rgba(33, 150, 243, 0.3);
}

.kana-key:active,
.kana-key.pressed {
  transform: translateY(0);
  background: #bbdefb !important;
}

/* 特殊字符网格 */
.special-chars-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(60px, 1fr));
  gap: 6px;
  background: #ffffff !important;
}

.special-key {
  width: auto;
  height: auto;
  min-height: 32px;
  padding: 4px 8px;
  font-size: 14px;
  color: #333333 !important;
}

.char-label {
  font-size: 9px;
  color: #666666 !important;
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
  background: #f5f5f5;
  border-top: 1px solid #e0e0e0;
}

.kana-backspace {
  padding: 6px 16px;
  background: rgba(231, 76, 60, 0.1);
  border: 1px solid rgba(231, 76, 60, 0.3);
  border-radius: 4px;
  color: #e74c3c !important;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.kana-backspace:hover {
  background: rgba(231, 76, 60, 0.2);
  border-color: rgba(231, 76, 60, 0.5);
}
</style>
