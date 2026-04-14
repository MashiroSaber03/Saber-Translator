<template>
  <div class="editor-card">
    <h4>批量编辑</h4>
    <p class="hint">对已勾选的草稿角色应用统一修改。</p>

    <div class="targets">
      <strong>目标角色（{{ targetCharacters.length }}）</strong>
      <span v-if="targetCharacters.length === 0" class="empty">请先在左侧勾选角色</span>
      <span v-else class="chips">
        {{ targetCharacters.join(" / ") }}
      </span>
    </div>

    <div class="grid">
      <label>
        Tags（逗号分隔）
        <input v-model="tagsText" type="text" :disabled="locks['data.tags']">
      </label>
      <label>
        Tags 应用模式
        <select v-model="tagMode" :disabled="locks['data.tags']">
          <option value="append">追加</option>
          <option value="replace">覆盖</option>
        </select>
      </label>
      <label class="full">
        批量 System Prompt
        <textarea v-model="systemPromptText" rows="3" :disabled="locks['data.system_prompt']"></textarea>
      </label>
      <label class="full">
        批量 Post History Instructions
        <textarea
          v-model="postHistoryText"
          rows="2"
          :disabled="locks['data.post_history_instructions']"
        ></textarea>
      </label>
      <label class="full">
        追加 Alternate Greetings（每行一条）
        <textarea
          v-model="alternateGreetingsText"
          rows="3"
          :disabled="locks['data.alternate_greetings']"
        ></textarea>
      </label>
    </div>

    <div class="actions">
      <button class="btn primary" :disabled="targetCharacters.length === 0" @click="applyBatch">
        应用到所选角色
      </button>
      <button class="btn" @click="resetForm">清空输入</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

interface BatchEditPayload {
  characters: string[]
  tag_mode: 'append' | 'replace'
  tags: string[]
  system_prompt: string
  post_history_instructions: string
  alternate_greetings: string[]
}

const props = defineProps<{
  targetCharacters: string[]
  locks: Record<string, boolean>
}>()

const emit = defineEmits<{
  (e: 'apply', payload: BatchEditPayload): void
}>()

const tagMode = ref<'append' | 'replace'>('append')
const tagsText = ref('')
const systemPromptText = ref('')
const postHistoryText = ref('')
const alternateGreetingsText = ref('')

function splitText(value: string, separator: RegExp): string[] {
  return value
    .split(separator)
    .map(item => item.trim())
    .filter(Boolean)
}

function applyBatch() {
  const payload: BatchEditPayload = {
    characters: [...props.targetCharacters],
    tag_mode: tagMode.value,
    tags: splitText(tagsText.value, /[,;]+/),
    system_prompt: systemPromptText.value.trim(),
    post_history_instructions: postHistoryText.value.trim(),
    alternate_greetings: splitText(alternateGreetingsText.value, /\n+/),
  }
  emit('apply', payload)
}

function resetForm() {
  tagMode.value = 'append'
  tagsText.value = ''
  systemPromptText.value = ''
  postHistoryText.value = ''
  alternateGreetingsText.value = ''
}
</script>

<style scoped>
.editor-card {
  background: var(--bg-secondary, #fff);
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 10px;
  padding: 14px;
}

.editor-card h4 {
  margin: 0 0 6px;
  font-size: 14px;
}

.hint {
  margin: 0 0 10px;
  color: var(--text-secondary, #64748b);
  font-size: 12px;
}

.targets {
  margin-bottom: 10px;
  font-size: 12px;
}

.empty {
  color: var(--text-secondary, #64748b);
}

.chips {
  margin-left: 6px;
  color: var(--text-primary, #0f172a);
}

.grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}

label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  color: var(--text-secondary, #64748b);
}

label.full {
  grid-column: 1 / -1;
}

input,
textarea,
select {
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 8px;
  padding: 8px;
  font-size: 13px;
  color: var(--text-primary, #0f172a);
  background: var(--bg-primary, #f8fafc);
}

.actions {
  display: flex;
  gap: 8px;
  margin-top: 10px;
}

.btn {
  border: 1px solid var(--border-color, #e2e8f0);
  background: var(--bg-tertiary, #f1f5f9);
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 12px;
  cursor: pointer;
}

.btn.primary {
  color: #fff;
  background: var(--color-primary, #6366f1);
  border-color: var(--color-primary, #6366f1);
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

@media (max-width: 900px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
</style>
