<template>
  <div class="script-panel">
    <h3>📝 生成续写脚本</h3>
    
    <div class="script-editor" v-if="script">
      <div class="script-header">
        <h4>{{ script.chapter_title }}</h4>
        <span class="script-meta">共 {{ script.page_count }} 页 · {{ script.generated_at }}</span>
      </div>
      
      <textarea 
        v-model="scriptText"
        class="script-textarea"
        rows="15"
        placeholder="脚本将在此显示..."
      ></textarea>
      
      <div class="script-actions">
        <button class="btn secondary small" @click="scriptText = script!.script_text">↺ 重置</button>
      </div>
    </div>
    
    <div v-else class="no-script">
      <p>点击下方按钮生成续写脚本</p>
    </div>
    
    <button 
      class="btn primary"
      :disabled="isGenerating"
      @click="$emit('generate')"
    >
      {{ isGenerating ? '生成中...' : '🎯 生成脚本' }}
    </button>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import type { ChapterScript } from '@/api/continuation'

const props = defineProps<{
  script: ChapterScript | null
  isGenerating: boolean
}>()

defineEmits<{
  'generate': []
}>()

const scriptText = ref('')

watch(() => props.script, (newScript) => {
  if (newScript) {
    scriptText.value = newScript.script_text
  }
}, { immediate: true })
</script>

<style scoped>
.script-panel {
  padding: 24px;
}

.script-panel h3 {
  margin: 0 0 20px 0;
  font-size: 18px;
  font-weight: 600;
}

.script-editor {
  margin-bottom: 20px;
}

.script-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.script-header h4 {
  margin: 0;
  font-size: 16px;
}

.script-meta {
  font-size: 13px;
  color: var(--text-secondary, #666);
}

.script-textarea {
  width: 100%;
  padding: 16px;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 8px;
  font-family: inherit;
  font-size: 14px;
  line-height: 1.6;
  resize: vertical;
}

.script-textarea:focus {
  outline: none;
  border-color: var(--primary, #6366f1);
}

.script-actions {
  margin-top: 12px;
}

.no-script {
  text-align: center;
  padding: 40px 20px;
  color: var(--text-secondary, #666);
}

.no-script p {
  margin: 0;
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  width: 100%;
}

.btn.primary {
  background: var(--primary, #6366f1);
  color: white;
}

.btn.primary:hover:not(:disabled) {
  background: var(--primary-dark, #4f46e5);
}

.btn.primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn.secondary {
  background: var(--bg-secondary, #f3f4f6);
  color: var(--text-primary, #333);
  border: 1px solid var(--border-color, #e0e0e0);
}

.btn.secondary:hover {
  background: var(--bg-hover, #e5e7eb);
}

.btn.small {
  padding: 6px 12px;
  font-size: 13px;
  width: auto;
}
</style>
