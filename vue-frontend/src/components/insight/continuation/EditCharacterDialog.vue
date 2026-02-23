<template>
  <div class="modal-overlay" @click.self="$emit('close')">
    <div class="modal-dialog edit-char-dialog">
      <div class="modal-header">
        <h3>✏️ 编辑角色</h3>
        <button class="close-btn" @click="$emit('close')">×</button>
      </div>
      
      <div class="modal-body">
        <div class="form-group">
          <label>角色名称</label>
          <input 
            v-model="localName" 
            type="text" 
            class="form-input"
            placeholder="输入角色主名称"
          >
        </div>
        
        <div class="form-group">
          <label>别名（用逗号分隔）</label>
          <input 
            v-model="localAliases" 
            type="text" 
            class="form-input"
            placeholder="例如: 桐乃, 新垣彩世"
          >
          <p class="form-hint">AI生成脚本时可能使用这些名字引用角色</p>
        </div>
      </div>
      
      <div class="modal-footer">
        <button class="btn secondary" @click="$emit('close')">取消</button>
        <button class="btn primary" @click="save">💾 保存</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import type { CharacterProfile } from '@/api/continuation'

const props = defineProps<{
  character: CharacterProfile
}>()

const emit = defineEmits<{
  'close': []
  'save': [name: string, aliases: string[]]
}>()

const localName = ref(props.character.name)
const localAliases = ref(props.character.aliases.join(', '))

// 监听character变化，更新本地值
watch(() => props.character, (newChar) => {
  localName.value = newChar.name
  localAliases.value = newChar.aliases.join(', ')
}, { immediate: true })

function save() {
  const name = localName.value.trim()
  const aliases = localAliases.value
    .split(/[,，]/)
    .map(a => a.trim())
    .filter(a => a.length > 0)
  
  if (!name) {
    alert('角色名不能为空')
    return
  }
  
  emit('save', name, aliases)
}
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgb(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: var(--z-overlay);
  animation: fadeIn 0.2s;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px;
  border-bottom: 1px solid var(--border-color, #e0e0e0);
}

.modal-header h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
}

.close-btn {
  background: none;
  border: none;
  font-size: 28px;
  line-height: 1;
  cursor: pointer;
  color: var(--text-secondary, #666);
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
  transition: background 0.2s;
}

.close-btn:hover {
  background: var(--bg-secondary, #f5f5f5);
}

.modal-body {
  padding: 24px;
}

.form-group {
  margin-bottom: 16px;
}

.form-group:last-child {
  margin-bottom: 0;
}

.form-group label {
  display: block;
  font-weight: 600;
  margin-bottom: 6px;
  font-size: 14px;
}

.form-input {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 8px;
  font-size: 14px;
  transition: border-color 0.2s;
}

.form-input:focus {
  outline: none;
  border-color: var(--primary, #6366f1);
}

.form-hint {
  font-size: 12px;
  color: var(--text-secondary, #888);
  margin-top: 6px;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  padding: 16px 24px;
  border-top: 1px solid var(--border-color, #e0e0e0);
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.btn.primary {
  background: var(--primary, #6366f1);
  color: white;
}

.btn.primary:hover {
  background: var(--primary-dark, #4f46e5);
}

.btn.secondary {
  background: var(--bg-secondary, #f3f4f6);
  color: var(--text-primary, #333);
  border: 1px solid var(--border-color, #e0e0e0);
}

.btn.secondary:hover {
  background: var(--bg-hover, #e5e7eb);
}
</style>
