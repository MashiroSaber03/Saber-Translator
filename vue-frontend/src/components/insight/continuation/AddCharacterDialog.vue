<template>
  <div class="modal-overlay" @click.self="$emit('close')">
    <div class="modal-dialog add-char-dialog">
      <div class="modal-header">
        <h3>➕ 新增角色</h3>
        <button class="close-btn" @click="$emit('close')">×</button>
      </div>
      
      <div class="modal-body">
        <div class="form-group">
          <label>角色名称 <span class="required">*</span></label>
          <input 
            v-model="name" 
            type="text" 
            class="form-input"
            placeholder="输入角色名称"
          >
        </div>
        
        <div class="form-group">
          <label>别名（用逗号分隔，可选）</label>
          <input 
            v-model="aliases" 
            type="text" 
            class="form-input"
            placeholder="例如: 小明, 阿明"
          >
        </div>
        
        <div class="form-group">
          <label>角色描述（可选）</label>
          <textarea 
            v-model="description"
            rows="3"
            class="form-input"
            placeholder="简单描述角色的外观特征..."
          ></textarea>
        </div>
      </div>
      
      <div class="modal-footer">
        <button class="btn secondary" @click="$emit('close')">取消</button>
        <button 
          class="btn primary" 
          :disabled="!name.trim() || isAdding"
          @click="add"
        >
          {{ isAdding ? '添加中...' : '✓ 确认添加' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const emit = defineEmits<{
  'close': []
  'add': [name: string, aliases: string[], description: string]
}>()

const name = ref('')
const aliases = ref('')
const description = ref('')
const isAdding = ref(false)

function add() {
  const charName = name.value.trim()
  if (!charName) {
    alert('角色名不能为空')
    return
  }
  
  const aliasList = aliases.value
    .split(/[,，]/)
    .map(a => a.trim())
    .filter(a => a.length > 0)
  
  isAdding.value = true
  emit('add', charName, aliasList, description.value.trim())
  
  // 重置状态（由父组件控制关闭）
  setTimeout(() => {
    isAdding.value = false
  }, 500)
}
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  animation: fadeIn 0.2s;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.modal-dialog {
  background: var(--bg-primary, #fff);
  border-radius: 12px;
  width: 90%;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  animation: slideUp 0.3s;
}

@keyframes slideUp {
  from {
    transform: translateY(20px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

.add-char-dialog {
  max-width: 450px;
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

.required {
  color: #dc3545;
}

.form-input {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 8px;
  font-size: 14px;
  transition: border-color 0.2s;
  font-family: inherit;
}

.form-input:focus {
  outline: none;
  border-color: var(--primary, #6366f1);
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
</style>
