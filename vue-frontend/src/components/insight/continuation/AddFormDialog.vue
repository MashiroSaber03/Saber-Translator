<template>
  <div class="modal-overlay" @click.self="$emit('close')">
    <div class="modal-dialog add-form-dialog">
      <div class="modal-header">
        <h3>➕ 新增形态</h3>
        <button class="close-btn" @click="$emit('close')">×</button>
      </div>
      
      <div class="modal-body">        
        <div class="form-group">
          <label>形态名称 <span class="required">*</span></label>
          <input 
            v-model="formName" 
            type="text" 
            class="form-input"
            placeholder="例如: 战斗服、黑化形态、常服"
          >
        </div>
        
        <div class="form-group">
          <label>形态描述（可选）</label>
          <textarea 
            v-model="description"
            rows="2"
            class="form-input"
            placeholder="简单描述该形态的特征..."
          ></textarea>
        </div>
      </div>
      
      <div class="modal-footer">
        <button class="btn secondary" @click="$emit('close')">取消</button>
        <button 
          class="btn primary" 
          :disabled="!formName.trim() || isAdding"
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
  'add': [formName: string, description: string]
}>()

const formName = ref('')
const description = ref('')
const isAdding = ref(false)

function add() {
  const name = formName.value.trim()
  
  if (!name) {
    alert('请填写形态名称')
    return
  }
  
  isAdding.value = true
  emit('add', name, description.value.trim())
  
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

.add-form-dialog {
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
