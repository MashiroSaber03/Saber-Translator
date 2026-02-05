<template>
  <div class="form-tile" :class="{ disabled: form.enabled === false }">
    <!-- 图片区域 -->
    <div class="form-image-section">
      <img v-if="form.reference_image" :src="formImageUrl" alt="">
      <div v-else class="image-placeholder">
        <span class="placeholder-icon">📷</span>
        <p class="placeholder-text">未上传参考图</p>
      </div>
      <label class="upload-overlay">
        <span class="upload-text">{{ form.reference_image ? '更换图片' : '上传图片' }}</span>
        <input type="file" accept="image/*" hidden @change="handleUpload">
      </label>
    </div>
    
    <!-- 信息区域 -->
    <div class="form-content">
      <div class="form-header">
        <h4 class="form-title">{{ form.form_name }}</h4>
        <span v-if="form.enabled === false" class="status-badge disabled">已禁用</span>
      </div>
      <p v-if="form.description" class="form-description">{{ form.description }}</p>
    </div>
    
    <!-- 操作区域 -->
    <div class="form-actions">
      <div class="action-row">
        <label class="toggle-control" :title="form.enabled !== false ? '点击禁用' : '点击启用'">
          <input 
            type="checkbox" 
            :checked="form.enabled !== false"
            @change="$emit('toggle-enabled', ($event.target as HTMLInputElement).checked)"
          >
          <span class="toggle-track"></span>
        </label>
        <button class="action-btn generate-btn" @click="$emit('generate-orthographic')" title="生成三视图">
          <span>🎨</span>
        </button>
        <button v-if="form.reference_image" class="action-btn delete-btn" @click="$emit('delete-image')" title="删除图片">
          <span>🗑️</span>
        </button>
      </div>
      <div class="action-row secondary">
        <button class="icon-btn edit-btn" @click="$emit('edit')" title="编辑形态">✏️</button>
        <button class="icon-btn delete-btn" @click="$emit('delete')" title="删除形态">🗑️</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { CharacterForm } from '@/api/continuation'

const props = defineProps<{
  form: CharacterForm
  characterName: string
  formImageUrl: string
}>()

const emit = defineEmits<{
  'edit': []
  'delete': []
  'upload-image': [file: File]
  'delete-image': []
  'generate-orthographic': []
  'toggle-enabled': [enabled: boolean]
}>()

function handleUpload(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files?.length) return
  
  const file = input.files[0]
  if (!file) return
  
  emit('upload-image', file)
  input.value = ''
}
</script>

<style scoped>
/* 卡片容器 */
.form-tile {
  background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
  border-radius: 16px;
  overflow: hidden;
  border: 1.5px solid #e8eaf6;
  box-shadow: 0 2px 8px rgba(99, 102, 241, 0.08);
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  display: flex;
  flex-direction: column;
}

.form-tile:hover {
  border-color: #c7d2fe;
  box-shadow: 0 8px 24px rgba(99, 102, 241, 0.16);
  transform: translateY(-2px);
}

.form-tile.disabled {
  opacity: 0.6;
  filter: grayscale(60%);
}

.form-tile.disabled:hover {
  transform: none;
}

/* 图片区域 */
.form-image-section {
  aspect-ratio: 1;
  position: relative;
  background: linear-gradient(135deg, #f5f7ff 0%, #eef2ff 100%);
  overflow: hidden;
}

.form-image-section img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.image-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #9ca3af;
}

.placeholder-icon {
  font-size: 48px;
  margin-bottom: 8px;
  opacity: 0.5;
}

.placeholder-text {
  margin: 0;
  font-size: 12px;
  font-weight: 500;
  color: #9ca3af;
}

/* 上传遮罩 */
.upload-overlay {
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.92), rgba(124, 58, 237, 0.92));
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity 0.25s ease;
  cursor: pointer;
}

.upload-text {
  color: white;
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0.3px;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}

.form-image-section:hover .upload-overlay {
  opacity: 1;
}

/* 内容区域 */
.form-content {
  padding: 14px 12px 12px 12px;
  flex: 1;
  display: flex;
  flex-direction: column;
}

.form-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 6px;
}

.form-title {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  color: #1e293b;
  flex: 1;
  line-height: 1.3;
}

.status-badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.3px;
}

.status-badge.disabled {
  background: linear-gradient(135deg, #fee2e2, #fecaca);
  color: #dc2626;
}

.form-description {
  margin: 0;
  font-size: 11px;
  color: #64748b;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* 操作区域 */
.form-actions {
  padding: 10px 12px;
  background: linear-gradient(to bottom, #fafbff, #f8f9ff);
  border-top: 1px solid #e8eaf6;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.action-row {
  display: flex;
  gap: 6px;
  align-items: center;
}

.action-row.secondary {
  padding-top: 2px;
}

/* Toggle开关 */
.toggle-control {
  position: relative;
  display: inline-block;
  width: 32px;
  height: 18px;
  cursor: pointer;
  flex-shrink: 0;
}

.toggle-control input {
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle-track {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(135deg, #cbd5e1, #94a3b8);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border-radius: 18px;
}

.toggle-track:before {
  position: absolute;
  content: "";
  height: 14px;
  width: 14px;
  left: 2px;
  bottom: 2px;
  background: white;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border-radius: 50%;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.toggle-control input:checked + .toggle-track {
  background: linear-gradient(135deg, #10b981, #059669);
}

.toggle-control input:checked + .toggle-track:before {
  transform: translateX(14px);
}

/* 图标按钮 */
.action-btn {
  flex: 1;
  height: 32px;
  padding: 0 10px;
  border: 1.5px solid #e2e8f0;
  background: white;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.action-btn:hover {
  background: #f8fafc;
  border-color: #cbd5e1;
}

.action-btn.generate-btn {
  border-color: #a5b4fc;
  color: #6366f1;
}

.action-btn.generate-btn:hover {
  background: #eef2ff;
  border-color: #818cf8;
}

.action-btn.delete-btn:hover {
  background: #fef2f2;
  border-color: #fecaca;
}

/* 图标按钮（次要行） */
.icon-btn {
  width: 32px;
  height: 32px;
  padding: 0;
  border: 1.5px solid #e2e8f0;
  background: white;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.icon-btn:hover {
  background: #f8fafc;
  border-color: #cbd5e1;
}

.icon-btn.edit-btn:hover {
  background: #eef2ff;
  border-color: #a5b4fc;
}

.icon-btn.delete-btn:hover {
  background: #fef2f2;
  border-color: #fecaca;
}
</style>
