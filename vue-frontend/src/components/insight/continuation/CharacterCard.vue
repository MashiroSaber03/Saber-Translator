<template>
  <div class="character-card" :class="{ selected, disabled: !character.enabled }">
    <div class="character-preview">
      <img 
        v-if="character.reference_image" 
        :src="imageUrl"
        alt=""
      >
      <div v-else class="no-image-placeholder">
        <span>👤</span>
        <p>无图片</p>
      </div>
      
      <div v-if="character.forms && character.forms.length > 1" class="form-count-badge">
        {{ character.forms.length }} 形态
      </div>
      
      <div v-if="!character.enabled" class="disabled-overlay">
        <span>已禁用</span>
      </div>
    </div>
    
    <div class="character-details">
      <div class="character-header">
        <div class="character-name-row">
          <span class="character-name">{{ character.name }}</span>
          <button class="edit-btn" @click.stop="$emit('edit')">✏️</button>
        </div>
        <span v-if="character.aliases && character.aliases.length > 0" class="character-aliases">
          别名: {{ character.aliases.join(', ') }}
        </span>
      </div>
      
      <div class="character-actions">
        <button 
          class="action-btn toggle"
          :class="{ enabled: character.enabled !== false }"
          @click.stop="$emit('toggle-enabled', !character.enabled)"
        >
          {{ character.enabled !== false ? '✓ 启用' : '禁用' }}
        </button>
        <button class="action-btn delete" @click.stop="$emit('delete')">🗑️</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { CharacterProfile } from '@/api/continuation'

const props = defineProps<{
  character: CharacterProfile
  selected?: boolean
  imageUrl: string
}>()

defineEmits<{
  'click': []
  'toggle-enabled': [enabled: boolean]
  'edit': []
  'delete': []
}>()
</script>

<style scoped>
.character-card {
  background: var(--bg-secondary, #f5f5f5);
  border-radius: 12px;
  overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s;
  cursor: pointer;
  border: 2px solid transparent;
}

.character-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.character-card.selected {
  border-color: var(--primary, #6366f1);
}

.character-card.disabled {
  opacity: 0.6;
}

.character-preview {
  width: 100%;
  height: 180px;
  background: #e0e0e0;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  position: relative;
}

.character-preview img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.no-image-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #999;
  height: 100%;
}

.no-image-placeholder span {
  font-size: 48px;
  margin-bottom: 8px;
}

.no-image-placeholder p {
  margin: 0;
  font-size: 14px;
}

.form-count-badge {
  position: absolute;
  top: 8px;
  right: 8px;
  background: rgba(99, 102, 241, 0.9);
  color: white;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 600;
}

.disabled-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
}

.disabled-overlay span {
  background: rgba(255, 255, 255, 0.9);
  padding: 6px 12px;
  border-radius: 6px;
  font-weight: 600;
  font-size: 12px;
}

.character-details {
  padding: 12px;
}

.character-header {
  margin-bottom: 12px;
}

.character-name-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
}

.character-name {
  font-size: 15px;
  font-weight: 600;
}

.edit-btn {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 14px;
  padding: 2px 4px;
  border-radius: 4px;
  opacity: 0.6;
  transition: opacity 0.2s, background 0.2s;
}

.edit-btn:hover {
  opacity: 1;
  background: rgba(0, 0, 0, 0.05);
}

.character-aliases {
  font-size: 12px;
  color: var(--text-secondary, #666);
  display: block;
}

.character-actions {
  display: flex;
  gap: 8px;
}

.action-btn {
  flex: 1;
  padding: 6px 12px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 6px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s;
  background: var(--bg-primary, #fff);
}

.action-btn.toggle.enabled {
  background: #d1fae5;
  color: #065f46;
  border-color: #a7f3d0;
}

.action-btn.toggle:not(.enabled) {
  background: #fee2e2;
  color: #991b1b;
  border-color: #fecaca;
}

.action-btn.delete:hover {
  background: #fee2e2;
  color: #991b1b;
  border-color: #fecaca;
}
</style>
