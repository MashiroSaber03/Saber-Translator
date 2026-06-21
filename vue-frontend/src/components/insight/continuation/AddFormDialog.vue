<template>
  <ContinuationDialogShell title="➕ 新增形态" @close="close">
    <div class="continuation-dialog-form">
      <div class="continuation-dialog-form__field">
        <label>形态名称 <span class="required">*</span></label>
        <UiInput
          v-model="formName"
          type="text"
          class="continuation-dialog__form-input"
          placeholder="例如: 战斗服、黑化形态、常服"
        />
      </div>

      <div class="continuation-dialog-form__field">
        <label>形态描述（可选）</label>
        <UiTextarea
          v-model="description"
          rows="2"
          class="continuation-dialog__form-input"
          placeholder="简单描述该形态的特征..."
        />
      </div>
    </div>

    <template #footer>
      <div class="continuation-dialog-actions">
        <UiButton variant="secondary" @click="close">取消</UiButton>
        <UiButton
          variant="primary"
          :disabled="!formName.trim() || isAdding"
          @click="add"
        >
          {{ isAdding ? '添加中...' : '✓ 确认添加' }}
        </UiButton>
      </div>
    </template>
  </ContinuationDialogShell>
</template>

<script setup lang="ts">
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import { ref } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import ContinuationDialogShell from './ContinuationDialogShell.vue'

const emit = defineEmits<{
  close: []
  add: [formName: string, description: string]
}>()

const formName = ref('')
const description = ref('')
const isAdding = ref(false)
const close = () => emit('close')

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
