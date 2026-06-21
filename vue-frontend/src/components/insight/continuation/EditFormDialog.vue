<template>
  <ContinuationDialogShell title="✏️ 编辑形态" @close="close">
    <div class="continuation-dialog-form">
      <div class="continuation-dialog-form__field">
        <label>形态名称</label>
        <UiInput
          v-model="localFormName"
          type="text"
          class="continuation-dialog__form-input"
          placeholder="形态显示名"
        />
      </div>

      <div class="continuation-dialog-form__field">
        <label>形态描述</label>
        <UiTextarea
          v-model="localDescription"
          rows="2"
          class="continuation-dialog__form-input"
          placeholder="形态描述..."
        />
      </div>
    </div>

    <template #footer>
      <div class="continuation-dialog-actions">
        <UiButton variant="secondary" @click="close">取消</UiButton>
        <UiButton
          variant="primary"
          :disabled="isSaving"
          @click="save"
        >
          {{ isSaving ? '保存中...' : '💾 保存' }}
        </UiButton>
      </div>
    </template>
  </ContinuationDialogShell>
</template>

<script setup lang="ts">
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import { ref, watch } from 'vue'
import type { CharacterForm } from '@/api/continuation'
import UiButton from '@/components/ui/UiButton.vue'
import ContinuationDialogShell from './ContinuationDialogShell.vue'

const props = defineProps<{
  form: CharacterForm
}>()

const emit = defineEmits<{
  close: []
  save: [formName: string, description: string]
}>()

const localFormName = ref(props.form.form_name)
const localDescription = ref(props.form.description)
const isSaving = ref(false)
const close = () => emit('close')

watch(() => props.form, (newForm) => {
  localFormName.value = newForm.form_name
  localDescription.value = newForm.description
}, { immediate: true })

function save() {
  isSaving.value = true
  emit('save', localFormName.value.trim(), localDescription.value.trim())

  setTimeout(() => {
    isSaving.value = false
  }, 500)
}
</script>
