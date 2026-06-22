<template>
  <ContinuationDialogShell title="✏️ 编辑形态" @close="close">
    <ContinuationDialogForm>
      <ContinuationDialogField label="形态名称">
        <UiInput
          v-model="localFormName"
          type="text"
          class="continuation-dialog__form-input"
          style="font: inherit"
          placeholder="形态显示名"
        />
      </ContinuationDialogField>

      <ContinuationDialogField label="形态描述">
        <UiTextarea
          v-model="localDescription"
          rows="2"
          class="continuation-dialog__form-input"
          style="font: inherit"
          placeholder="形态描述..."
        />
      </ContinuationDialogField>
    </ContinuationDialogForm>

    <template #footer>
      <ContinuationDialogActions>
        <UiButton variant="secondary" @click="close">取消</UiButton>
        <UiButton
          variant="primary"
          :disabled="isSaving"
          @click="save"
        >
          {{ isSaving ? '保存中...' : '💾 保存' }}
        </UiButton>
      </ContinuationDialogActions>
    </template>
  </ContinuationDialogShell>
</template>

<script setup lang="ts">
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import { ref, watch } from 'vue'
import type { CharacterForm } from '@/api/continuation'
import UiButton from '@/components/ui/UiButton.vue'
import ContinuationDialogActions from './ContinuationDialogActions.vue'
import ContinuationDialogField from './ContinuationDialogField.vue'
import ContinuationDialogForm from './ContinuationDialogForm.vue'
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
