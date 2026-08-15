<template>
  <ContinuationDialogShell title="编辑形态" :dismissible="!busy" @close="close">
    <ContinuationDialogForm>
      <ContinuationDialogField
        label="形态名称"
        control-id="continuationEditFormName"
        required
        :error="formNameError"
      >
        <UiInput
          id="continuationEditFormName"
          v-model="localFormName"
          type="text"
          aria-label="形态名称"
          class="continuation-dialog__form-input"
          :error="Boolean(formNameError)"
          placeholder="形态显示名"
        />
      </ContinuationDialogField>

      <ContinuationDialogField label="形态描述" control-id="continuationEditFormDescription">
        <UiTextarea
          id="continuationEditFormDescription"
          v-model="localDescription"
          rows="2"
          variant="panel"
          aria-label="形态描述"
          class="continuation-dialog__form-input"
          placeholder="形态描述..."
        />
      </ContinuationDialogField>
    </ContinuationDialogForm>

    <template #footer>
      <ContinuationDialogActions>
        <UiButton variant="secondary" :disabled="busy" @click="close">取消</UiButton>
        <UiButton
          variant="primary"
          :disabled="!localFormName.trim() || busy"
          @click="save"
        >
          <UiIcon v-if="!busy" name="save" size="15" />
          <span>{{ busy ? '保存中...' : '保存' }}</span>
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
import UiIcon from '@/components/ui/UiIcon.vue'
import ContinuationDialogActions from './ContinuationDialogActions.vue'
import ContinuationDialogField from './ContinuationDialogField.vue'
import ContinuationDialogForm from './ContinuationDialogForm.vue'
import ContinuationDialogShell from './ContinuationDialogShell.vue'

const props = defineProps<{
  form: CharacterForm
  busy?: boolean
}>()

const emit = defineEmits<{
  close: []
  save: [formName: string, description: string]
}>()

const localFormName = ref(props.form.form_name)
const localDescription = ref(props.form.description)
const formNameError = ref('')
const close = () => emit('close')

watch(() => props.form, (newForm) => {
  localFormName.value = newForm.form_name
  localDescription.value = newForm.description
}, { immediate: true })

function save() {
  const formName = localFormName.value.trim()
  if (!formName) {
    formNameError.value = '请输入形态名称'
    return
  }
  formNameError.value = ''
  emit('save', formName, localDescription.value.trim())
}
</script>
