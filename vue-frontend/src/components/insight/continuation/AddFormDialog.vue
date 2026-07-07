<template>
  <ContinuationDialogShell title="新增形态" @close="close">
    <ContinuationDialogForm>
      <ContinuationDialogField
        label="形态名称"
        control-id="continuationAddFormName"
        required
        :error="formNameError"
      >
        <UiInput
          id="continuationAddFormName"
          v-model="formName"
          type="text"
          aria-label="形态名称"
          class="continuation-dialog__form-input"
          :error="Boolean(formNameError)"
          placeholder="例如: 战斗服、黑化形态、常服"
        />
      </ContinuationDialogField>

      <ContinuationDialogField label="形态描述（可选）" control-id="continuationAddFormDescription">
        <UiTextarea
          id="continuationAddFormDescription"
          v-model="description"
          rows="2"
          variant="panel"
          aria-label="形态描述（可选）"
          class="continuation-dialog__form-input"
          placeholder="简单描述该形态的特征..."
        />
      </ContinuationDialogField>
    </ContinuationDialogForm>

    <template #footer>
      <ContinuationDialogActions>
        <UiButton variant="secondary" @click="close">取消</UiButton>
        <UiButton
          variant="primary"
          :disabled="!formName.trim() || isAdding"
          @click="add"
        >
          <UiIcon v-if="!isAdding" name="check" size="15" />
          <span>{{ isAdding ? '添加中...' : '确认添加' }}</span>
        </UiButton>
      </ContinuationDialogActions>
    </template>
  </ContinuationDialogShell>
</template>

<script setup lang="ts">
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import { onBeforeUnmount, ref } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import ContinuationDialogActions from './ContinuationDialogActions.vue'
import ContinuationDialogField from './ContinuationDialogField.vue'
import ContinuationDialogForm from './ContinuationDialogForm.vue'
import ContinuationDialogShell from './ContinuationDialogShell.vue'

const emit = defineEmits<{
  close: []
  add: [formName: string, description: string]
}>()

const formName = ref('')
const description = ref('')
const formNameError = ref('')
const isAdding = ref(false)
let loadingTimer: ReturnType<typeof setTimeout> | null = null
const close = () => emit('close')

function clearLoadingTimer(): void {
  if (loadingTimer) {
    clearTimeout(loadingTimer)
    loadingTimer = null
  }
}

function add() {
  const name = formName.value.trim()

  if (!name) {
    formNameError.value = '请输入形态名称'
    return
  }
  formNameError.value = ''

  isAdding.value = true
  emit('add', name, description.value.trim())

  clearLoadingTimer()
  loadingTimer = setTimeout(() => {
    isAdding.value = false
    loadingTimer = null
  }, 500)
}

onBeforeUnmount(() => {
  clearLoadingTimer()
})
</script>
