<template>
  <ContinuationDialogShell title="新增角色" @close="close">
    <ContinuationDialogForm>
      <ContinuationDialogField
        label="角色名称"
        control-id="continuationAddCharacterName"
        required
        :error="nameError"
      >
        <UiInput
          id="continuationAddCharacterName"
          v-model="name"
          type="text"
          aria-label="角色名称"
          class="continuation-dialog__form-input"
          :error="Boolean(nameError)"
          placeholder="输入角色名称"
        />
      </ContinuationDialogField>

      <ContinuationDialogField label="别名（用逗号分隔，可选）" control-id="continuationAddCharacterAliases">
        <UiInput
          id="continuationAddCharacterAliases"
          v-model="aliases"
          type="text"
          aria-label="别名（用逗号分隔，可选）"
          class="continuation-dialog__form-input"
          placeholder="例如: 小明, 阿明"
        />
      </ContinuationDialogField>

      <ContinuationDialogField label="角色描述（可选）" control-id="continuationAddCharacterDescription">
        <UiTextarea
          id="continuationAddCharacterDescription"
          v-model="description"
          rows="3"
          variant="panel"
          aria-label="角色描述（可选）"
          class="continuation-dialog__form-input"
          placeholder="简单描述角色的外观特征..."
        />
      </ContinuationDialogField>
    </ContinuationDialogForm>

    <template #footer>
      <ContinuationDialogActions>
        <UiButton variant="secondary" @click="close">取消</UiButton>
        <UiButton
          variant="primary"
          :disabled="!name.trim() || isAdding"
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
  add: [name: string, aliases: string[], description: string]
}>()

const name = ref('')
const aliases = ref('')
const description = ref('')
const nameError = ref('')
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
  const charName = name.value.trim()
  if (!charName) {
    nameError.value = '请输入角色名称'
    return
  }
  nameError.value = ''

  const aliasList = aliases.value
    .split(/[,，]/)
    .map(a => a.trim())
    .filter(a => a.length > 0)

  isAdding.value = true
  emit('add', charName, aliasList, description.value.trim())

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
