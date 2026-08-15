<template>
  <ContinuationDialogShell title="编辑角色" :dismissible="!busy" @close="close">
    <ContinuationDialogForm>
      <ContinuationDialogField
        label="角色名称"
        control-id="continuationEditCharacterName"
        required
        :error="nameError"
      >
        <UiInput
          id="continuationEditCharacterName"
          v-model="localName"
          type="text"
          aria-label="角色名称"
          class="continuation-dialog__form-input"
          :error="Boolean(nameError)"
          placeholder="输入角色主名称"
        />
      </ContinuationDialogField>

      <ContinuationDialogField
        label="别名（用逗号分隔）"
        control-id="continuationEditCharacterAliases"
        hint="AI生成脚本时可能使用这些名字引用角色"
      >
        <UiInput
          id="continuationEditCharacterAliases"
          v-model="localAliases"
          type="text"
          aria-label="别名（用逗号分隔）"
          class="continuation-dialog__form-input"
          placeholder="例如: 桐乃, 新垣彩世"
        />
      </ContinuationDialogField>
    </ContinuationDialogForm>

    <template #footer>
      <ContinuationDialogActions>
        <UiButton variant="secondary" :disabled="busy" @click="close">取消</UiButton>
        <UiButton variant="primary" :disabled="!localName.trim() || busy" @click="save">
          <UiIcon v-if="!busy" name="save" size="15" />
          <span>{{ busy ? '保存中...' : '保存' }}</span>
        </UiButton>
      </ContinuationDialogActions>
    </template>
  </ContinuationDialogShell>
</template>

<script setup lang="ts">
import UiInput from '@/components/ui/UiInput.vue'
import { ref, watch } from 'vue'
import type { CharacterProfile } from '@/api/continuation'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import ContinuationDialogActions from './ContinuationDialogActions.vue'
import ContinuationDialogField from './ContinuationDialogField.vue'
import ContinuationDialogForm from './ContinuationDialogForm.vue'
import ContinuationDialogShell from './ContinuationDialogShell.vue'

const props = defineProps<{
  character: CharacterProfile
  busy?: boolean
}>()

const emit = defineEmits<{
  close: []
  save: [name: string, aliases: string[]]
}>()

const localName = ref(props.character.name)
const localAliases = ref(props.character.aliases.join(', '))
const nameError = ref('')
const close = () => emit('close')

watch(() => props.character, (newChar) => {
  localName.value = newChar.name
  localAliases.value = newChar.aliases.join(', ')
}, { immediate: true })

function save() {
  const name = localName.value.trim()
  const aliases = localAliases.value
    .split(/[,，]/)
    .map(a => a.trim())
    .filter(a => a.length > 0)

  if (!name) {
    nameError.value = '请输入角色名称'
    return
  }
  nameError.value = ''

  emit('save', name, aliases)
}
</script>
