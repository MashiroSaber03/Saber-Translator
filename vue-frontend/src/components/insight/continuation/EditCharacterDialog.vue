<template>
  <ContinuationDialogShell title="✏️ 编辑角色" @close="close">
    <ContinuationDialogForm>
      <ContinuationDialogField label="角色名称">
        <UiInput
          v-model="localName"
          type="text"
          aria-label="角色名称"
          class="continuation-dialog__form-input"
          style="font: inherit"
          placeholder="输入角色主名称"
        />
      </ContinuationDialogField>

      <ContinuationDialogField
        label="别名（用逗号分隔）"
        hint="AI生成脚本时可能使用这些名字引用角色"
      >
        <UiInput
          v-model="localAliases"
          type="text"
          aria-label="别名（用逗号分隔）"
          class="continuation-dialog__form-input"
          style="font: inherit"
          placeholder="例如: 桐乃, 新垣彩世"
        />
      </ContinuationDialogField>
    </ContinuationDialogForm>

    <template #footer>
      <ContinuationDialogActions>
        <UiButton variant="secondary" @click="close">取消</UiButton>
        <UiButton variant="primary" @click="save">💾 保存</UiButton>
      </ContinuationDialogActions>
    </template>
  </ContinuationDialogShell>
</template>

<script setup lang="ts">
import UiInput from '@/components/ui/UiInput.vue'
import { ref, watch } from 'vue'
import type { CharacterProfile } from '@/api/continuation'
import UiButton from '@/components/ui/UiButton.vue'
import ContinuationDialogActions from './ContinuationDialogActions.vue'
import ContinuationDialogField from './ContinuationDialogField.vue'
import ContinuationDialogForm from './ContinuationDialogForm.vue'
import ContinuationDialogShell from './ContinuationDialogShell.vue'

const props = defineProps<{
  character: CharacterProfile
}>()

const emit = defineEmits<{
  close: []
  save: [name: string, aliases: string[]]
}>()

const localName = ref(props.character.name)
const localAliases = ref(props.character.aliases.join(', '))
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
    alert('角色名不能为空')
    return
  }

  emit('save', name, aliases)
}
</script>
