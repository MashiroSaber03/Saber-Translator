<template>
  <ContinuationDialogShell title="✏️ 编辑角色" @close="close">
    <div class="continuation-dialog-form">
      <div class="continuation-dialog-form__field">
        <label>角色名称</label>
        <UiInput
          v-model="localName"
          type="text"
          class="continuation-dialog__form-input"
          placeholder="输入角色主名称"
        />
      </div>

      <div class="continuation-dialog-form__field">
        <label>别名（用逗号分隔）</label>
        <UiInput
          v-model="localAliases"
          type="text"
          class="continuation-dialog__form-input"
          placeholder="例如: 桐乃, 新垣彩世"
        />
        <p class="form-hint">AI生成脚本时可能使用这些名字引用角色</p>
      </div>
    </div>

    <template #footer>
      <div class="continuation-dialog-actions">
        <UiButton variant="secondary" @click="close">取消</UiButton>
        <UiButton variant="primary" @click="save">💾 保存</UiButton>
      </div>
    </template>
  </ContinuationDialogShell>
</template>

<script setup lang="ts">
import UiInput from '@/components/ui/UiInput.vue'
import { ref, watch } from 'vue'
import type { CharacterProfile } from '@/api/continuation'
import UiButton from '@/components/ui/UiButton.vue'
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
