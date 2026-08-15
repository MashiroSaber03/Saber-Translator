<script setup lang="ts">
import { computed } from 'vue'
import ProductLogPanel from '@/components/product/ProductLogPanel.vue'
import type { AgentLog, WebImportStatus } from '@/types/webImport'

type ProductLogTone = 'neutral' | 'info' | 'success' | 'warning' | 'danger' | 'accent'

const props = defineProps<{
  expanded: boolean
  logs: AgentLog[]
  status: WebImportStatus
}>()

defineEmits<{
  (event: 'toggle'): void
}>()

const toneByLogType: Record<AgentLog['type'], ProductLogTone> = {
  info: 'info',
  tool_call: 'warning',
  tool_result: 'success',
  thinking: 'accent',
  error: 'danger',
}

const logItems = computed(() => {
  return props.logs.map((log, index) => ({
    id: index,
    message: log.message,
    timestamp: log.timestamp,
    tone: toneByLogType[log.type],
  }))
})
</script>

<template>
  <ProductLogPanel
    v-if="logs.length > 0"
    :expanded="expanded"
    :items="logItems"
    title="AI 工作日志"
    aria-label="网页导入 AI 工作日志"
    :active-hint="status === 'extracting' ? '(提取中...)' : ''"
    @toggle="$emit('toggle')"
  />
</template>
