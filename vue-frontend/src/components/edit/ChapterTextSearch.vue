<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { apiClient } from '@/api/client'
import type { components } from '@/api/generated/v2'
import { getPageDocument } from '@/api/v2/content'
import { getChapterTextExportUrl } from '@/api/v2/translation'
import { pageDocumentToBubbles } from '@/adapters/v2ContentAdapter'
import { flushPageDocument, queuePageDocumentMutation, registerPageDocument } from '@/services/pageDocumentPersistence'
import { useImageStore } from '@/stores/imageStore'
import { replaceBubbleText } from './richTextDocument'
import { showToast } from '@/utils/toast'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiIcon from '@/components/ui/UiIcon.vue'

type TextPage = components['schemas']['ChapterTextExportPage']
const props = defineProps<{ chapterId: string; disabled?: boolean; prepare: () => Promise<void> }>()
const emit = defineEmits<{ locate: [pageId: string, bubbleId: string]; changed: []; busy: [value: boolean] }>()
const images = useImageStore()
const opened = ref(false), find = ref(''), replacement = ref(''), loading = ref(false), busy = ref(false), confirm = ref(false)
const pages = ref<TextPage[]>([])
const error = ref('')
const controller = new AbortController()
const currentTexts = computed(() => pages.value.map(page => {
  const image = images.currentImage
  return image?.id === page.page_id && image.bubbleStates ? { ...page,
    document_revision: image.documentRevision ?? page.document_revision,
    bubbles: image.bubbleStates.flatMap(bubble => bubble.backendBubbleId ? [{ bubble_id: bubble.backendBubbleId,
      translated_text: bubble.translatedText, original_text: bubble.originalText, textbox_text: bubble.textboxText, text_direction: bubble.textDirection }] : []),
  } : page
}))
const matches = computed(() => find.value ? currentTexts.value.flatMap(page => page.bubbles.filter(bubble => bubble.translated_text.includes(find.value))
  .map(bubble => ({ page, bubble, count: bubble.translated_text.split(find.value).length - 1 }))) : [])
const total = computed(() => matches.value.reduce((sum, item) => sum + item.count, 0))
watch([find, replacement], () => { confirm.value = false })
watch(() => props.chapterId, () => { opened.value = false; pages.value = []; error.value = '' })
watch(opened, value => { if (value) void load() })
onBeforeUnmount(() => controller.abort())

async function load(): Promise<void> {
  loading.value = true
  error.value = ''
  const chapter = props.chapterId
  try {
    await props.prepare()
    const result = await apiClient.get<components['schemas']['ChapterTextExport']>(getChapterTextExportUrl(chapter), { signal: controller.signal })
    if (chapter === props.chapterId) pages.value = result.pages
  } catch (cause) { if (!controller.signal.aborted) error.value = cause instanceof Error ? cause.message : '无法读取章节文字' }
  finally { loading.value = false }
}
async function replace(all: boolean): Promise<void> {
  if (busy.value || loading.value || props.disabled || !matches.value.length || !find.value) return
  if (all && !confirm.value) { confirm.value = true; return }
  const needle = find.value, value = replacement.value, chapter = props.chapterId
  busy.value = true
  emit('busy', true)
  error.value = ''
  let completed = 0
  try {
    await props.prepare()
    const targets = all ? [...matches.value] : matches.value.slice(0, 1)
    const pageIds = [...new Set(targets.map(item => item.page.page_id))]
    for (const pageId of pageIds) {
      await flushPageDocument(pageId)
      const document = await getPageDocument(pageId, controller.signal)
      const expected = targets.find(item => item.page.page_id === pageId)!.page
      if (chapter !== props.chapterId || document.documentRevision !== expected.document_revision) throw new Error('章节文字已发生变化，请刷新查找结果后重试；已完成的页面已保存')
      const bubbles = pageDocumentToBubbles(document)
      const selected = new Set(targets.filter(item => item.page.page_id === pageId).map(item => item.bubble.bubble_id))
      const fields: Record<string, (keyof typeof bubbles[number])[]> = {}
      const next = bubbles.map(bubble => {
        if (!selected.has(bubble.backendBubbleId!)) return bubble
        fields[bubble.backendBubbleId!] = ['translatedText', 'richText']
        return { ...bubble, ...replaceBubbleText(bubble, needle, value) }
      })
      registerPageDocument(document)
      await queuePageDocumentMutation(pageId, document.documentRevision, next, { bubbleFields: fields })
      completed++
    }
    showToast(`已保存 ${completed} 页译文替换，正在更新出图`, 'success')
  } catch (cause) { error.value = `${cause instanceof Error ? cause.message : '替换失败'}${completed ? `（已完成 ${completed} 页）` : ''}` }
  finally {
    busy.value = false; emit('busy', false); confirm.value = false; emit('changed')
    if (!controller.signal.aborted && images.currentImage?.chapterId === chapter) {
      const replacementError = error.value
      await load()
      if (replacementError) error.value = replacementError
    }
  }
}
</script>

<template>
  <section class="chapter-text-search">
    <UiButton class="chapter-text-search__toggle" size="sm" variant="toolbar" :aria-expanded="opened" :disabled="disabled || busy" @click="opened = !opened">
      <UiIcon name="search" size="15" />
      <span class="chapter-text-search__title">章节查找替换</span>
      <UiIcon :name="opened ? 'chevron-down' : 'chevron-right'" size="14" />
    </UiButton>
    <div v-if="opened" class="chapter-text-search__panel" :aria-busy="loading || busy" @keydown.esc.stop="opened = false">
      <UiField variant="editor" label="查找译文" control-id="chapterFindText">
        <UiInput id="chapterFindText" v-model="find" size="sm" aria-label="查找译文" placeholder="输入关键词，区分大小写" :disabled="busy" />
      </UiField>
      <UiField variant="editor" label="替换为" control-id="chapterReplaceText">
        <UiInput id="chapterReplaceText" v-model="replacement" size="sm" aria-label="替换为" placeholder="留空则删除匹配文字" :disabled="busy" />
      </UiField>
      <div class="chapter-text-search__status">
        <span role="status">{{ loading ? '读取中…' : `共 ${total} 处 / ${matches.length} 个框` }}</span>
        <UiButton size="xs" variant="toolbar" :disabled="disabled || loading || busy" @click="load"><UiIcon name="refresh" size="13" />刷新</UiButton>
      </div>
      <div class="chapter-text-search__actions">
        <UiButton size="sm" variant="secondary" :disabled="disabled || loading || busy || !total" @click="replace(false)">替换首个框</UiButton>
        <UiButton size="sm" variant="primary" :disabled="disabled || loading || busy || !total" @click="replace(true)">{{ confirm ? `确认替换全部 ${total} 处` : '全部替换' }}</UiButton>
      </div>
      <p v-if="error" class="chapter-text-search__error" role="alert">{{ error }}</p>
      <div v-if="matches.length" class="chapter-text-search__results">
        <UiButton class="chapter-text-search__result" v-for="item in matches.slice(0, 100)" :key="item.bubble.bubble_id" :aria-label="`第 ${item.page.page_number} 页 · ${item.bubble.translated_text.slice(0, 80)}`" variant="toolbar" size="sm" :disabled="disabled || busy" @click="emit('locate', item.page.page_id, item.bubble.bubble_id)">
          <span class="chapter-text-search__page">第 {{ item.page.page_number }} 页</span><span class="chapter-text-search__excerpt">{{ item.bubble.translated_text.slice(0, 80) }}</span><UiIcon name="chevron-right" size="13" />
        </UiButton>
        <p v-if="matches.length > 100">仅列出前 100 个框；全部替换包含其余匹配。</p>
      </div>
    </div>
  </section>
</template>

<style scoped>
.chapter-text-search { flex-shrink: 0; padding: 6px 14px; border-bottom: 1px solid var(--color-border-muted); background: var(--color-surface-card); }
.chapter-text-search__toggle { display: flex; align-items: center; gap: 8px; width: 100%; padding: 8px 0; color: var(--color-text-strong); font-size: 12px; font-weight: 600; }
.chapter-text-search__title { flex: 1; text-align: left; }
.chapter-text-search__panel { display: grid; gap: 10px; padding: 6px 0 10px; }
.chapter-text-search__status { display: flex; align-items: center; justify-content: space-between; color: var(--color-text-secondary); font-size: 12px; }
.chapter-text-search__actions { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
.chapter-text-search__results { display: grid; grid-template-columns: minmax(0, 1fr); gap: 4px; max-height: 120px; overflow-y: auto; border-top: 1px solid var(--color-border-muted); padding-top: 8px; }
.chapter-text-search__result { display: flex; align-items: center; gap: 8px; min-width: 0; padding: 8px; border-radius: 6px; background: var(--color-surface-quiet); text-align: left; }
.chapter-text-search__page { flex-shrink: 0; color: var(--color-text-brand); font-size: 11px; }
.chapter-text-search__excerpt { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.chapter-text-search__error { margin: 0; color: var(--color-text-danger-strong); font-size: 12px; }
</style>
