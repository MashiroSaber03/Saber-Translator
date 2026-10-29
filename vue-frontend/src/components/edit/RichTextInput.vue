<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { EditorState, TextSelection, type Selection } from 'prosemirror-state'
import { Decoration, DecorationSet, EditorView } from 'prosemirror-view'
import { baseKeymap, joinBackward, splitBlockKeepMarks } from 'prosemirror-commands'
import { keymap } from 'prosemirror-keymap'
import { history, undo, redo } from 'prosemirror-history'
import type { BubbleState } from '@/types/bubble'
import { defaultRun, fromTextDocument, patchRunStyle, textSchema, toTextDocument, type EditorFont, type RunStyle } from './richTextDocument'

const props = defineProps<{ bubble: BubbleState; fonts: EditorFont[]; disabled?: boolean }>()
const emit = defineEmits<{
  update: [value: Partial<BubbleState>]
  selection: [value: { active: boolean; count: number; styles: RunStyle[] }]
}>()
const container = ref<HTMLElement>()
let view: EditorView | undefined
let identity: string | undefined
let previousBubble: BubbleState | undefined
let active = false
let disposed = false
const faces = new Map<string, FontFace>()
const fallback = () => defaultRun(props.bubble, props.fonts.find(f => f.id === props.bubble.fontFamily)?.style)
const plugins = () => [history(), keymap({
  'Mod-z': undo, 'Mod-y': redo, 'Mod-Shift-z': redo,
  'Mod-b': () => toggle('faux_bold'), 'Mod-i': () => toggle('faux_italic'), 'Mod-u': () => toggle('underline'),
  Enter: splitBlockKeepMarks, 'Shift-Enter': splitBlockKeepMarks,
}), keymap(baseKeymap)]

function inputState(doc: ReturnType<typeof toTextDocument>, selection?: Selection): EditorState {
  return EditorState.create({ doc, selection, plugins: plugins(),
    storedMarks: doc.textContent ? null : [textSchema.marks.run!.create({ style: doc.firstChild?.attrs.breakStyle ?? fallback() })],
  })
}

function selection(): void {
  if (!view) return
  const { from, to, empty, $from } = view.state.selection
  container.value?.toggleAttribute('data-selection-active', active && !empty)
  const styles: RunStyle[] = []
  if (!active) view.state.doc.descendants(node => { if (node.isText) styles.push(node.marks[0]?.attrs.style ?? fallback()) })
  else if (empty) styles.push((view.state.storedMarks ?? $from.marks())[0]?.attrs.style ?? fallback())
  else view.state.doc.nodesBetween(from, to, node => { if (node.isText) styles.push(node.marks[0]?.attrs.style ?? fallback()) })
  if (!styles.length) styles.push((view.state.storedMarks ?? $from.marks())[0]?.attrs.style ?? fallback())
  emit('selection', { active, count: empty || !active ? 0 : Array.from(view.state.doc.textBetween(from, to, '\n')).length, styles })
}

function sync(): void {
  if (!view || view.composing) return
  const nextIdentity = props.bubble.backendBubbleId ?? props.bubble.clientMutationId ?? ''
  const doc = toTextDocument(props.bubble.translatedText, props.bubble.richText, fallback())
  const previousIdentity = previousBubble?.backendBubbleId ?? previousBubble?.clientMutationId
  if (nextIdentity !== identity && nextIdentity !== previousIdentity) {
    active = false
    view.updateState(inputState(doc))
  } else if (!doc.eq(view.state.doc)) {
    const { from, to } = view.state.selection
    if (fromTextDocument(view.state.doc, fallback()).translatedText === props.bubble.translatedText) {
      // Canonical saves expand native defaults. Update marks without replacing
      // the characters: replacing the whole document would invalidate input undo.
      const transaction = view.state.tr.removeMark(0, doc.content.size, textSchema.marks.run)
      doc.descendants((node, position) => {
        if (node.isText) node.marks.forEach(mark => transaction.addMark(position, position + node.nodeSize, mark))
        else if (node.type.name === 'paragraph') transaction.setNodeMarkup(position, undefined, node.attrs)
      })
      transaction.setStoredMarks(view.state.storedMarks)
      view.updateState(view.state.apply(transaction.setMeta('addToHistory', false)))
    } else {
      // A different editor/OCR changed the content. Start a fresh input history.
      view.updateState(inputState(doc, TextSelection.between(
        doc.resolve(Math.min(from, doc.content.size - 1)), doc.resolve(Math.min(to, doc.content.size - 1)))))
    }
  }
  identity = nextIdentity
  previousBubble = props.bubble
  view.setProps({ editable: () => !props.disabled })
  view.dom.style.setProperty('--rich-text-document-size', String(Math.max(1, props.bubble.fontSize)))
  selection()
  void loadFonts()
}

async function loadFonts(): Promise<void> {
  const styles = props.bubble.richText?.runs.map(run => run.style) ?? [fallback()]
  const used = props.fonts.filter(font => font.id === props.bubble.fontFamily || styles.some(style =>
    style.postscript_name ? style.postscript_name === font.style.postscript_name : style.font_family === font.style.font_family))
  const ids = new Set(used.map(font => font.id))
  for (const [id, face] of faces) if (!ids.has(id)) { document.fonts.delete(face); faces.delete(id) }
  for (const font of used) {
    if (faces.has(font.id)) continue
    const face = new FontFace(String(font.style.font_family), `url("${font.url}")`, {
      weight: String(font.style.weight ?? 400), style: font.style.italic ? 'italic' : 'normal',
    })
    faces.set(font.id, face)
    try { await face.load(); if (!disposed && faces.get(font.id) === face) document.fonts.add(face) }
    catch { if (faces.get(font.id) === face) faces.delete(font.id) }
  }
}

function format(patch: RunStyle, replace = false): void {
  if (!view || props.disabled || view.composing) return
  const { state } = view
  const { from, to, empty, $from } = state.selection
  const apply = (style: RunStyle) => replace ? patch : patchRunStyle(style, patch)
  const transaction = state.tr
  if (active && empty && state.doc.textContent) {
    const style = (state.storedMarks ?? $from.marks())[0]?.attrs.style ?? fallback()
    transaction.setStoredMarks([textSchema.marks.run!.create({ style: apply(style) })])
  } else {
    const start = active ? from : 0
    const end = active ? to : state.doc.content.size
    state.doc.nodesBetween(start, end, (node, position) => {
      if (node.isText) transaction.addMark(Math.max(start, position), Math.min(end, position + node.nodeSize),
        textSchema.marks.run!.create({ style: apply(node.marks[0]?.attrs.style ?? fallback()) }))
      else if (node.type.name === 'paragraph' && (!active || (start <= position + node.nodeSize - 1 && end > position + node.nodeSize - 1)))
        transaction.setNodeMarkup(position, undefined, { ...node.attrs, breakStyle: apply(node.attrs.breakStyle ?? fallback()) })
    })
    if (!state.doc.textContent) {
      const style = apply(state.storedMarks?.[0]?.attrs.style ?? state.doc.firstChild?.attrs.breakStyle ?? fallback())
      transaction.setNodeMarkup(0, undefined, { ...state.doc.firstChild?.attrs, breakStyle: style })
      transaction.setStoredMarks([textSchema.marks.run!.create({ style })])
    }
  }
  view.dispatch(transaction)
}

function clearScope(event: PointerEvent): void {
  if (event.target instanceof Element && !event.target.closest('.bubble-editor, [role="listbox"], [role="dialog"]')
    && !container.value?.closest('.edit-workspace--picking-color')) { active = false; selection() }
}
function publish(): void { if (view && !disposed) emit('update', fromTextDocument(view.state.doc, fallback())) }
function toggle(field: string): boolean {
  if (!view) return false
  const styles: RunStyle[] = []
  view.state.doc.nodesBetween(view.state.selection.from, view.state.selection.to, node => { if (node.isText) styles.push(node.marks[0]?.attrs.style ?? fallback()) })
  if (!styles.length) styles.push((view.state.storedMarks ?? view.state.selection.$from.marks())[0]?.attrs.style ?? fallback())
  format({ [field]: !styles.every(style => style[field]) })
  return true
}
onMounted(() => {
  view = new EditorView(container.value!, {
    state: EditorState.create({ schema: textSchema }),
    attributes: { role: 'textbox', 'aria-label': '译文富文本', 'aria-multiline': 'true', style: 'outline:none;white-space:pre-wrap;overflow-wrap:anywhere;min-height:64px;font-size:16px;line-height:1.6' },
    // View-only feedback from the same selection used by formatting commands.
    decorations: ({ doc, selection: { from, to, empty } }) => empty ? DecorationSet.empty : DecorationSet.create(doc, [
      Decoration.inline(from, to, {
        class: 'rich-text-input__selection',
        style: 'background-color:var(--rich-text-input-selection-background, transparent)',
      }),
    ]),
    dispatchTransaction(transaction) {
      if (!view) return
      view.updateState(view.state.apply(transaction))
      if (transaction.docChanged && !view.composing) publish()
      selection()
    },
    handleDOMEvents: {
      focus: () => { active = true; selection(); return false },
      // Let ProseMirror finish its composition event in this turn. A delayed
      // timer could lose the last Chinese word when the user immediately flips pages.
      compositionend: () => { queueMicrotask(publish); return false },
    },
  })
  document.addEventListener('pointerdown', clearScope)
  sync()
})
watch(() => [props.bubble, props.fonts, props.disabled], sync, { deep: true, flush: 'post' })
onBeforeUnmount(() => {
  disposed = true
  document.removeEventListener('pointerdown', clearScope)
  view?.destroy()
  for (const face of faces.values()) document.fonts.delete(face)
  faces.clear()
})
defineExpose({
  format,
  getSelectedText: () => view && active && !view.state.selection.empty
    ? view.state.doc.textBetween(view.state.selection.from, view.state.selection.to, '\n') : '',
  focus: () => { active = true; view?.focus(); selection() },
  insertText: (text: string) => { if (view) view.dispatch(view.state.tr.insertText(text)) },
  deleteBackward: () => {
    if (!view) return
    const { state } = view
    if (!state.selection.empty) { view.dispatch(state.tr.deleteSelection()); return }
    if (joinBackward(state, view.dispatch, view)) return
    const before = state.selection.$from.parent.textBetween(0, state.selection.$from.parentOffset)
    const last = Array.from(new Intl.Segmenter(undefined, { granularity: 'grapheme' }).segment(before)).at(-1)?.segment
    if (last) view.dispatch(state.tr.delete(state.selection.from - last.length, state.selection.to))
  },
})
</script>

<template>
  <div ref="container" class="rich-text-input" :class="{ 'rich-text-input--disabled': disabled }" />
</template>

<style scoped>
.rich-text-input { padding: 10px 12px; min-height: 86px; max-height: 280px; overflow-y: auto; border: 1px solid var(--color-border-muted); border-radius: var(--radius-md); background: var(--color-surface-input); color: var(--color-text-default); }
.rich-text-input:focus-within { border-color: var(--color-border-brand); box-shadow: 0 0 0 3px var(--color-focus-brand-soft); }
.rich-text-input[data-selection-active]:not(:focus-within) { --rich-text-input-selection-background: color-mix(in srgb, var(--color-action-brand) 25%, transparent); }
.rich-text-input--disabled { opacity: 0.6; pointer-events: none; }
</style>
