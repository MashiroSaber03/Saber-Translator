import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { mount } from '@vue/test-utils'
import { defineComponent, ref } from 'vue'

import OpenAIExtraBodyEditor from '@/components/common/OpenAIExtraBodyEditor.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import {
  applyOpenAiOptionsPatch,
  cloneOpenAiOptions,
  deserializeOpenAICompatibleOptionsFromApi,
  createDefaultOpenAiOptions,
  serializeOpenAICompatibleOptionsForApi,
} from '@/utils/openaiOptions'

function source(file: string): string {
  return readFileSync(resolve(process.cwd(), file), 'utf8')
}

const uiFieldRootSelector = '.ui' + '-field'

describe('openai options schema boundaries', () => {
  it('uses the shared clone helper inside the OpenAI option owner', () => {
    const content = source('src/utils/openaiOptions.ts')

    expect(content).toContain("import { deepClone } from './deepClone'")
    expect(content).not.toContain('JSON.parse(JSON.stringify')
  })

  it('keeps the shared extra body editor on the shared clone helper', () => {
    const content = source('src/components/common/OpenAIExtraBodyEditor.vue')

    expect(content).toContain("import { deepClone } from '@/utils/deepClone'")
    expect(content).not.toContain('function cloneRecord')
    expect(content).not.toContain('JSON.parse(JSON.stringify')
  })

  it('updates the shared extra body editor through typed textarea model events', () => {
    const content = source('src/components/common/OpenAIExtraBodyEditor.vue')

    expect(content).not.toContain('@input="handleInput"')
    expect(content).not.toContain('event.target as HTMLTextAreaElement')
    expect(content).toContain('@update:model-value="handleInput"')

    const wrapper = mount(OpenAIExtraBodyEditor)
    expect(wrapper.getComponent(UiTextarea).props('variant')).toBe('panel')
    wrapper.getComponent(UiTextarea).vm.$emit('update:modelValue', '{"top_p":0.9}')

    expect(wrapper.emitted('update:modelValue')?.[0]).toEqual([{ top_p: 0.9 }])
  })

  it('routes the shared extra body editor label and feedback through UiField', async () => {
    const content = source('src/components/common/OpenAIExtraBodyEditor.vue')

    expect(content).toContain("import UiField from '@/components/ui/UiField.vue'")
    expect(content).not.toMatch(/<label\b/)
    expect(content).not.toContain('class="ui-form-hint"')
    expect(content).not.toContain('class="input-error"')
    expect(content).not.toContain('.editor-header label')
    expect(content).not.toContain(`.openai-extra-body-editor${uiFieldRootSelector}`)

    const wrapper = mount(OpenAIExtraBodyEditor, {
      props: {
        label: '附加请求字段',
        hint: '只允许 JSON 对象',
      },
    })

    const field = wrapper.getComponent(UiField)
    expect(field.props('variant')).toBe('settings')
    expect(field.props('label')).toBe('附加请求字段')
    expect(field.props('controlId')).toBeTruthy()
    expect(wrapper.getComponent(UiTextarea).attributes('id')).toBe(field.props('controlId'))
    expect(field.props('hint')).toBe('只允许 JSON 对象')
    expect(field.findComponent(UiButton).text()).toBe('格式化')

    wrapper.getComponent(UiTextarea).vm.$emit('update:modelValue', '[]')
    await wrapper.vm.$nextTick()

    expect(wrapper.getComponent(UiField).props('error')).toBe('必须输入 JSON 对象，不能是数组、字符串或数字')
  })

  it('keeps editor ids unique and does not reformat a local valid draft', async () => {
    const Host = defineComponent({
      components: { OpenAIExtraBodyEditor },
      setup() {
        const first = ref<Record<string, unknown>>()
        const second = ref<Record<string, unknown>>()
        return { first, second }
      },
      template: `
        <OpenAIExtraBodyEditor v-model="first" />
        <OpenAIExtraBodyEditor v-model="second" />
      `,
    })
    const wrapper = mount(Host)
    const editors = wrapper.findAllComponents(OpenAIExtraBodyEditor)
    const fields = wrapper.findAllComponents(UiField)

    expect(fields[0]!.props('controlId')).not.toBe(fields[1]!.props('controlId'))

    const firstTextarea = editors[0]!.getComponent(UiTextarea)
    firstTextarea.vm.$emit('update:modelValue', '{"top_p": 0.9}')
    await wrapper.vm.$nextTick()

    expect(firstTextarea.props('modelValue')).toBe('{"top_p": 0.9}')
    expect(editors[0]!.emitted('update:modelValue')?.at(-1)).toEqual([{ top_p: 0.9 }])
  })

  it('clones option payloads without sharing nested extra body references', () => {
    const options = {
      request: {
        forceJsonOutput: true,
        extraBody: { response_format: { type: 'json_object' } },
      },
      execution: {
        useStream: true,
        rpmLimit: 12,
        transportRetries: 2,
        businessRetries: 3,
      },
    }

    const cloned = cloneOpenAiOptions(options)
    const serialized = serializeOpenAICompatibleOptionsForApi(options)

    expect(cloned).toEqual(options)
    expect(cloned.request.extraBody).not.toBe(options.request.extraBody)
    expect(serialized.request.extra_body).not.toBe(options.request.extraBody)
  })

  it('applies UI patch fields to the nested OpenAI option shape', () => {
    const options = createDefaultOpenAiOptions({
      request: {
        forceJsonOutput: false,
        extraBody: { previous: true },
      },
      execution: {
        useStream: false,
        rpmLimit: 0,
        transportRetries: 1,
        businessRetries: 0,
      },
    })

    const patched = applyOpenAiOptionsPatch(options, {
      provider: 'siliconflow',
      rpmLimit: 12,
      transportRetries: 3,
      businessRetries: 2,
      forceJsonOutput: true,
      useStream: true,
      extraBody: { top_p: 0.9 },
    })

    expect(patched).toBe(options)
    expect(patched).toEqual({
      request: {
        forceJsonOutput: true,
        extraBody: { top_p: 0.9 },
      },
      execution: {
        useStream: true,
        rpmLimit: 12,
        transportRetries: 3,
        businessRetries: 2,
      },
    })

    applyOpenAiOptionsPatch(options, { extraBody: undefined })

    expect(options.request.extraBody).toBeUndefined()
  })

  it('serializes current frontend options to the backend wire shape', () => {
    const payload = serializeOpenAICompatibleOptionsForApi({
      request: {
        forceJsonOutput: true,
        temperature: 0.2,
        extraBody: { top_p: 0.9 },
      },
      execution: {
        useStream: true,
        rpmLimit: 12,
        transportRetries: 2,
        businessRetries: 3,
      },
    })

    expect(payload).toEqual({
      request: {
        force_json_output: true,
        temperature: 0.2,
        extra_body: { top_p: 0.9 },
      },
      execution: {
        use_stream: true,
        rpm_limit: 12,
        transport_retries: 2,
        business_retries: 3,
      },
    })
  })

  it('deserializes backend wire options through the API adapter', () => {
    const options = deserializeOpenAICompatibleOptionsFromApi({
      request: {
        force_json_output: true,
        temperature: 0.2,
        extra_body: { top_p: 0.9 },
      },
      execution: {
        use_stream: true,
        rpm_limit: 12,
        transport_retries: 2,
        business_retries: 3,
      },
    })

    expect(options).toEqual({
      request: {
        forceJsonOutput: true,
        temperature: 0.2,
        extraBody: { top_p: 0.9 },
      },
      execution: {
        useStream: true,
        rpmLimit: 12,
        transportRetries: 2,
        businessRetries: 3,
      },
    })
  })

})
