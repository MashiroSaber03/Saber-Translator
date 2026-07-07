import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { mount } from '@vue/test-utils'

import OpenAIExtraBodyEditor from '@/components/common/OpenAIExtraBodyEditor.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import {
  applyOpenAiOptionsPatch,
  cloneOpenAiOptions,
  deserializeOpenAICompatibleOptionsFromApi,
  normalizeOpenAiOptions,
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
    expect(field.props('controlId')).toBe('openAiExtraBody')
    expect(field.props('hint')).toBe('只允许 JSON 对象')
    expect(field.findComponent(UiButton).text()).toBe('格式化')

    wrapper.getComponent(UiTextarea).vm.$emit('update:modelValue', '[]')
    await wrapper.vm.$nextTick()

    expect(wrapper.getComponent(UiField).props('error')).toBe('必须输入 JSON 对象，不能是数组、字符串或数字')
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
    const options = normalizeOpenAiOptions({
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

  it('normalizes only the current frontend option shape', () => {
    const options = normalizeOpenAiOptions({
      request: {
        force_json_output: true,
        extra_body: { top_p: 0.9 },
      },
      execution: {
        use_stream: true,
        rpm_limit: 12,
        transport_retries: 2,
        business_retries: 3,
      },
    })

    expect(options.request.forceJsonOutput).toBe(false)
    expect(options.request.extraBody).toBeUndefined()
    expect(options.execution.useStream).toBe(false)
    expect(options.execution.rpmLimit).toBe(0)
    expect(options.execution.transportRetries).toBe(1)
    expect(options.execution.businessRetries).toBe(0)
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

  it('falls back when numeric option fields are not finite', () => {
    const frontendOptions = normalizeOpenAiOptions(
      {
        request: {
          temperature: Number.POSITIVE_INFINITY,
        },
        execution: {
          rpmLimit: 'Infinity',
          transportRetries: Number.NEGATIVE_INFINITY,
          businessRetries: Number.NaN,
        },
      },
      {
        request: {
          temperature: 0.6,
        },
        execution: {
          rpmLimit: 8,
          transportRetries: 2,
          businessRetries: 3,
        },
      }
    )

    expect(frontendOptions.request.temperature).toBe(0.6)
    expect(frontendOptions.execution.rpmLimit).toBe(8)
    expect(frontendOptions.execution.transportRetries).toBe(2)
    expect(frontendOptions.execution.businessRetries).toBe(3)

    const apiOptions = deserializeOpenAICompatibleOptionsFromApi(
      {
        request: {
          temperature: 'Infinity',
        },
        execution: {
          rpm_limit: Number.POSITIVE_INFINITY,
          transport_retries: '-Infinity',
          business_retries: Number.NaN,
        },
      },
      {
        request: {
          temperature: 0.4,
        },
        execution: {
          rpmLimit: 4,
          transportRetries: 5,
          businessRetries: 6,
        },
      }
    )

    expect(apiOptions.request.temperature).toBe(0.4)
    expect(apiOptions.execution.rpmLimit).toBe(4)
    expect(apiOptions.execution.transportRetries).toBe(5)
    expect(apiOptions.execution.businessRetries).toBe(6)
  })

  it('falls back when boolean option fields are not real booleans', () => {
    const frontendOptions = normalizeOpenAiOptions({
      request: {
        forceJsonOutput: 'false',
      },
      execution: {
        useStream: 'false',
      },
    })

    expect(frontendOptions.request.forceJsonOutput).toBe(false)
    expect(frontendOptions.execution.useStream).toBe(false)

    const apiOptions = deserializeOpenAICompatibleOptionsFromApi({
      request: {
        force_json_output: 'false',
      },
      execution: {
        use_stream: 'false',
      },
    })

    expect(apiOptions.request.forceJsonOutput).toBe(false)
    expect(apiOptions.execution.useStream).toBe(false)
  })
})
