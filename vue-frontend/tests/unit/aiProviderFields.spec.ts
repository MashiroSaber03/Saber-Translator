import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import AiProviderCredentialFields from '@/components/settings/AiProviderCredentialFields.vue'
import AiProviderSelectField from '@/components/settings/AiProviderSelectField.vue'
import UiField from '@/components/ui/UiField.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiPasswordField from '@/components/ui/UiPasswordField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'

describe('AI provider fields', () => {
  it('emits normalized provider selection events through the shared select field', async () => {
    const wrapper = mount(AiProviderSelectField, {
      props: {
        modelValue: 'openai',
        inputId: 'provider-id',
        options: [
          { label: 'OpenAI', value: 'openai' },
          { label: 'DeepSeek', value: 'deepseek' },
        ],
      },
    })

    wrapper.getComponent(UiSelect).vm.$emit('update:modelValue', 'deepseek')
    wrapper.getComponent(UiSelect).vm.$emit('change', 'deepseek')

    expect(wrapper.emitted('update:modelValue')).toEqual([['deepseek']])
    expect(wrapper.emitted('change')).toEqual([['deepseek']])
  })

  it('renders only the credential fields owned by the current provider', async () => {
    const wrapper = mount(AiProviderCredentialFields, {
      props: {
        apiKey: 'secret',
        apiKeyInputId: 'api-key-id',
        baseUrl: 'https://example.test/v1',
        baseUrlInputId: 'base-url-id',
        showApiKey: true,
        showBaseUrl: false,
      },
    })

    expect(wrapper.findComponent(UiPasswordField).exists()).toBe(true)
    expect(wrapper.findAllComponents(UiInput)).toHaveLength(2)
    const baseUrlField = () => wrapper.findAllComponents(UiField)
      .find(field => field.props('label') === 'Base URL')
    expect(baseUrlField()?.attributes('style')).toContain('display: none')

    await wrapper.setProps({ showBaseUrl: true })
    wrapper.getComponent(UiPasswordField).vm.$emit('update:modelValue', 'next-secret')
    expect(baseUrlField()?.attributes('style')).not.toContain('display: none')
    wrapper.findAllComponents(UiInput)[1].vm.$emit('update:modelValue', 'https://next.test/v1')

    expect(wrapper.emitted('update:apiKey')).toEqual([['next-secret']])
    expect(wrapper.emitted('update:baseUrl')).toEqual([['https://next.test/v1']])
  })
})
