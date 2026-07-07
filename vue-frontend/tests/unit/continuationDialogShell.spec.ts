import { afterEach, describe, expect, it } from 'vitest'
import { mount, type VueWrapper } from '@vue/test-utils'

import ContinuationDialogShell from '@/components/insight/continuation/ContinuationDialogShell.vue'

const mountedWrappers: VueWrapper[] = []

function mountShell(props: Record<string, unknown> = {}) {
  const wrapper = mount(ContinuationDialogShell, {
    attachTo: document.body,
    props: {
      title: 'Dialog Title',
      ...props,
    },
    slots: {
      default: '<p>Dialog body</p>',
      footer: '<button>Done</button>',
    },
  })

  mountedWrappers.push(wrapper)
  return wrapper
}

function getContainer(): HTMLDivElement {
  const container = document.body.querySelector('[data-testid="base-dialog-container"]')
  expect(container).toBeTruthy()
  return container as HTMLDivElement
}

afterEach(() => {
  while (mountedWrappers.length > 0) {
    mountedWrappers.pop()?.unmount()
  }
  document.body.innerHTML = ''
  document.body.style.overflow = ''
})

describe('ContinuationDialogShell', () => {
  it('uses a typed width variant instead of inferring layout from custom class names', () => {
    mountShell({ customClass: 'continuation-dialog-modal--wide' })

    expect(getContainer().style.maxWidth).toBe('520px')
  })

  it('supports the wide layout through an explicit prop', () => {
    mountShell({ widthVariant: 'wide' })

    expect(getContainer().style.maxWidth).toBe('600px')
  })
})
