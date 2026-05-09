import { afterEach, describe, expect, it } from 'vitest'
import { mount, type VueWrapper } from '@vue/test-utils'
import BaseModal from '@/components/common/BaseModal.vue'

const mountedWrappers: VueWrapper[] = []

function mountModal(props: Record<string, unknown> = {}): VueWrapper {
  const wrapper = mount(BaseModal, {
    attachTo: document.body,
    props: {
      modelValue: true,
      title: 'Test Modal',
      ...props,
    },
    slots: {
      default: '<div class="modal-test-content">Modal content</div>',
    },
  })

  mountedWrappers.push(wrapper)
  return wrapper
}

function getOverlay(): HTMLDivElement {
  const overlay = document.body.querySelector('.modal-overlay')
  expect(overlay).toBeTruthy()
  return overlay as HTMLDivElement
}

function getContainer(): HTMLDivElement {
  const container = document.body.querySelector('.modal-container')
  expect(container).toBeTruthy()
  return container as HTMLDivElement
}

function getCloseButton(): HTMLButtonElement {
  const closeButton = document.body.querySelector('.modal-close-btn')
  expect(closeButton).toBeTruthy()
  return closeButton as HTMLButtonElement
}

function dispatchMouseEvent(target: Element, type: 'mousedown' | 'mouseup' | 'click') {
  target.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true }))
}

afterEach(() => {
  while (mountedWrappers.length > 0) {
    mountedWrappers.pop()?.unmount()
  }
  document.body.innerHTML = ''
  document.body.style.overflow = ''
})

describe('BaseModal', () => {
  it('closes when the pointer press and release both happen on the overlay', () => {
    const wrapper = mountModal()
    const overlay = getOverlay()

    dispatchMouseEvent(overlay, 'mousedown')
    dispatchMouseEvent(overlay, 'mouseup')
    dispatchMouseEvent(overlay, 'click')

    expect(wrapper.emitted('close')).toHaveLength(1)
    expect(wrapper.emitted('update:modelValue')).toEqual([[false]])
  })

  it('does not close on overlay interaction when closeOnOverlay is disabled', () => {
    const wrapper = mountModal({ closeOnOverlay: false })
    const overlay = getOverlay()

    dispatchMouseEvent(overlay, 'mousedown')
    dispatchMouseEvent(overlay, 'mouseup')
    dispatchMouseEvent(overlay, 'click')

    expect(wrapper.emitted('close')).toBeUndefined()
    expect(wrapper.emitted('update:modelValue')).toBeUndefined()
  })

  it('does not close when pointer press starts inside the modal and releases on the overlay', () => {
    const wrapper = mountModal()
    const overlay = getOverlay()
    const container = getContainer()

    dispatchMouseEvent(container, 'mousedown')
    dispatchMouseEvent(overlay, 'mouseup')
    dispatchMouseEvent(overlay, 'click')

    expect(wrapper.emitted('close')).toBeUndefined()
    expect(wrapper.emitted('update:modelValue')).toBeUndefined()
  })

  it('does not close when pointer press starts on the overlay and releases inside the modal', () => {
    const wrapper = mountModal()
    const overlay = getOverlay()
    const container = getContainer()

    dispatchMouseEvent(overlay, 'mousedown')
    dispatchMouseEvent(container, 'mouseup')
    dispatchMouseEvent(container, 'click')

    expect(wrapper.emitted('close')).toBeUndefined()
    expect(wrapper.emitted('update:modelValue')).toBeUndefined()
  })

  it('does not close when clicking inside the modal content', () => {
    const wrapper = mountModal()
    const container = getContainer()

    dispatchMouseEvent(container, 'mousedown')
    dispatchMouseEvent(container, 'mouseup')
    dispatchMouseEvent(container, 'click')

    expect(wrapper.emitted('close')).toBeUndefined()
    expect(wrapper.emitted('update:modelValue')).toBeUndefined()
  })

  it('closes when Escape is pressed', () => {
    const wrapper = mountModal()

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))

    expect(wrapper.emitted('close')).toHaveLength(1)
    expect(wrapper.emitted('update:modelValue')).toEqual([[false]])
  })

  it('closes when the close button is clicked', () => {
    const wrapper = mountModal()
    const closeButton = getCloseButton()

    dispatchMouseEvent(closeButton, 'click')

    expect(wrapper.emitted('close')).toHaveLength(1)
    expect(wrapper.emitted('update:modelValue')).toEqual([[false]])
  })
})
