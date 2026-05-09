import { defineComponent, h } from 'vue'
import { afterEach, describe, expect, it } from 'vitest'
import { mount, type VueWrapper } from '@vue/test-utils'
import { useOverlayDismiss } from '@/composables/useOverlayDismiss'

const mountedWrappers: VueWrapper[] = []

const OverlayHarness = defineComponent({
  props: {
    enabled: {
      type: Boolean,
      default: true,
    },
  },
  emits: ['dismiss'],
  setup(props, { emit }) {
    const { overlayRef, handleOverlayMouseDown } = useOverlayDismiss(() => emit('dismiss'), {
      enabled: () => props.enabled,
    })

    return () => h(
      'div',
      {
        ref: overlayRef,
        class: 'overlay-harness',
        onMousedown: (event: MouseEvent) => {
          if (event.target === event.currentTarget) {
            handleOverlayMouseDown(event)
          }
        },
      },
      h('div', { class: 'overlay-content' }, 'content'),
    )
  },
})

function mountHarness(props: Record<string, unknown> = {}): VueWrapper {
  const wrapper = mount(OverlayHarness, {
    attachTo: document.body,
    props,
  })
  mountedWrappers.push(wrapper)
  return wrapper
}

function getOverlay(): HTMLDivElement {
  const overlay = document.body.querySelector('.overlay-harness')
  expect(overlay).toBeTruthy()
  return overlay as HTMLDivElement
}

function getContent(): HTMLDivElement {
  const content = document.body.querySelector('.overlay-content')
  expect(content).toBeTruthy()
  return content as HTMLDivElement
}

function dispatchMouseEvent(target: Element | Document, type: 'mousedown' | 'mouseup') {
  target.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true }))
}

afterEach(() => {
  while (mountedWrappers.length > 0) {
    mountedWrappers.pop()?.unmount()
  }
  document.body.innerHTML = ''
})

describe('useOverlayDismiss', () => {
  it('dismisses only when pointer press and release both happen on the overlay', () => {
    const wrapper = mountHarness()
    const overlay = getOverlay()

    dispatchMouseEvent(overlay, 'mousedown')
    dispatchMouseEvent(overlay, 'mouseup')

    expect(wrapper.emitted('dismiss')).toHaveLength(1)
  })

  it('does not dismiss when pointer press starts inside content and releases on the overlay', () => {
    const wrapper = mountHarness()
    const overlay = getOverlay()
    const content = getContent()

    dispatchMouseEvent(content, 'mousedown')
    dispatchMouseEvent(overlay, 'mouseup')

    expect(wrapper.emitted('dismiss')).toBeUndefined()
  })

  it('does not dismiss when disabled', () => {
    const wrapper = mountHarness({ enabled: false })
    const overlay = getOverlay()

    dispatchMouseEvent(overlay, 'mousedown')
    dispatchMouseEvent(overlay, 'mouseup')

    expect(wrapper.emitted('dismiss')).toBeUndefined()
  })

  it('clears a pending overlay press when dismissal becomes disabled before mouseup', async () => {
    const wrapper = mountHarness()
    const overlay = getOverlay()

    dispatchMouseEvent(overlay, 'mousedown')
    await wrapper.setProps({ enabled: false })
    dispatchMouseEvent(overlay, 'mouseup')

    expect(wrapper.emitted('dismiss')).toBeUndefined()
  })
})
