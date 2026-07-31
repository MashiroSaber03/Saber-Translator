import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import VirtualPageStream from '@/components/virtual/VirtualPageStream.vue'

describe('VirtualPageStream', () => {
  it('reserves each page aspect ratio before its lazy image loads', () => {
    const wrapper = mount(VirtualPageStream, {
      props: {
        items: [{
          alt: '第 1 页',
          height: 1200,
          id: 'page-1',
          url: '/api/v2/assets/page-1',
          width: 800,
        }],
      },
    })

    expect(wrapper.get('figure').attributes('style')).toContain('aspect-ratio: 800 / 1200')
    expect(wrapper.get('img').attributes()).toMatchObject({
      loading: 'lazy',
      decoding: 'async',
    })
  })

  it('reports pages from the actual viewport instead of the overscan margin', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/virtual/VirtualPageStream.vue'),
      'utf8',
    )

    expect(source).toContain('}, { root })')
    expect(source).not.toContain("rootMargin: '200%")
  })
})
