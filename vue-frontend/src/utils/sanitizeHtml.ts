const REMOVED_CONTENT_TAGS = new Set(['script', 'style', 'template', 'iframe', 'object', 'embed', 'svg', 'math'])

const ALLOWED_TAGS = new Set([
  'a',
  'blockquote',
  'br',
  'code',
  'del',
  'em',
  'h1',
  'h2',
  'h3',
  'h4',
  'h5',
  'h6',
  'hr',
  'li',
  'ol',
  'p',
  'pre',
  'strong',
  'table',
  'tbody',
  'td',
  'th',
  'thead',
  'tr',
  'ul',
])

function isSafeUrl(value: string): boolean {
  const trimmed = value.trim()
  if (!trimmed) return false
  if (trimmed.startsWith('#')) return true

  try {
    const url = new URL(trimmed, window.location.href)
    return url.protocol === 'http:' || url.protocol === 'https:' || url.protocol === 'mailto:'
  } catch {
    return false
  }
}

function sanitizeAttributes(element: Element, tagName: string): void {
  const href = element.getAttribute('href') || ''

  for (const attribute of [...element.attributes]) {
    element.removeAttribute(attribute.name)
  }

  if (tagName !== 'a') return

  if (isSafeUrl(href)) {
    element.setAttribute('href', href.trim())
    element.setAttribute('rel', 'noopener noreferrer')
  }
}

function sanitizeChildren(parent: ParentNode): void {
  for (const child of [...parent.childNodes]) {
    if (child.nodeType !== Node.ELEMENT_NODE) continue

    const element = child as Element
    const tagName = element.tagName.toLowerCase()

    if (REMOVED_CONTENT_TAGS.has(tagName)) {
      element.remove()
      continue
    }

    sanitizeChildren(element)

    if (!ALLOWED_TAGS.has(tagName)) {
      element.replaceWith(...element.childNodes)
      continue
    }

    sanitizeAttributes(element, tagName)
  }
}

export function sanitizeHtml(html: string): string {
  if (!html) return ''

  const template = document.createElement('template')
  template.innerHTML = html
  sanitizeChildren(template.content)
  return template.innerHTML
}
