const MANGADEX_HOST = /^(?:www\.)?mangadex\.org$/i
const MANGADEX_CHAPTER_PAGE = /^\/chapter\/([^/]+)(?:\/\d+)?\/?$/i

function mangaDexChapterPath(url: URL): string | null {
  if (!MANGADEX_HOST.test(url.hostname)) return null
  const match = MANGADEX_CHAPTER_PAGE.exec(url.pathname)
  return match?.[1] ? `/chapter/${match[1]}` : null
}

export function normalizedTaskPageUrl(value: string): string {
  const url = new URL(value)
  url.hash = ''
  const chapterPath = mangaDexChapterPath(url)
  if (chapterPath) url.pathname = chapterPath
  return url.toString()
}

export function stablePageTitle(
  pageUrl: string,
  documentTitle: string,
  fallback: string,
): string {
  const title = documentTitle.trim()
  const url = new URL(pageUrl)
  if (mangaDexChapterPath(url)) {
    return title.replace(/^\d+\s*\|\s*/, '').trim() || fallback
  }
  return title || fallback
}
