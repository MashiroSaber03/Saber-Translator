import { RequestFailure, saberResponse } from '../api'

export async function downloadOriginals(sessionId: string, title: string): Promise<void> {
  let response: Response
  try {
    response = await saberResponse(`/sessions/${encodeURIComponent(sessionId)}/originals.zip`, {}, undefined, 120_000)
  } catch (error) {
    if (error instanceof RequestFailure && (error.code === 'http_404'
      || (error.code === 'not_found' && error.message === 'not found'))) {
      throw new Error('本机 Saber 尚未提供原图下载接口。请在 GUI「概览」停止并重新启动后端，然后重试。')
    }
    throw error
  }
  const blob = await response.blob()
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `${title.replace(/[<>:"/\\|?*\u0000-\u001f]/g, '_').slice(0, 100).trim() || '漫画原图'}.zip`
  document.body.append(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 60_000)
}
