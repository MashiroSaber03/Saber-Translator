import init from './photocraft.js'
try {
  await init()
} catch (error) {
  const message = `编辑内核加载失败：${error.message}。请刷新重试。`
  document.getElementById('photocraft_loading').textContent = message
  parent.postMessage({ saberEditor: true, event: 'error', error: message }, location.origin)
}
