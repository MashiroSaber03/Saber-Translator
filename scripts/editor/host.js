import init from './photocraft.js'
try {
  await init()
} catch (error) {
  document.getElementById('photocraft_loading').textContent = `编辑内核加载失败：${error.message}。请刷新重试。`
}
