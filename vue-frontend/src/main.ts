import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
import { showToast } from './utils/toast'

import './styles/tokens/foundation.css'
import './styles/tokens/semantic.css'
import './styles/tokens/component.css'
import './styles/tokens/domain.css'
import './styles/reset.css'

const app = createApp(App)

const pinia = createPinia()
app.use(pinia)

app.use(router)

app.config.errorHandler = (err, _instance, info) => {
  const message = err instanceof Error ? err.message : String(err)
  showToast(`应用运行出错：${message || info}`, 'error', 5000)
}

app.mount('#app')
