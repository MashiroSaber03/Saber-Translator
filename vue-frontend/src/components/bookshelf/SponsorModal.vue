<script setup lang="ts">
/**
 * 赞助模态框组件
 * 显示微信和支付宝二维码
 */

import BaseModal from '@/components/common/BaseModal.vue'

const emit = defineEmits<{
  close: []
}>()
</script>

<template>
  <BaseModal title="支持作者" @close="emit('close')">
    <div class="sponsor-content">
      <p class="sponsor-message">
        如果这个项目对你有帮助，欢迎请作者喝杯咖啡 ☕
      </p>
      
      <div class="qr-codes">
        <!-- 微信支付 -->
        <div class="qr-item">
          <div class="qr-image">
            <img :src="'/pic/wechat_qrcode.png'" alt="微信支付" />
          </div>
          <span class="qr-label">微信支付</span>
        </div>
        
        <!-- 支付宝 -->
        <div class="qr-item">
          <div class="qr-image">
            <img :src="'/pic/alipay_qrcode.png'" alt="支付宝" />
          </div>
          <span class="qr-label">支付宝</span>
        </div>
      </div>
      
      <p class="sponsor-thanks">感谢您的支持！🙏</p>
    </div>

    <template #footer>
      <button type="button" class="btn btn-primary" @click="emit('close')">关闭</button>
    </template>
  </BaseModal>
</template>

<style scoped>
.sponsor-content {
  text-align: center;
  padding: 20px 0;
}

.sponsor-message {
  font-size: 16px;
  color: var(--text-primary, #333);
  margin-bottom: 24px;
}

.qr-codes {
  display: flex;
  justify-content: center;
  gap: 40px;
  margin-bottom: 24px;
}

.qr-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.qr-image {
  width: 180px;
  height: 180px;
  padding: 10px;
  background: #fff;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 8px;
}

.qr-image img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.qr-label {
  font-size: 14px;
  color: var(--text-secondary, #666);
  font-weight: 500;
}

.sponsor-thanks {
  font-size: 14px;
  color: var(--text-secondary, #666);
}

/* ===================================
   赞助模态框样式 - 完整迁移自 modals.css
   =================================== */

.donate-modal {
  display: none;
  position: fixed;
  z-index: 9999;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  overflow: auto;
  background-color: rgba(0,0,0,0.6);
  animation: fadeIn 0.3s;
}

.donate-modal-content {
  background-color: #fefefe;
  margin: 10% auto;
  padding: 25px;
  border-radius: 15px;
  box-shadow: 0 5px 20px rgba(0,0,0,0.2);
  width: 90%;
  max-width: 500px;
  text-align: center;
  position: relative;
}

[data-theme="dark"] .donate-modal-content {
  background-color: #2d3748;
  color: #f7fafc;
}

.donate-close {
  position: absolute;
  right: 15px;
  top: 10px;
  color: #aaa;
  font-size: 28px;
  font-weight: bold;
  cursor: pointer;
  transition: color 0.2s;
}

.donate-close:hover {
  color: #000;
}

[data-theme="dark"] .donate-close:hover {
  color: #fff;
}

.donate-modal-content h3 {
  margin-top: 5px;
  margin-bottom: 10px;
  color: #e91e63;
  font-size: 1.6em;
}

.donate-modal-content p {
  margin-bottom: 20px;
  font-size: 1.1em;
  color: #666;
}

[data-theme="dark"] .donate-modal-content p {
  color: #cbd5e0;
}

.donate-qrcodes {
  display: flex;
  justify-content: center;
  gap: 30px;
  margin-top: 20px;
}

.donate-qrcode-item {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.donate-qrcode-item img {
  width: 180px;
  height: 180px;
  border-radius: 10px;
  box-shadow: 0 3px 10px rgba(0,0,0,0.1);
}

.donate-qrcode-item p {
  margin-top: 10px;
  font-weight: bold;
  color: #333;
  font-size: 0.9em;
}

[data-theme="dark"] .donate-qrcode-item p {
  color: #e2e8f0;
}

@keyframes fadeIn {
  from {opacity: 0}
  to {opacity: 1}
}
</style>
