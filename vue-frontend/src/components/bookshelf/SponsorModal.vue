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

</style>
