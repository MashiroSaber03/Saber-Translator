<script setup lang="ts">
import BaseModal from '@/components/common/BaseModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import UiButton from '@/components/ui/UiButton.vue'

const wechatQrUrl = '/pic/wechat_qrcode.png'
const alipayQrUrl = '/pic/alipay_qrcode.png'

const emit = defineEmits<{
  close: []
}>()
</script>

<template>
  <BaseModal title="支持作者" @close="emit('close')">
    <div class="sponsor-modal__content">
      <p class="sponsor-modal__message">
        如果这个项目对你有帮助，欢迎支持作者继续维护。
      </p>

      <div class="sponsor-modal__qr-codes">
        <div class="sponsor-modal__qr-item">
          <div class="sponsor-modal__qr-image">
            <img class="sponsor-modal__qr-img" :src="wechatQrUrl" alt="微信支付" />
          </div>
          <span class="sponsor-modal__qr-label">微信支付</span>
        </div>

        <div class="sponsor-modal__qr-item">
          <div class="sponsor-modal__qr-image">
            <img class="sponsor-modal__qr-img" :src="alipayQrUrl" alt="支付宝" />
          </div>
          <span class="sponsor-modal__qr-label">支付宝</span>
        </div>
      </div>

      <p class="sponsor-modal__thanks">感谢您的支持！</p>
    </div>

    <template #footer>
      <ProductActionRow
        aria-label="赞助弹窗操作"
        variant="dialog"
      >
        <UiButton type="button" variant="primary" @click="emit('close')">关闭</UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>
</template>

<style scoped>
.sponsor-modal__content {
  text-align: center;
  padding: 20px 0;
}

.sponsor-modal__message {
  font-size: 16px;
  color: var(--color-text-default);
  margin-bottom: 24px;
}

.sponsor-modal__qr-codes {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 24px 40px;
  margin-bottom: 24px;
}

.sponsor-modal__qr-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.sponsor-modal__qr-image {
  width: 180px;
  height: 180px;
  padding: 10px;
  background: var(--color-surface-base);
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
}

.sponsor-modal__qr-img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.sponsor-modal__qr-label {
  font-size: 14px;
  color: var(--color-text-supporting);
  font-weight: 500;
}

.sponsor-modal__thanks {
  font-size: 14px;
  color: var(--color-text-supporting);
}

@media (--breakpoint-sm-down) {
  .sponsor-modal__qr-image {
    width: 150px;
    height: 150px;
  }
}
</style>
