export const UI_STYLES = `
:host {
  --saber-pink: #f43f8c;
  --saber-pink-strong: #db2777;
  --saber-pink-soft: #fff1f7;
  --saber-surface: rgba(255, 255, 255, 0.98);
  --saber-surface-raised: #ffffff;
  --saber-text: #241b22;
  --saber-muted: #766a72;
  --saber-border: rgba(244, 63, 140, 0.18);
  --saber-shadow: 0 18px 48px rgba(74, 23, 51, 0.2);
  all: initial;
  position: fixed;
  inset: 0;
  display: block;
  z-index: 2147483646;
  pointer-events: none;
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :host {
    --saber-pink-soft: #341825;
    --saber-surface: rgba(30, 25, 29, 0.98);
    --saber-surface-raised: #292127;
    --saber-text: #fff7fb;
    --saber-muted: #c5b6bf;
    --saber-border: rgba(251, 113, 178, 0.24);
    --saber-shadow: 0 18px 48px rgba(0, 0, 0, 0.44);
  }
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
button, input, select { font: inherit; }
.saber-root {
  position: absolute; inset: 0; pointer-events: none;
  color: var(--saber-text);
  font: 400 13px/1.45 Inter, "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
}
.saber-fab {
  position: fixed; right: 16px; bottom: 18px; width: 46px; height: 46px;
  display: grid; place-items: center; border: 0; border-radius: 15px;
  color: white; background: linear-gradient(145deg, #fb5ba2, var(--saber-pink-strong));
  box-shadow: 0 10px 24px rgba(219, 39, 119, .34); cursor: pointer;
  pointer-events: auto; cursor: grab; touch-action: none; transition: transform .16s ease, box-shadow .16s ease;
}
.saber-fab[data-dragging="true"] { cursor: grabbing; }
.saber-fab:hover { transform: translateY(-2px); box-shadow: 0 14px 28px rgba(219, 39, 119, .43); }
.saber-fab:focus-visible, button:focus-visible, select:focus-visible, input:focus-visible, summary:focus-visible {
  outline: 3px solid rgba(244, 63, 140, .26); outline-offset: 2px;
}
.saber-fab__glyph { font: 800 17px/1 Georgia, serif; letter-spacing: -.06em; }
.saber-fab__dot {
  position: absolute; top: -2px; right: -2px; width: 11px; height: 11px;
  border: 2px solid var(--saber-surface-raised); border-radius: 50%; background: #22c55e;
}
.saber-fab[data-state="busy"] .saber-fab__dot { background: #fbbf24; animation: saberPulse 1.2s infinite; }
.saber-fab[data-state="error"] .saber-fab__dot { background: #ef4444; }
@keyframes saberPulse { 50% { opacity: .45; transform: scale(.78); } }
.saber-panel {
  position: fixed; right: 16px; bottom: 76px; width: min(336px, calc(100vw - 24px));
  max-height: min(580px, calc(100vh - 32px)); display: none; flex-direction: column;
  overflow: hidden; pointer-events: auto; color: var(--saber-text); background: var(--saber-surface);
  border: 1px solid var(--saber-border); border-radius: 20px; box-shadow: var(--saber-shadow);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
}
.saber-panel[data-open="true"] { display: flex; }
.saber-header {
  display: flex; align-items: center; gap: 10px; padding: 13px 14px 11px;
  border-bottom: 1px solid var(--saber-border); cursor: move; touch-action: none; user-select: none;
}
.saber-logo { width: 30px; height: 30px; display: grid; place-items: center; flex: 0 0 auto; border-radius: 10px; color: white; font-weight: 800; background: linear-gradient(145deg, #fb5ba2, var(--saber-pink-strong)); }
.saber-heading { min-width: 0; flex: 1; }
.saber-title { margin: 0; font-size: 15px; font-weight: 750; line-height: 1.2; }
.saber-subtitle { margin-top: 2px; overflow: hidden; color: var(--saber-muted); font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
.saber-icon-button { width: 30px; height: 30px; padding: 0; border: 0; border-radius: 9px; color: var(--saber-muted); background: transparent; cursor: pointer; }
.saber-icon-button:hover { color: var(--saber-text); background: var(--saber-pink-soft); }
.saber-body { min-height: 0; display: flex; flex-direction: column; gap: 11px; overflow: auto; padding: 12px 13px 14px; }
.saber-panel[data-view="candidates"] .saber-body { overflow: hidden; }
.saber-banner { display: flex; gap: 9px; padding: 10px 11px; flex: 0 0 auto; border: 1px solid var(--saber-border); border-radius: 13px; background: var(--saber-pink-soft); }
.saber-banner__dot { width: 8px; height: 8px; margin-top: 5px; flex: 0 0 auto; border-radius: 50%; background: var(--saber-pink); }
.saber-banner__text { min-width: 0; font-size: 12px; line-height: 1.45; }
.saber-banner__text strong { display: block; margin-bottom: 1px; font-size: 13px; }
.saber-banner[data-tone="error"] { border-color: rgba(239, 68, 68, .28); background: rgba(239, 68, 68, .09); }
.saber-banner[data-tone="error"] .saber-banner__dot { background: #ef4444; }
.saber-error-actions { display: flex; justify-content: flex-end; margin-top: -4px; }
.saber-settings { flex: 0 0 auto; padding: 0 10px 10px; border: 1px solid var(--saber-border); border-radius: 13px; background: var(--saber-surface-raised); }
.saber-settings__summary { padding: 10px 0 0; color: var(--saber-text); font-size: 12px; font-weight: 700; cursor: pointer; }
.saber-settings__summary::marker { color: var(--saber-pink); }
.saber-settings[open] .saber-settings__summary { margin-bottom: 10px; }
.saber-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
.saber-field { position: relative; display: grid; gap: 5px; min-width: 0; }
.saber-field::after { content: ""; position: absolute; right: 12px; bottom: 14px; width: 6px; height: 6px; pointer-events: none; border-right: 1.5px solid var(--saber-muted); border-bottom: 1.5px solid var(--saber-muted); transform: rotate(45deg); }
.saber-label { color: var(--saber-muted); font-size: 11px; font-weight: 650; }
.saber-select {
  width: 100%; height: 36px; padding: 0 28px 0 10px; appearance: none; -webkit-appearance: none;
  border: 1px solid var(--saber-border); border-radius: 10px; color: var(--saber-text);
  background: var(--saber-surface); font-size: 13px; cursor: pointer;
}
.saber-check { display: flex; align-items: center; gap: 7px; min-height: 30px; color: var(--saber-text); font-size: 12px; cursor: pointer; }
.saber-check input { width: 15px; height: 15px; margin: 0; accent-color: var(--saber-pink); }
.saber-actions { display: flex; flex-wrap: wrap; gap: 7px; flex: 0 0 auto; }
.saber-actions--end { justify-content: flex-end; }
.saber-idle-actions .saber-button--primary { flex: 1; }
.saber-button {
  min-height: 34px; padding: 0 11px; border: 1px solid var(--saber-border); border-radius: 10px;
  color: var(--saber-text); background: var(--saber-surface-raised); font-size: 12px; font-weight: 700; cursor: pointer;
}
.saber-button:hover { border-color: rgba(244, 63, 140, .42); background: var(--saber-pink-soft); }
.saber-button--primary { color: white; border-color: transparent; background: linear-gradient(145deg, #fb5ba2, var(--saber-pink-strong)); }
.saber-button--primary:hover { background: linear-gradient(145deg, #f43f8c, #be185d); }
.saber-button--quiet { min-height: 30px; padding: 0 9px; color: var(--saber-muted); background: transparent; }
.saber-button--danger { color: #ef4444; }
.saber-button:disabled { opacity: .48; cursor: not-allowed; }
.saber-section { min-height: 0; padding-top: 11px; border-top: 1px solid var(--saber-border); }
.saber-section__title { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 9px; font-size: 13px; font-weight: 750; }
.saber-section__tools { display: flex; align-items: center; gap: 7px; }
.saber-counter { color: var(--saber-muted); font-size: 11px; font-weight: 550; }
.saber-link-button { padding: 0; border: 0; color: var(--saber-pink-strong); background: transparent; font-size: 11px; font-weight: 700; cursor: pointer; }
.saber-candidate-section { display: flex; flex: 1 1 auto; flex-direction: column; }
.saber-candidates { display: flex; flex-wrap: wrap; align-content: flex-start; gap: 7px; min-height: 150px; max-height: min(280px, 34vh); overflow: auto; padding-right: 2px; }
.saber-candidate { position: relative; overflow: hidden; flex: 0 0 calc((100% - 14px) / 3); min-width: 0; aspect-ratio: .76; contain: layout paint; border: 1px solid var(--saber-border); border-radius: 10px; background: var(--saber-surface-raised); }
.saber-candidate img { width: 100%; height: 100%; display: block; object-fit: cover; }
.saber-candidate__fallback { height: 100%; display: grid; place-items: center; padding: 7px; color: var(--saber-muted); font-size: 11px; text-align: center; }
.saber-candidate input { position: absolute; top: 6px; left: 6px; width: 16px; height: 16px; margin: 0; accent-color: var(--saber-pink); }
.saber-candidate__meta { position: absolute; inset: auto 4px 4px; padding: 2px 4px; border-radius: 6px; color: white; background: rgba(0,0,0,.62); font-size: 10px; text-align: center; }
.saber-candidate-actions { margin-top: 9px; justify-content: space-between; }
.saber-progress-section { flex: 0 0 auto; }
.saber-preparation { margin-bottom: 11px; padding: 10px 11px; border-radius: 11px; background: var(--saber-pink-soft); }
.saber-preparation__heading { display: flex; align-items: center; justify-content: space-between; gap: 8px; font-size: 12px; font-weight: 700; }
.saber-preparation__count { color: var(--saber-pink-strong); font-size: 11px; font-variant-numeric: tabular-nums; }
.saber-preparation__meter { display: block; width: 100%; height: 7px; margin: 8px 0 6px; overflow: hidden; appearance: none; border: 0; border-radius: 999px; background: rgba(244, 63, 140, .16); }
.saber-preparation__meter::-webkit-progress-bar { border-radius: 999px; background: rgba(244, 63, 140, .16); }
.saber-preparation__meter::-webkit-progress-value { border-radius: 999px; background: linear-gradient(90deg, #fb5ba2, var(--saber-pink-strong)); transition: width .16s ease; }
.saber-preparation__meter::-moz-progress-bar { border-radius: 999px; background: linear-gradient(90deg, #fb5ba2, var(--saber-pink-strong)); }
.saber-preparation__detail { color: var(--saber-muted); font-size: 10px; font-variant-numeric: tabular-nums; }
.saber-progress { display: grid; grid-template-columns: repeat(5, 1fr); gap: 5px; }
.saber-stat { padding: 7px 2px; border-radius: 9px; background: var(--saber-pink-soft); text-align: center; }
.saber-stat strong { display: block; font-size: 14px; line-height: 1.2; }
.saber-stat span { color: var(--saber-muted); font-size: 10px; }
.saber-page-actions { margin-top: 8px; }
.saber-page-actions__summary { cursor: pointer; color: var(--saber-muted); font-size: 11px; font-weight: 700; }
.saber-page-actions__list { display: grid; max-height: 176px; gap: 6px; margin-top: 7px; overflow-y: auto; }
.saber-page-action { display: flex; align-items: center; gap: 7px; padding: 7px 8px; border-radius: 9px; background: rgba(244, 63, 139, .06); font-size: 11px; }
.saber-page-action--error { color: #ef4444; background: rgba(239, 68, 68, .08); }
.saber-page-action__label { min-width: 0; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.saber-page-action__buttons { display: flex; flex: 0 0 auto; gap: 5px; }
.saber-page-action .saber-button { min-height: 28px; padding: 0 8px; }
.saber-progress-actions { margin-top: 10px; }
.saber-terms-section { flex: 0 0 auto; }
.saber-terms-summary { color: var(--saber-text); font-size: 12px; font-weight: 700; cursor: pointer; }
.saber-terms { max-height: 112px; margin-top: 8px; overflow: auto; color: var(--saber-muted); font-size: 12px; line-height: 1.55; }
.saber-import-overlay {
  position: fixed; inset: 0; display: none; place-items: center; padding: 14px;
  pointer-events: auto; background: rgba(35, 18, 28, .34); backdrop-filter: blur(4px);
  -webkit-backdrop-filter: blur(4px);
}
.saber-import-overlay[data-open="true"] { display: grid; }
.saber-import-dialog {
  width: min(360px, calc(100vw - 28px)); max-height: calc(100vh - 28px); overflow: auto;
  padding: 15px; border: 1px solid var(--saber-border); border-radius: 18px;
  color: var(--saber-text); background: var(--saber-surface-raised); box-shadow: var(--saber-shadow);
}
.saber-import-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
.saber-import-title { margin: 0; font-size: 16px; line-height: 1.3; }
.saber-import-summary { margin: 4px 0 0; color: var(--saber-muted); font-size: 11px; line-height: 1.5; }
.saber-import-destinations { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 14px; }
.saber-import-choice {
  display: flex; align-items: center; gap: 7px; min-height: 40px; padding: 0 10px;
  border: 1px solid var(--saber-border); border-radius: 11px; color: var(--saber-text);
  background: var(--saber-surface); font-size: 12px; font-weight: 700; cursor: pointer;
}
.saber-import-choice:has(input:checked) { border-color: var(--saber-pink); background: var(--saber-pink-soft); }
.saber-import-choice:has(input:disabled) { opacity: .46; cursor: not-allowed; }
.saber-import-choice input { width: 15px; height: 15px; margin: 0; accent-color: var(--saber-pink); }
.saber-import-fields { display: grid; gap: 9px; margin-top: 12px; }
.saber-input-field { display: grid; gap: 5px; margin-top: 12px; }
.saber-import-fields .saber-input-field { margin-top: 0; }
.saber-input {
  width: 100%; height: 38px; padding: 0 10px; border: 1px solid var(--saber-border);
  border-radius: 10px; color: var(--saber-text); background: var(--saber-surface); font-size: 12px;
}
.saber-input::placeholder { color: var(--saber-muted); }
.saber-import-select-wrap { position: relative; }
.saber-import-select-wrap::after {
  content: ""; position: absolute; right: 13px; top: 14px; width: 6px; height: 6px;
  pointer-events: none; border-right: 1.5px solid var(--saber-muted);
  border-bottom: 1.5px solid var(--saber-muted); transform: rotate(45deg);
}
.saber-import-book-select { padding-right: 30px; appearance: none; -webkit-appearance: none; cursor: pointer; }
.saber-import-actions { margin-top: 15px; }
.saber-pick-mask { position: fixed; inset: 0; display: none; pointer-events: auto; cursor: crosshair; background: rgba(244, 63, 140, .04); }
.saber-pick-mask[data-open="true"] { display: block; }
.saber-pick-tip { position: fixed; left: 50%; top: 18px; transform: translateX(-50%); padding: 9px 13px; border-radius: 11px; color: white; background: #be185d; box-shadow: var(--saber-shadow); font-size: 12px; font-weight: 700; }
@media (max-width: 520px) {
  .saber-fab { right: 12px; bottom: 14px; }
  .saber-panel { right: 12px; bottom: 68px; width: min(336px, calc(100vw - 16px)); }
}
@media (max-height: 520px) {
  .saber-panel { max-height: calc(100vh - 16px); }
  .saber-candidates { min-height: 110px; max-height: 28vh; }
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
`
