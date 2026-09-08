export const HOST_STYLES = `
:host { all: initial; position: fixed; inset: 0; pointer-events: none; z-index: 2147483600; color-scheme: light dark; }
* { box-sizing: border-box; }
.saber-fab { position: fixed; right: 22px; bottom: 24px; width: 48px; height: 48px; border: 1px solid #ffffff99; border-radius: 17px; background: linear-gradient(145deg,#f378aa,#da326f); color: #fff; font: italic 700 27px Georgia,serif; box-shadow: 0 5px 20px #a42a6250; cursor: pointer; pointer-events: auto; touch-action: none; }
.saber-fab:hover { filter: brightness(1.08); }
.saber-fab:focus-visible { outline: 3px solid #f69abd; outline-offset: 3px; }
.saber-fab[data-state=busy]::after,.saber-fab[data-state=error]::after { content:''; position:absolute; right:2px; top:2px; width:9px; height:9px; border:2px solid white; border-radius:50%; background:#ffd37e; }
.saber-fab[data-state=error]::after { background:#b61e45; }
.saber-panel { display:none; position:fixed; width:min(380px,calc(100vw - 16px)); height:min(680px,calc(100dvh - 100px)); border-radius:22px; overflow:hidden; box-shadow:0 14px 60px #49233730,0 0 0 1px #d891af50; pointer-events:auto; background:#fffafc; }
.saber-panel[data-open=true] { display:block; }
.saber-drag-mask { display:none; position:fixed; inset:0; pointer-events:auto; cursor:grabbing; touch-action:none; }
.saber-drag-mask[data-open=true] { display:block; }
iframe { display:block; width:100%; height:100%; border:0; }
.saber-pick-mask { display:none; position:fixed; inset:0; pointer-events:auto; cursor:crosshair; background:#e8418208; }
.saber-pick-mask span { position:fixed; left:50%; top:20px; transform:translateX(-50%); padding:14px 22px; border-radius:16px; background:#fff6fb; color:#963259; box-shadow:0 5px 30px #49233730; font:500 14px/1.5 'Segoe UI','Microsoft YaHei',sans-serif; pointer-events:none; }
.saber-pick-mask[data-open=true] { display:block; }
@media(prefers-color-scheme:dark) { .saber-panel { background:#211a20; } .saber-pick-mask span { background:#33232e; color:#ffa6cb; } }
`
