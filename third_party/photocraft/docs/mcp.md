# MCP conventions

PhotoCraft follows the same conventions as FilmCraft #28. Start a headless session:

```sh
photocraft-cli mcp --automation-read-root /work/project --automation-write-root /work/project
```

Paths are relative to the corresponding root, or absolute paths beneath it; without a root that
filesystem authority is absent. Bridge mode uses `--bridge` and `--control-token-file` as
described in [development.md](development.md). UI tools require a running desktop app.

## Tools and resources

- `command_list`: discover engine ids, parameter docs and current enabled state; accepts
  `filter` and `enabled_only`. Returns a bare JSON array of
  `{id, label, menu, shortcut, params, enabled, disabledReason, needsDocument}` objects (see
  [Command list replies](control-protocol.md#command-list-replies)).
- `command_run`: `id`, optional `params` and `wait`. Command parameters remain engine-defined.
- `command_batch`: ordered `steps`, optional `stop_on_error` (default true); stops at the first
  error unless false. Not an atomic transaction: earlier successful steps stay applied, with
  no rollback. Existing per-step results and reply budgets are preserved.
- `doc_inspect`: layer tree, history and selection; an empty session returns `document:null`.
  Every layer (including groups and nested children) reports raster-only `hasMask`,
  `hasVectorMask` for vector mask presence, and `vectorMaskEnabled` (`true`/`false` when
  present, `null` when absent). A disabled vector mask still has `hasVectorMask:true`.
- `render_preview`: bounded PNG content; `index` and `max_side` (default 1024, maximum 2048).
- `ui_inspect` and `ui_screenshot`: live app state/window in bridge mode.

Existing document/session/job/UI tools remain listed, including the documented
`doc_render_preview` spelling. There are no hidden legacy aliases. Every tool has a title
and all four annotation hints. Generic command/control tools conservatively advertise writes;
file saves may replace existing targets. Hints do not grant permission.

After a failed batch, `completed` and per-step `results` report successful steps, not a
rolled-back transaction. Inspect the document and its history with `doc_inspect`; commands
do not necessarily add one history entry each. For undoable edits, use `command_run` with
`{"id":"edit.undo","params":{}}` once per undoable history entry added, inspecting history
as you recover. Undo does not roll back files written to disk. If a reply-budget error says
the current step may have completed, inspect its effects before retrying or undoing it.

Before experiments, use `command_run` with `image.duplicate` and `{}` params, or
`history.newDocument` and `{}` params (copies the current state by default). Both create a
new active document, leaving the source unchanged; run experimental edits on that copy.
The optional `state` parameter of `history.newDocument` selects a retained historical state.

Unknown tool arguments return JSON-RPC `-32602`, naming the key, before execution. Batch
step envelopes are strict too. Nested engine `params` retain their command-specific handling.
Tool execution failures return `isError:true`. A worker panic becomes a tool error and the
existing poisoned-session recovery keeps serving. Invalid JSON lines receive `-32700` with
`id:null`; the next line is still processed.

Embedded Smart Objects support `command_run` with `layer.smartObjects.editContents` and
`layer.smartObjects.convertToLayers`, optionally targeting `params.layer`. Use
`layer.smartObjects.saveContents` in an opened contents document to update its parent in memory;
save nested contents from the inside outward, then `doc_save` the parent to a scoped path.
`doc_save` on a contents document exports that document rather than updating its parent.
No filesystem root is needed for the in-memory commands. External linked sources that require
a file read remain refused. See the [control policy](control-protocol.md#engine-commands)
for the shared restrictions.

`photocraft://document` and `photocraft://commands` return live JSON matching `doc_inspect`
and `command_list`. Resource reads are uncached; tool/resource catalogs are private and cached
for ten minutes. List/read responses include the MCP 2026-07-28 result/cache fields.

## Progress and cancellation

A direct headless `command_run` of `file.export.renderVideo` reports frame counts when
`_meta.progressToken` is present (a string or number). Notifications increase strictly,
are throttled to at most ten per second except completion, and include `total`. No token
means no progress notifications. Ping remains responsive while rendering.

```json
{"jsonrpc":"2.0","id":20,"method":"tools/call","params":{"name":"command_run","arguments":{"id":"file.export.renderVideo","params":{"dir":"frames","format":"png"}},"_meta":{"progressToken":"export-20"}}}
{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":20}}
```

Create a document and timeline first (`doc_new`, then `timeline.create`). Export uses the
configured write root and creates `dir` beneath it. Filenames must be single components;
absolute paths, traversal and escaping symlinks are rejected. Existing target filenames
are refused, including symlinks, so a cancelled or failed job cannot replace an earlier export.
Only files actually created by this job are removed on cancellation, error or an escaped panic;
unrelated files and the destination directory remain. PNG sequences and animated GIF share
the engine renderer. The document and playhead are unchanged.

Cancellation suppresses the request response and stops at a frame boundary (the current
frame/encoding may finish first). Unknown request ids are ignored. Render Video stays a
synchronous command even with `wait:false`; other job-capable commands keep their existing
`jobs_list`/`jobs_cancel` behavior. Batch exports do not emit MCP progress or support per-step
MCP cancellation. Nested engine actions and bridge Render Video remain denied by the existing
ambient-filesystem policy; the new scoped writer is available for direct headless commands.
