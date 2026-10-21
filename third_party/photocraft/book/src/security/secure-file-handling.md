# Secure file handling

PhotoCraft reads and writes documents in the desktop app, CLI, headless RPC server, and MCP server.
Remote automation paths are untrusted relative names resolved beneath explicitly granted roots.

## Implemented

- Format loaders return structured errors for I/O and malformed data.
- `.pcraft` ZIP loading validates archive offsets, entry sizes, CRCs, content hashes, and configured decompressed-size budgets.
- Raster and PSD loaders apply parser/decode limits before major pixel allocation.
- Native `.pcraft` saves use temporary-file/rename patterns in relevant store paths.
- `AuthorizedWorkspace` holds independent read and write directory capabilities for remote
  automation.
- Absolute paths, parent traversal, alternate separators, drive/device/stream prefixes, reserved
  Windows device names, empty or dot components, and link escapes are rejected before file
  effects.
- Non-existent output files are supported when their parent already exists beneath the write
  root. Automation does not create caller-selected directory trees.
- Filesystem-bearing engine commands that have not migrated to capability I/O fail closed for
  automation. Trusted local CLI subcommands and user-driven desktop file pickers keep their
  intended ambient authority.

## Known limitations

- Directory `.pcraft` bundles have not been documented as resistant to link replacement or time-of-check/time-of-use races.
- Root grants are process launch configuration, not per-document grants.
- General method capabilities and structured filesystem audit events are not implemented.
- Existing hard links inside a granted root are trusted root contents; the path policy cannot
  infer how they were created.

## Capability-based design

Prefer an opened directory capability or workspace handle over repeated string sanitization:

```text
AuthorizedWorkspace
  - open_document(relative_path)
  - save_document(relative_path)
  - export_render(relative_path)
```

The handle is created from an explicit launch-time grant, preserves separate read/write
permissions, rejects absolute and escaping paths, and performs operations relative to the held
directory capability. New files require an existing in-root parent because a non-existent target
cannot be canonicalized safely.

## Design requirements

- Canonicalization alone is insufficient: links can change after validation.
- Policy should operate on handles and relative components where platform APIs allow it.
- Reads and writes should have distinct capabilities.
- Replace, create-new, truncate, and directory creation semantics must be explicit.
- Logs should avoid file contents and should minimize disclosure of full private paths.
- Tests should cover `..`, absolute paths, mixed separators, case behavior, symlinks/junctions, non-existent targets, and replacement races on supported platforms.

OS-account, VM, container, or sandbox isolation remains appropriate defense in depth for
untrusted automation and for controls not yet covered by general method capabilities.
