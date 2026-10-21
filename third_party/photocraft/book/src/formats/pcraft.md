# The `.pcraft` format

`photocraft-format` stores the native lossless document as a ZIP archive or directory bundle. Its main entries are:

```text
manifest.json
tiles/<blake3>.zst
blobs/<blake3>.zst
thumb.png
composite/preview.png
```

The manifest is versioned and describes the document tree. Tiles and blobs are content-addressed. Loaders verify ZIP entry bounds and CRCs; document content also uses stored hashes to detect mismatched or corrupted tile/blob content. Unsupported newer format versions fail explicitly.

## Implemented load limits

Default `LoadOptions` apply these maximums:

| Limit | Default |
|---|---:|
| `manifest.json` | 256 MiB |
| One decompressed blob | 1 GiB |
| Total decompressed tiles and blobs | 16 GiB |

Layer groups may nest at most 100 levels (`MAX_GROUP_DEPTH`). Saving a deeper document fails with an error instead of writing a bundle that can't be opened, and a manifest whose JSON nesting goes beyond what that depth needs is rejected before it is parsed, so a hostile file can't exhaust the stack.

The ZIP reader bounds decompression by the caller-supplied maximum, rejects encrypted entries and unsupported compression methods, validates offsets with checked arithmetic, and rejects CRC or declared-size mismatches. Corruption and property tests cover truncation, byte mutation, missing content, invalid manifests, excessive sizes, and random bytes.

These defaults are intentionally finite but can still be expensive on constrained systems. Future hardening should add context-sensitive budgets and explicit limits for entry count, document complexity, and cumulative work, then test directory bundles against symlink and replacement races.

When fields are added to `photocraft-doc`, `photocraft-format` intentionally requires corresponding manifest and conversion updates so native saves do not silently drop new state.
