# Batch output naming

Select a recorded action, then open **File > Automate > Batch**. Set the input and output
folders. **File Naming** accepts a filename stem template; the selected export format supplies
the extension. Leave it blank to keep the existing source filenames and collision reporting.

| Token | Value |
| --- | --- |
| `{name}` | Input filename without its last extension |
| `{serial}` | Input position plus **Starting serial** (default 1), padded to at least three digits |
| `{date}` | UTC date at the start of the batch, in `YYYY-MM-DD` form |

For example, `{name}_{serial}` with starting serial 7 writes `photo_007.png` and
`photo_008.png` for two PNG inputs named `photo.png`. A serial after 999 becomes 1000;
it does not wrap. Folder inputs use the existing filename order, and an explicit input array
keeps its order. Failed inputs consume their position, so later filenames do not shift.
Braces in an input filename are literal text, not additional tokens.

Unknown tokens, unmatched braces, invalid parameter types, path separators, reserved filename
characters, excessive filename length (255 UTF-8 bytes), and serial overflow are rejected before
export writes begin. Starting serial must be an integer from 1 to 4,294,967,295.
A custom template that resolves to an existing input file is also rejected; choose a separate
destination or another template. Ordinary per-file read, action and save failures still appear
in the batch's `errors` result. When two results share a name, including a case-only difference,
the first successful result is kept and the later collision is reported.

The same options are available on `file.automate.batch` through the engine command interface:

```json
{
  "input": ["first/photo.png", "second/photo.png"],
  "output": "exports",
  "steps": [["image.adjustments.invert", {}]],
  "format": "png",
  "naming": "{name}_{serial}",
  "startingSerial": 7
}
```

The CLI can invoke this command with `run --cmd file.automate.batch --params ...`.
This option belongs to Batch; Image Processor and droplet options retain their existing behavior.
