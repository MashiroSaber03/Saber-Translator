# Paint.NET documents

PhotoCraft opens PDN3 `.pdn` files natively, without Paint.NET, Python, .NET or an external
converter. File › Open, recent files, drops, the CLI and automation use the shared importer.
The browser file picker also accepts `.pdn`.

Each bitmap layer remains editable. Layer order, Unicode names, visibility, opacity and pixel
alpha are retained. Painting, masks, transforms, reordering, duplication and undo use the same
commands as other pixel layers. All 14 PDN3 blend modes are supported on the CPU and GPU.
Reflect, Glow, Negation and XOR have no Photoshop equivalent, and Color Burn and Color Dodge have
separate Paint.NET variants because their edge cases differ. The blend menus stay Photoshop's: a
layer that uses one of these six modes lists its own mode after Photoshop's, and other layers don't
show them.

PDN support is **import only**. Save opens Save As and suggests `.pcraft`, which preserves the
layers and every blend mode. Exporting as `.pdn` fails with an explanatory error. PSD cannot
represent six Paint.NET variants; exporting those as PSD reports that their blending was
changed to Normal. Use `.pcraft` for continued editing, and PNG for a flattened rendering.

The reader supports PDN3's typed NRBF objects, both numeric and older blend-operation records,
BGRA32 surfaces with padded rows, shared pixel memory, and gzip or raw numbered pixel chunks.
It validates references, dimensions, strides, counts, chunk ranges and gzip checksums. Metadata
object nesting is capped at 64, layers at 1024 and total decoded pixel memory at 1 GiB. Imports
can be cancelled while parsing, decompressing or copying pixels. Unsupported layouts and
versions return errors. Paint.NET document metadata (including print resolution and profiles)
is currently omitted; nonempty user metadata produces an import warning. Layer data is preserved.

## Verification

Synthetic unit tests exercise every blend mode, names, hidden layers, opacity, alpha, row
padding, reversed chunks, old/new mode records, native saves, cancellation and malformed data.
The GPU parity tests include the additional modes. Local real-file checks are opt-in:

```sh
PDN_TEST_DIR=/path/to/pdn-and-png-exports cargo test -p photocraft-io --test pdn_corpus -- --ignored --nocapture
PDN_TEST_FILE=/path/to/layered.pdn cargo test -p photocraft-engine --test pdn -- --ignored
```

Each PDN in `PDN_TEST_DIR` needs a corresponding `.png` export with the same basename. The
test checks pixels against that export and verifies layer preservation through `.pcraft`.
Keep real-file corpora outside the application repository.

The local Paint.NET 4.0.10 corpus (14 small texture documents, 1–8 layers) also matches its
embedded full-size previews within 2.142/255. Older Paint.NET rounded between layers; PhotoCraft
keeps floating-point precision, so layered composites are not necessarily byte-identical.
A supplied 5.x document (1280×720, 11 layers, header version `5.3.8488.42200`) passes editing,
painting, undo/redo and native-save checks. Its embedded 256×144 preview matches a Catmull–Rom
downsample of our rendering with mean error 0.418/255 and maximum error 8/255. Its neighboring
PNG differs from the saved PDN preview, so it is unsuitable as a full-size rendering oracle.
These checks do not establish compatibility with every Paint.NET release.

Format references: [MS-NRBF](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-nrbf/)
and the [pypdn reader](https://github.com/addisonElliott/pypdn/blob/master/pypdn/reader.py).
This reader implements the layout independently; it does not run .NET deserialization.
