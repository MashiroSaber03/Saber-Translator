# Affinity documents

PhotoCraft opens native Affinity documents: `.af` files from Affinity 3 and `.afdesign`,
`.afphoto` and `.afpub` files from Affinity 1 and 2. Layers come in as editable PhotoCraft layers:

- pages and artboards become artboards (Publisher spreads sit side by side);
- layers and groups become groups;
- curves and shapes (rectangles, ellipses, polygons, stars and the other geometric shapes) become
  shape layers with their fill, gradient and stroke;
- artistic and frame text becomes type layers, drawn with the fonts installed on your computer;
- placed JPEG and PNG images become embedded smart objects, and pixel layers become pixel layers;
- vector and pixel masks, opacity, visibility, locks, names and blend modes are kept.

Artboards are recognized from both the legacy flag and current Affinity 3 artboard properties.
They keep their names, bounds and editable children when saved as `.pcraft` or PSD; moving an
artboard moves its children together. Nested boards become masked groups, and rotated or curved
boards currently use their bounding rectangle.

The document is 8-bit RGB at the Affinity document's resolution. What PhotoCraft can't reproduce is
listed in a warning when the file opens, for example layer effects, adjustment layers and live
filters, brush strokes (drawn as plain strokes), special shapes such as clouds and hearts (drawn as
ellipses), master pages, and CMYK or Lab colour (converted to RGB without the document's profile).
Nothing is dropped without a warning.

PhotoCraft can't write Affinity files. An opened Affinity document has no save path: Save asks for
a new file (`.pcraft`, PSD, PNG…) and never writes over the Affinity file.

If the native data can't be read (a damaged file, or a layout PhotoCraft doesn't know), PhotoCraft
opens the small PNG preview Affinity stores in the file instead, as a single pixel layer named
“Affinity preview”, and the warning says why. That preview can't be placed or used by File ▸ Place,
stacks, batch processing or other imports that can't show the warning. Export PSD or PNG from
Affinity in that case.

See [the crate's notes](https://github.com/storytold/photocraft/blob/main/crates/affinity/README.md)
for the full list of what is imported, how the reader was written and validated, and its safety
limits.
