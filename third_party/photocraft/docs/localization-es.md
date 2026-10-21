# Spanish terminology

This note goes with the review of the Spanish catalog
(`crates/ui-egui/src/i18n/es.tsv`). The catalog already covered every measured key; the review
aligned its wording with the terminology a Spanish-speaking Photoshop user already knows, fixed
mistranslations, and made the register consistent. No proprietary translation resources, fonts,
screenshots or user documents are included: the terms are the ones a Spanish Photoshop user sees
in the product and its public help, and each was checked against the meaning of PhotoCraft's own
English label.

## Register and style

- **Formal "usted"**, as in Photoshop's Spanish UI: «Haga clic», «Seleccione», «¿Desea guardar…?».
  The catalog mixed «usted» and «tú»; every user-facing instruction now uses «usted».
- **Sentence case** after the first word, except panel and tool names, which keep their capital
  as in Photoshop: «Panel Carácter», «Herramienta Pluma».
- Quotation marks are «angulares»: «{name}».
- Neutral Spanish, matching Adobe's single Spanish catalog (for example «coma flotante»).

## Terminology

| English source | Spanish UI | Was |
|---|---|---|
| Artboard | Mesa de trabajo | Lienzo (clashed with *Canvas*) |
| Pattern (fill, overlay, stamp) | Motivo | Patrón |
| Preset | Ajuste preestablecido | Preajuste |
| Keyboard Shortcuts | Métodos abreviados de teclado | Atajos de teclado |
| Grid | Cuadrícula | Rejilla |
| Slice | Sector | Sección |
| Plug-ins | Plugins | Complementos |
| Scratch Disks | Discos de memoria virtual | Discos temporales |
| Puppet Warp | Deformación de posición libre | Deformación de clavos |
| Pin (Puppet Warp) | Chincheta | Pin |
| Dodge / Burn (tools) | Sobreexponer / Subexponer | **swapped** |
| Color Dodge / Color Burn | Sobreexposición / Subexposición de color | **swapped** |
| Linear Dodge (Add) / Linear Burn | Sobreexposición lineal (añadir) / Subexposición lineal | |
| Hard Light / Pin Light / Hard Mix | Luz fuerte / Luz focal / Mezcla definida | Luz dura / Luz puntual / Mezcla dura |
| Pass Through (group) | Pasar a través | Herencia |
| Clone Stamp / Pattern Stamp | Tampón de clonar / Tampón de motivo | Sello… |
| Feather | Calar | Desvanecer (clashed with *Fade*) |
| Fade | Transición | Desvanecer |
| Color Range | Gama de colores | Intervalo de colores |
| Select and Mask | Seleccionar y aplicar máscara | Selección y máscara |
| Selective Color | Corrección selectiva | Color selectivo |
| Shadows/Highlights; Highlights | Sombras/iluminaciones; Iluminaciones | Sombras/luces; Luces |
| Color Lookup | Consulta de colores | Búsqueda de color |
| Content-Aware Fill / Scale / Move | Relleno / Escala según el contenido; Mover con detección de contenido | Relleno de contenido… |
| Adaptive Wide Angle | Gran angular adaptable | Ángulo ancho adaptativo |
| Constrain proportions | Restringir proporciones | **Construir** proporciones |
| Render (filter menu) | Interpretar | Renderizar |
| Smart Sharpen | Enfoque suavizado | Enfoque inteligente |
| Blend If | Fusionar si | Combinar si |
| Knockout (blending options) | Extraer | Cobertura |
| Lights / Darks (Camera Raw) | Claros / Oscuros | Tonos claros / Tonos oscuros |
| Master (Hue/Saturation) | Todos | General |

Menu names that are also actions keep Photoshop's distinction: **Vista** (the *View* menu, not
«Ver»), **Volver** (*Revert*), **Borrar** (*Clear*), **Cortar** (*Trim*, unlike *Crop* «Recortar»),
**Descubrir** (*Reveal*), **Purgar** (*Purge*). Product and technology names (Photomerge, Camera
Raw, droplet, OpenType, sRGB, Wayland) are not translated.

## Out of scope

Engine error, status and parameter-help messages are still English in every language (see
[`localization-parity.md`](localization-parity.md)); this review does not change that. The review
was AI-assisted, row by row against Photoshop's Spanish terminology; a review by a native
speaker is still welcome.
