# focuspoint

Find the focus point of an image: the (x, y) position of the main subject, so
that any crop centred on it keeps the subject in view.

- **Small**: ~3 kB gzipped, no dependencies.
- **Fast**: about 2 ms per image. It works on a 64 px thumbnail.
- **Runs anywhere**: browser, Node, Bun, Deno, workers. It only needs RGBA pixels.

```ts
import {focusPoint} from '@alinea/focuspoint'

const {x, y, box, confidence} = focusPoint({data, width, height})
// x, y: focus point, 0..1 from the top left
// box: bounding box of the subject, 0..1
// confidence: how clearly a subject stands out, 0..1
```

## Usage

### Browser

```ts
import {focusPointFromImage} from '@alinea/focuspoint/browser'

const img = new Image()
img.src = '/photo.jpg'
await img.decode()
const {x, y} = focusPointFromImage(img)
img.style.objectPosition = `${x * 100}% ${y * 100}%`
```

`focusPointFromImage` accepts an `HTMLImageElement`, `HTMLCanvasElement`,
`HTMLVideoElement`, `ImageBitmap` or `OffscreenCanvas`. It draws the image onto
a small canvas so the browser does the downscaling.

### Node / Bun with sharp

Pass any `{data, width, height}` RGBA buffer. Decoding and shrinking the image
first keeps things fast. Around 160 px on the longest side is plenty.

```ts
import sharp from 'sharp'
import {focusPoint} from '@alinea/focuspoint'

const {data, info} = await sharp(file)
  .rotate()
  .resize(160, 160, {fit: 'inside'})
  .ensureAlpha()
  .raw()
  .toBuffer({resolveWithObject: true})
const point = focusPoint({data, width: info.width, height: info.height})
```

Larger inputs work too. They are downsampled internally on a sparse grid.

### Saliency map

`saliency(image)` returns the importance map behind the point
(`{width, height, data: Float32Array}`, values 0..1), handy for debugging or
custom cropping logic.

## How it works

1. **Downscale** the image to 64 px, averaging in linear light, and convert it
   to CIE Lab.
2. **Score how important each pixel is** by combining:
   - **Minimum barrier distance** ([FastMBD, Zhang et al. 2015](https://openaccess.thecvf.com/content_iccv_2015/papers/Zhang_Minimum_Barrier_Salient_ICCV_2015_paper.pdf)):
     how strongly a pixel is cut off from the image border. Backgrounds such as
     sky, walls and floors connect to the border. Subjects don't.
   - **Border colour contrast** (from MB+ in the same paper): the Mahalanobis
     distance to the colour distribution of each border strip. The strongest
     border is left out, so a subject that touches one edge still counts.
   - **Skin tone**, a soft Lab hue/chroma band, at a low weight.
   - A **centre prior**, and the **alpha channel** for transparent images.
3. **Pick the subject**:
   - Threshold the map strictly (Otsu, raised towards the peak) and keep the
     strongest connected region. This decides *which* subject.
   - Move the point with a mean shift over the looser region around it, which
     finds the *centre* of that subject. A crop centred there keeps most of it.
   - The bounding box covers that area.

This differs from [smartcrop.js](https://github.com/jwagner/smartcrop.js),
which scores candidate crop windows for one aspect ratio. Its crop centre is
not a subject position, and its edge, skin and saturation heuristics are drawn
to busy texture.

## Evaluation

Five public datasets were used. The parameters were tuned on ECSSD and
PASCAL-S only, so DUTS-TE, MSRA10K and SALICON are held out. See
[eval/README.md](eval/README.md) for the datasets and metrics.

**Subject kept** is the average share of the subject that remains visible
when the image is cropped around the point to 1:1, 9:16, 3:1 and a 2× zoom.
**On subject** is how often the point lands on the subject. **Oracle** uses
the true centre of the subject: it is the best a single point can do.

| Subject kept | ECSSD | PASCAL-S | DUTS-TE | MSRA10K* | SALICON |
|---|---|---|---|---|---|
| oracle | 82.7% | 77.8% | 87.4% | 85.6% | 77.0% |
| **focuspoint** | **78.6%** | **73.0%** | **79.9%** | **82.5%** | 72.7% |
| centre | 76.3% | 71.3% | 77.8% | 75.5% | **72.9%** |
| smartcrop | 66.6% | 63.9% | 69.8% | 73.0% | 66.9% |

| On subject | ECSSD | PASCAL-S | DUTS-TE | MSRA10K* |
|---|---|---|---|---|
| oracle | 93.6% | 87.2% | 86.4% | 97.8% |
| **focuspoint** | **87.5%** | **80.1%** | **66.3%** | **94.7%** |
| centre | 77.8% | 71.3% | 56.2% | 74.2% |
| smartcrop | 75.2% | 68.2% | 52.2% | 79.2% |

\* MSRA10K: an evenly spaced sample of 2000 images.

smartcrop is given each target aspect ratio, which our single point is not,
and it still keeps less of the subject than a fixed centre point on every
dataset. Speed: 2–3 ms per
image for focuspoint against about 15 ms for smartcrop, both on 256 px input
in Bun.

**Where it falls short.** It closes 20–70% of the gap between centre-cropping
and the oracle. The rest is mostly semantic: it can't tell that a small person
matters more than a big colourful sign. On SALICON, where cluttered everyday
scenes often have no single subject, it is no better than the centre.
Closing that gap needs a learned model. That is a possible follow-up: the
evaluation harness here is ready to compare one.

## Development

```sh
bun install
bun test          # unit tests
bun run check     # typecheck (TypeScript 7)
bun run build     # emit dist/
bun run size      # bundle size
bun eval/run.ts <dataset-dir>    # benchmark against masks
bun eval/tune.ts <dataset-dir>   # parameter search (train/test split)
bun eval/debug.ts <out-dir> <images...>  # render map + point overlays
```

A dataset directory holds `images/*.jpg` and `masks/*.png` with matching names.
See [eval/README.md](eval/README.md) for how to get the datasets.
