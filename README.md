# focuspoint

Find the focus point of an image: the (x, y) position of the main subject, so
that any crop centred on it keeps the subject in view.

- **Small**: ~3.5 kB gzipped, no dependencies.
- **Fast**: about 1 ms per image. It works on a 48 px thumbnail.
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

1. **Downscale** the image to 48 px, averaging in linear light, and convert it
   to CIE Lab.
2. **Score how important each pixel is** by combining:
   - **Minimum barrier distance** ([FastMBD, Zhang et al. 2015](https://openaccess.thecvf.com/content_iccv_2015/papers/Zhang_Minimum_Barrier_Salient_ICCV_2015_paper.pdf)):
     how strongly a pixel is cut off from the image border. Backgrounds such as
     sky, walls and floors connect to the border. Subjects don't.
   - **Skin tone**, a soft Lab hue/chroma band.
   - A **centre prior**, and the **alpha channel** for transparent images.
   - Optionally, **border colour contrast** (MB+ from the same paper), which
     is off by default.
3. **Find the subject**: threshold the map (Otsu), take the strongest connected
   region, and move to the centre of the importance around it with a mean
   shift.
4. **Place the point for cropping**: search x and y for the position that keeps
   the most importance in view when the image is cropped to common container
   shapes (3:1, 16:9, 1:1, 4:5, 9:16). Each crop window is centred on the
   point and clamped to the image. When the subject sits near an edge, this
   pulls the point towards that edge: a head near the top gets a point near
   the top, so the whole head stays in view. Positions within 1% of the best
   count as equal, and the one closest to the subject wins, so the point
   stays on the subject unless moving it clearly helps.

This differs from [smartcrop.js](https://github.com/jwagner/smartcrop.js),
which scores candidate crop windows for one aspect ratio. The centre of its
crop is not a subject position, and its edge, skin and saturation heuristics
are drawn to busy texture.

### Applying the point

The point is meant for **centre-and-clamp** cropping: centre the crop window
on the point, then shift it back inside the image where it would overflow.
This is what image CDNs and most crop tools do:

```ts
function crop(point, imageWidth, imageHeight, cropWidth, cropHeight) {
  const clamp = (v: number, max: number) => Math.min(max, Math.max(0, v))
  return {
    x: clamp(point.x * imageWidth - cropWidth / 2, imageWidth - cropWidth),
    y: clamp(point.y * imageHeight - cropHeight / 2, imageHeight - cropHeight),
    width: cropWidth,
    height: cropHeight
  }
}
```

CSS `object-position: x% y%` places the image differently. It aligns
proportionally rather than centring, so it keeps less of the subject when the
point is off-centre.

## Evaluation

Five public datasets were used. The parameters were tuned on half of SALICON,
ECSSD and PASCAL-S; the other half of each, plus DUTS-TE and MSRA10K, were
held out. See [eval/README.md](eval/README.md) for the datasets and metrics.

For each image, the point is used to crop to five container shapes (3:1, 16:9,
1:1, 4:5, 9:16) with centre-and-clamp. **Kept** is how much of what matters
stays in view, averaged over those crops. For SALICON, what matters is where
people looked. For the other datasets it is the subject mask. **Oracle** is
the best possible single point for each image, found by exhaustive search
against the ground truth.

| Kept | SALICON | DUTS-TE | ECSSD | PASCAL-S | MSRA10K* |
|---|---|---|---|---|---|
| oracle | 85.7% | 92.2% | 88.7% | 85.6% | 90.8% |
| **focuspoint** | **82.9%** | **86.4%** | **85.1%** | **80.9%** | **88.1%** |
| centre | 82.6% | 84.9% | 83.6% | 79.7% | 83.3% |
| smartcrop, as Alinea uses it | 81.9% | 84.3% | 82.0% | 78.8% | 85.0% |

**Peak kept** is how often the single most important spot survives the crop,
for example the face that most people looked at:

| Peak kept | SALICON | DUTS-TE | ECSSD | PASCAL-S | MSRA10K* |
|---|---|---|---|---|---|
| oracle | 93.8% | 87.0% | 77.2% | 80.6% | 79.3% |
| **focuspoint** | **87.8%** | **83.5%** | **77.8%** | **78.4%** | **79.2%** |
| centre | 87.4% | 82.1% | 76.1% | 77.3% | 75.8% |
| smartcrop, as Alinea uses it | 86.6% | 82.0% | 76.0% | 77.0% | 77.6% |

\* MSRA10K: an evenly spaced sample of 2000 images. Oracle maximises kept, not
peak, so it can score below focuspoint on peak.

"smartcrop, as Alinea uses it" is the centre of smartcrop's best 100×100
crop. It does no better than a fixed centre point on SALICON, DUTS-TE, ECSSD
and PASCAL-S.

**Speed**: about 1 ms per image in Bun, against 3–4 ms for smartcrop on the
same 256 px input.

**Where it falls short.** focuspoint closes 10–30% of the gap between
centre-cropping and the oracle on most datasets, and about 65% on the simple
MSRA10K photos. The rest is mostly semantic: it can't tell that a small person
matters more than a big colourful sign. Closing that gap needs a learned
model. The evaluation harness here is ready to compare one.

## Viewer

Open [`demo/focuspoint-viewer.html`](demo/focuspoint-viewer.html) in a browser to try it on your own
images. It shows the focus point, the importance map and the crops the point
produces. Everything runs locally.

## Development

```sh
bun install
bun test          # unit tests
bun run check     # typecheck (TypeScript 7)
bun run build     # emit dist/
bun run size      # bundle size
bun eval/run.ts <dataset-dir>    # benchmark against ground truth
bun eval/tune.ts <dataset-dir>   # parameter search (train/test split)
bun eval/debug.ts <out-dir> <images...>  # render map + point overlays
bun run demo      # rebuild demo/focuspoint-viewer.html
```

A dataset directory holds `images/*.jpg` and `masks/*.png` with matching names.
See [eval/README.md](eval/README.md) for how to get the datasets.
