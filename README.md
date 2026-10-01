# focuspoint

Find the focus point of an image: the (x, y) position of the main subject, so
that any crop centred on it keeps the subject in view.

- **Small**: ~3 kB gzipped, no dependencies.
- **Fast**: a few milliseconds per image, it works on an 80 px thumbnail.
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

1. **Downscale** the image to 160 px, averaging in linear light, then convert
   it to CIE Lab. Most signals work on a further halved 80 px version.
2. **Score each pixel's importance** by combining:
   - **Minimum barrier distance** ([FastMBD, Zhang et al. 2015](https://openaccess.thecvf.com/content_iccv_2015/papers/Zhang_Minimum_Barrier_Salient_ICCV_2015_paper.pdf)):
     how strongly a pixel is cut off from the image border. Backgrounds such as
     sky, walls and floors connect to the border, subjects don't.
   - **Border colour contrast** (from MB+ in the same paper): the Mahalanobis
     distance to the colour distribution of each border strip. The strongest
     border is left out, so a subject touching one edge still counts.
   - **Detail**: Laplacian energy, which favours in-focus areas over blurred
     backgrounds.
   - A mild **centre prior**, and the **alpha channel** for transparent images.
3. **Pick the subject**: threshold the map with Otsu's method, take the
   strongest connected region, and return its importance-weighted centroid and
   bounding box.

This differs from [smartcrop.js](https://github.com/jwagner/smartcrop.js),
which scores candidate crop windows for one aspect ratio. Its crop centre is
not a subject position, and its edge, skin and saturation heuristics are drawn
to busy texture.

## Evaluation

RESULTS

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
