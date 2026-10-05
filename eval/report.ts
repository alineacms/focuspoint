// Render an HTML report: the goal, current scores, training curves and a
// gallery comparing ground truth, the heuristic and a model on real images.
// Images are embedded, so keep the output local (the datasets are research
// only).
// Usage: bun eval/report.ts <out.html> <maps-dir> [--runs v2,v4,t1,teacher]
import sharp from 'sharp'
import {existsSync} from 'node:fs'
import {readFile, writeFile} from 'node:fs/promises'
import {basename, join} from 'node:path'
import {parseArgs} from 'node:util'
import {locate, type SaliencyMap} from '../src/index.ts'
import * as heuristic from './heuristic.ts'
import {containers, cover, mean, oracle, score, type Mask, type Point} from './metrics.ts'
import {fit, loadDataset, readMap, type Sample} from './run.ts'

const {values, positionals} = parseArgs({
  allowPositionals: true,
  options: {
    runs: {type: 'string', default: 'v4,v7,v8,teacher'},
    sets: {type: 'string', default: 'eval/data/SALICON-TR-VAL,eval/data/DUTS-TR-VAL'}
  }
})
const [out, maps] = positionals
if (!out || !maps) throw new Error('Usage: report.ts <out.html> <maps-dir> [--runs ...]')
const options = {emphasis: 1, fit: 1}
const model = basename(maps)

interface Row {
  s: Sample
  map: SaliencyMap
  points: {heuristic: Point; model: Point; oracle: Point}
  kept: {heuristic: number; model: number; oracle: number; center: number}
  /** Share of the five crops that keep the most important spot */
  peak: {heuristic: number; model: number; oracle: number; center: number}
}

const keptOf = (mask: Mask, p: Point) => mean([score(mask, p)]).keptAvg
const peakOf = (mask: Mask, p: Point, top: Point) => score(mask, p, top).peak

async function rows(dir: string): Promise<Array<Row>> {
  const samples = await loadDataset(dir)
  return Promise.all(
    samples.map(async s => {
      const map = fit(await readMap(join(maps!, s.set, `${s.name}.f32`)), s.image)
      const points = {heuristic: heuristic.focusPoint(s.image), model: locate(map, options), oracle: oracle(s.mask)}
      return {
        s,
        map,
        points,
        kept: {
          heuristic: keptOf(s.mask, points.heuristic),
          model: keptOf(s.mask, points.model),
          oracle: keptOf(s.mask, points.oracle),
          center: keptOf(s.mask, {x: 0.5, y: 0.5})
        },
        peak: {
          heuristic: peakOf(s.mask, points.heuristic, s.peak),
          model: peakOf(s.mask, points.model, s.peak),
          oracle: peakOf(s.mask, points.oracle, s.peak),
          center: peakOf(s.mask, {x: 0.5, y: 0.5}, s.peak)
        }
      }
    })
  )
}

/** The photo dimmed except where the map is important. */
async function spotlight(s: Sample, map: {width: number; height: number; data: Float32Array}, width = 260) {
  const height = Math.round((s.image.height / s.image.width) * width)
  const rgb = await sharp(Buffer.from(s.image.data), {raw: {width: s.image.width, height: s.image.height, channels: 4}})
    .removeAlpha()
    .resize(width, height)
    .raw()
    .toBuffer()
  let top = 0
  for (const v of map.data) if (v > top) top = v
  const bytes = Buffer.from(Uint8Array.from(map.data, v => Math.round((v / (top || 1)) * 255)))
  const m = await sharp(bytes, {raw: {width: map.width, height: map.height, channels: 1}})
    .resize(width, height, {fit: 'fill', kernel: 'cubic'})
    .extractChannel(0)
    .raw()
    .toBuffer()
  const mix = Buffer.alloc(rgb.length)
  for (let i = 0; i < width * height; i++) {
    const a = 0.18 + 0.82 * Math.sqrt(m[i]! / 255)
    for (let c = 0; c < 3; c++) mix[i * 3 + c] = rgb[i * 3 + c]! * a
  }
  const jpg = await sharp(mix, {raw: {width, height, channels: 3}}).jpeg({quality: 78}).toBuffer()
  return {src: `data:image/jpeg;base64,${jpg.toString('base64')}`, width, height}
}

function overlay(p: Point, s: Sample, width: number, height: number, cls: string) {
  const rects = Object.entries(containers).map(([name, aspect]) => {
    const r = cover(p, s.image.width, s.image.height, aspect, 'center')
    const k = width / s.image.width
    return `<rect class="crop" x="${r.x * k}" y="${r.y * k}" width="${r.width * k}" height="${r.height * k}"><title>${name}</title></rect>`
  })
  return `${rects.join('')}<circle class="pt ${cls}" cx="${p.x * width}" cy="${p.y * height}" r="6"/>`
}

async function panel(label: string, s: Sample, map: {width: number; height: number; data: Float32Array}, p: Point, kept: number, peak: number, cls: string, top = false) {
  const img = await spotlight(s, map)
  const spot = top ? `<circle class="top" cx="${s.peak.x * img.width}" cy="${s.peak.y * img.height}" r="11"><title>most-looked-at spot</title></circle>` : ''
  return `<figure class="panel"><svg viewBox="0 0 ${img.width} ${img.height}" role="img" aria-label="${label}">
  <image href="${img.src}" width="${img.width}" height="${img.height}"/>${spot}${overlay(p, s, img.width, img.height, cls)}</svg>
  <figcaption><span>${label}</span><span><b>${(kept * 100).toFixed(0)}% kept</b> · top spot ${Math.round(peak * 5)}/5</span></figcaption></figure>`
}

const pct = (v: number) => (v * 100).toFixed(1)

// Training curves from runs/<run>/out.txt
interface Curve {
  run: string
  points: Array<{epoch: number; val: Record<string, number>}>
}
const curves: Array<Curve> = []
for (const run of values.runs.split(',')) {
  const file = `train/runs/${run}/out.txt`
  if (!existsSync(file)) continue
  const points = (await readFile(file, 'utf8'))
    .split('\n')
    .map(l => l.match(/^epoch (\d+) .*?val (\{.*?\})/))
    .filter(m => m !== null)
    .map(m => ({epoch: Number(m[1]), val: JSON.parse(m[2]!) as Record<string, number>}))
  if (points.length) curves.push({run, points})
}

const describe: Record<string, string> = {
  v1: 'small, KL only',
  v2: 'small, 31k params',
  v3: 'small, 96 px input',
  v4: 'small, 41k params',
  v6: 'small, 41k, trained for top spot',
  v7: 'small, 41k, top spot, real labels only',
  v8: 'small, 41k, top spot, + 30k teacher-labelled photos',
  t1: 'medium, 208k, from scratch',
  teacher: 'pretrained MobileNetV3, 3M'
}

function lineChart(title: string, key: string) {
  const W = 460, H = 220, L = 44, R = 12, T = 12, B = 30
  const series = curves.filter(c => c.points.some(p => key in p.val))
  const all = series.flatMap(c => c.points.map(p => p.val[key]!).filter(v => v !== undefined))
  if (!all.length) return ''
  const lo = Math.floor(Math.min(...all) * 200) / 200
  const hi = Math.ceil(Math.max(...all) * 200) / 200
  const maxEpoch = Math.max(...series.flatMap(c => c.points.map(p => p.epoch)))
  const x = (e: number) => L + ((e - 1) / Math.max(1, maxEpoch - 1)) * (W - L - R)
  const y = (v: number) => T + (1 - (v - lo) / (hi - lo || 1)) * (H - T - B)
  const ticks = Array.from({length: 5}, (_, i) => lo + ((hi - lo) * i) / 4)
  const grid = ticks
    .map(t => `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}"/><text class="tick" x="${L - 6}" y="${y(t) + 4}" text-anchor="end">${pct(t)}</text>`)
    .join('')
  const lines = series
    .map((c, i) => {
      const pts = c.points.filter(p => key in p.val)
      const d = pts.map((p, j) => `${j ? 'L' : 'M'}${x(p.epoch).toFixed(1)},${y(p.val[key]!).toFixed(1)}`).join('')
      const last = pts.at(-1)!
      const dots = pts
        .map(p => `<circle class="hit" cx="${x(p.epoch)}" cy="${y(p.val[key]!)}" r="7" data-tip="${c.run} · epoch ${p.epoch} · ${pct(p.val[key]!)}%"/>`)
        .join('')
      return `<path class="line s${i + 1}" d="${d}"/><circle class="end s${i + 1}" cx="${x(last.epoch)}" cy="${y(last.val[key]!)}" r="4"/>${dots}`
    })
    .join('')
  return `<figure class="chart"><figcaption>${title}</figcaption><svg viewBox="0 0 ${W} ${H}">${grid}
  <text class="tick" x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">epoch</text>${lines}</svg></figure>`
}

function legend() {
  return `<ul class="legend">${curves
    .map((c, i) => `<li><span class="sw s${i + 1}"></span>${c.run} <small>${describe[c.run] ?? ''}</small></li>`)
    .join('')}</ul>`
}

function bars(title: string, r: Array<Row>, measure: 'kept' | 'peak') {
  const avg = (k: keyof Row['kept']) => r.reduce((s, x) => s + x[measure][k], 0) / r.length
  const items = [
    {label: 'centre', v: avg('center'), cls: 'muted'},
    {label: 'heuristic', v: avg('heuristic'), cls: 'muted'},
    {label: `model (${model})`, v: avg('model'), cls: 'accent'},
    {label: 'oracle', v: avg('oracle'), cls: 'muted'}
  ]
  const target = measure === 'peak' ? avg('heuristic') + 0.01 : avg('heuristic') - 0.005
  const W = 460, row = 34, L = 120, R = 56, H = items.length * row + 52
  const lo = Math.floor(Math.min(target, ...items.map(i => i.v)) * 100) / 100
  const hi = Math.ceil(Math.max(...items.map(i => i.v)) * 100) / 100
  const x = (v: number) => L + ((v - lo) / (hi - lo)) * (W - L - R)
  const step = hi - lo > 0.08 ? 0.02 : 0.01
  const ticks = Array.from({length: Math.floor((hi - lo) / step + 1e-9) + 1}, (_, i) => lo + i * step)
  const axis = ticks
    .map(t => `<line class="grid" x1="${x(t)}" x2="${x(t)}" y1="14" y2="${H - 22}"/><text class="tick" x="${x(t)}" y="${H - 22 + 12}" text-anchor="middle">${Math.round(t * 100)}</text>`)
    .join('')
  const rows = items
    .map(
      (it, i) => `<g data-tip="${it.label}: ${pct(it.v)}% ${measure}"><text class="label" x="${L - 8}" y="${i * row + 35}" text-anchor="end">${it.label}</text>
  <circle class="dotm ${it.cls}" cx="${x(it.v)}" cy="${i * row + 31}" r="7"/>
  <text class="value" x="${x(it.v) + 12}" y="${i * row + 35}">${pct(it.v)}%</text></g>`
    )
    .join('')
  return `<figure class="chart"><figcaption>${title} <small>${r.length} images · ${measure === 'kept' ? 'kept %, must stay above heuristic − 0.5' : '% of crops that keep the top spot, target is heuristic + 1'}</small></figcaption><svg viewBox="0 0 ${W} ${H}">${axis}
  <line class="target" x1="${x(target)}" x2="${x(target)}" y1="14" y2="${H - 22}"/><text class="tick target-label" x="${x(target)}" y="10" text-anchor="middle">${measure === 'peak' ? 'target' : 'floor'}</text>${rows}</svg></figure>`
}

const sets = values.sets.split(',')
const data = await Promise.all(sets.map(rows))

// Gallery: the model's biggest wins and losses against the heuristic, and a
// few at random, from each set
async function gallery(r: Array<Row>) {
  const byGain = [...r].sort((a, b) => b.kept.model - b.kept.heuristic - (a.kept.model - a.kept.heuristic))
  const step = Math.floor(r.length / 3)
  const picks: Array<[string, Row]> = [
    ...byGain.slice(0, 3).map(x => ['model wins', x] as [string, Row]),
    ...[r[step]!, r[2 * step]!].map(x => ['random', x] as [string, Row]),
    ...byGain.slice(-2).map(x => ['model loses', x] as [string, Row])
  ]
  const out: Array<string> = []
  for (const [tag, row] of picks) {
    const {s, points, kept, peak} = row
    const h = heuristic.saliency(s.image)
    out.push(`<article class="row"><header><span class="tag ${tag.replace(' ', '-')}">${tag}</span><code>${s.name}</code></header><div class="panels">
    ${await panel(s.set.startsWith('SALICON') ? 'Where people looked' : 'Subject mask', s, s.mask, points.oracle, kept.oracle, peak.oracle, 'oracle', true)}
    ${await panel('Heuristic', s, h, points.heuristic, kept.heuristic, peak.heuristic, 'heuristic')}
    ${await panel(`Model ${model}`, s, row.map, points.model, kept.model, peak.model, 'model')}</div></article>`)
  }
  return out.join('\n')
}

// The goal: one image with its five crops
const example = data[0]!
  .filter(x => x.s.image.width > x.s.image.height)
  .reduce((a, b) => (b.kept.model - b.kept.center > a.kept.model - a.kept.center ? b : a))
const exampleCrops = Object.entries(containers)
  .map(([name, aspect]) => {
    const r = cover(example.points.model, example.s.image.width, example.s.image.height, aspect, 'center')
    const kept = score(example.s.mask, example.points.model).kept[name as keyof typeof containers]
    return {name, r, kept}
  })
const exImg = await spotlight(example.s, example.s.mask, 420)
const k = exImg.width / example.s.image.width
const goal = `<figure class="goal"><svg viewBox="0 0 ${exImg.width} ${exImg.height}"><image href="${exImg.src}" width="${exImg.width}" height="${exImg.height}"/>
${exampleCrops.map((c, i) => `<rect class="crop c${i}" x="${c.r.x * k}" y="${c.r.y * k}" width="${c.r.width * k}" height="${c.r.height * k}"/>`).join('')}
<circle class="top" cx="${example.s.peak.x * exImg.width}" cy="${example.s.peak.y * exImg.height}" r="13"/><circle class="pt model" cx="${example.points.model.x * exImg.width}" cy="${example.points.model.y * exImg.height}" r="7"/></svg>
<figcaption>${exampleCrops.map((c, i) => `<span class="c${i}">${c.name} ${(c.kept * 100).toFixed(0)}%</span>`).join(' ')}</figcaption></figure>`

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Focus model report</title>
<style>
:root{color-scheme:light;--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8a8984;--line:#e6e5e1;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--accent:#2a78d6;--muted:#c9c8c2;--target:#e34948;
--model:#2a78d6;--heuristic:#eb6834;--oracle:#1baf7a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#1a1a19;--card:#222220;--ink:#fff;--ink2:#c3c2b7;--ink3:#8f8e86;--line:#3a3936;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--accent:#3987e5;--muted:#55544f;--target:#e66767;--model:#3987e5;--heuristic:#d95926;--oracle:#199e70}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1a1a19;--card:#222220;--ink:#fff;--ink2:#c3c2b7;--ink3:#8f8e86;--line:#3a3936;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--accent:#3987e5;--muted:#55544f;--target:#e66767;--model:#3987e5;--heuristic:#d95926;--oracle:#199e70}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:40px 0 8px}
p{color:var(--ink2);max-width:70ch}small{color:var(--ink3);font-weight:400}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.chart{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.chart figcaption{font-weight:600;margin-bottom:6px}.chart svg{width:100%;height:auto;display:block}
.grid{stroke:var(--line)}.tick,.label{fill:var(--ink3);font-size:11px}.label{fill:var(--ink2);font-size:12px}.value{fill:var(--ink);font-size:12px;font-weight:600}
.line{fill:none;stroke-width:2}.end{stroke:var(--card);stroke-width:2}.hit{fill:transparent;cursor:crosshair}
.s1{stroke:var(--s1);fill:var(--s1);background:var(--s1)}.s2{stroke:var(--s2);fill:var(--s2);background:var(--s2)}.s3{stroke:var(--s3);fill:var(--s3);background:var(--s3)}.s4{stroke:var(--s4);fill:var(--s4);background:var(--s4)}
path.line{fill:none}.dotm{stroke:var(--card);stroke-width:2}.dotm.muted{fill:var(--ink3)}.dotm.accent{fill:var(--accent)}.target-label{fill:var(--target)}.target{stroke:var(--target);stroke-width:2;stroke-dasharray:4 3}
.legend{list-style:none;padding:0;margin:8px 0 0;display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px}.sw{display:inline-block;width:12px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
.goal{margin:12px 0;max-width:460px}.goal svg{width:100%;height:auto;border-radius:8px;display:block}.goal figcaption{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:13px;margin-top:6px}
.goal .crop{fill:none;stroke-width:2.5}.c0{stroke:#2a78d6;color:#2a78d6}.c1{stroke:#eb6834;color:#eb6834}.c2{stroke:#1baf7a;color:#1baf7a}.c3{stroke:#eda100;color:#c98500}.c4{stroke:#e87ba4;color:#d55181}
.row{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:12px 0}
.row header{display:flex;gap:10px;align-items:center;margin-bottom:8px;font-size:13px}code{color:var(--ink3)}
.tag{padding:1px 8px;border-radius:99px;font-weight:600;font-size:12px;border:1px solid var(--line)}
.panels{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.panel{margin:0}.panel svg{width:100%;height:auto;display:block;border-radius:6px}
.panel figcaption{display:flex;flex-direction:column;font-size:13px;color:var(--ink2);margin-top:4px}.panel b{color:var(--ink)}
.panel .crop{fill:none;stroke:rgba(255,255,255,.7);stroke-width:1.2;opacity:0;transition:opacity .15s}.panel:hover .crop{opacity:1}
.top{fill:none;stroke:#fff;stroke-width:2;stroke-dasharray:3 3}h3{font-size:15px;margin:20px 0 6px}.pt{stroke:#fff;stroke-width:2}.pt.model{fill:var(--model)}.pt.heuristic{fill:var(--heuristic)}.pt.oracle{fill:var(--oracle)}
.key{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--ink2)}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px;border:2px solid #fff;box-shadow:0 0 0 1px var(--line)}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);font-size:12px;padding:4px 8px;border-radius:6px;opacity:0;transition:opacity .1s}
@media (max-width:640px){.panels{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Learned focus model</h1>
<p>Progress report, generated ${new Date().toLocaleString('en-GB', {dateStyle: 'medium', timeStyle: 'short'})}. All images and numbers come from the validation slices of the training data (every 20th SALICON train and DUTS-TR image). Those slices are never trained on, and the evaluation sets aren't touched here.</p>

<h2>What we're aiming for</h2>
<p>A focus point is used to crop an image into containers of five shapes: a 3:1 banner, 16:9, square, 4:5 and a 9:16 story. Each crop is centred on the point and pushed back inside the image. The goal is that the thing people look at first stays in every crop (<b>top spot</b>), without losing more of the rest than today (<b>kept</b>). Below, the bright areas are where people looked, the dashed ring is the most-looked-at spot, and the boxes are the five crops around the model's point.</p>
${goal}
<p>The model replaces only the importance map. The library still picks the subject and places the point. To ship, the model has to keep the top spot in at least about 1 point more of the crops than today's heuristic on SALICON and DUTS-TE, the two held-out sets with the most room, and must not lose more than 0.5 kept anywhere.</p>

<h2>Where we are</h2>
<h3>Top spot, the main measure</h3>
<p>How often the single most-looked-at spot (or the heart of the subject) stays inside each of the five crops. A focus point is one point, so this is what matters: the thing people look at first must survive every crop. The oracle maximises kept, not this, so it can score lower here.</p>
<div class="grid2">${data.map((r, i) => bars(basename(sets[i]!).replace('-VAL', ' validation'), r, 'peak')).join('')}</div>
<h3>Kept, the guard</h3>
<p>The share of all attention that stays inside the crops. When attention is split, for example between two people, the best-scoring point can sit between them and let the tall crop cut off the face most people looked at, so this only guards against the model getting worse overall.</p>
<div class="grid2">${data.map((r, i) => bars(basename(sets[i]!).replace('-VAL', ' validation'), r, 'kept')).join('')}</div>

<h2>Training</h2>
<p>Validation scores during training, measured on the low-resolution maps with pure placement, so they read differently from the full pipeline above, but they move together with it. Top spot is only logged by runs started after it became the main measure.</p>
<div class="grid2">${lineChart('SALICON train, validation top spot', 'SALICON-TR peak')}${lineChart('DUTS-TR, validation top spot', 'DUTS-TR peak')}</div>
<div class="grid2">${lineChart('SALICON train, validation kept', 'SALICON-TR kept')}${lineChart('DUTS-TR, validation kept', 'DUTS-TR kept')}</div>
${legend()}

<h2>On real images</h2>
<div class="key"><span><span class="dot" style="background:var(--oracle)"></span>best possible point</span><span><span class="dot" style="background:var(--heuristic)"></span>heuristic point</span><span><span class="dot" style="background:var(--model)"></span>model point</span><span>dashed ring: most-looked-at spot</span><span>hover a photo to see its five crops</span></div>
<p>Each row shows the same photo three times. Brightness shows the importance map: what people looked at, or the subject mask (left); what the heuristic thinks matters (middle); what the model predicts (right). The picks are the model's three biggest wins against the heuristic, two at random and its two biggest losses.</p>
${(await Promise.all(data.map(gallery))).map((g, i) => `<h2>${basename(sets[i]!).replace('-VAL', '')}</h2>${g}`).join('')}
</main><div id="tip"></div>
<script>
const tip=document.getElementById('tip')
document.addEventListener('pointermove',e=>{const t=e.target.closest('[data-tip]');if(!t){tip.style.opacity=0;return}
tip.textContent=t.dataset.tip;tip.style.left=e.clientX+12+'px';tip.style.top=e.clientY+12+'px';tip.style.opacity=1})
</script></body></html>`
await writeFile(out, html)
console.log(`wrote ${out} (${(html.length / 1024).toFixed(0)} kB)`)
