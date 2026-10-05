// Build the website into site/dist: index.html with the bundled script and
// photo list inlined, plus the photos themselves (Unsplash License), so the
// page works on any static host without hotlinking.
// Usage: bun scripts/site.ts [--serve]
// --serve also serves it on http://localhost:4517 and saves the ideal points
// marked on the page to site/labels.json.
import sharp from 'sharp'
import {existsSync} from 'node:fs'
import {mkdir} from 'node:fs/promises'

interface Photo {
  id: string
  alt: string
  author: string
  page: string
}

const out = 'site/dist'
await mkdir(`${out}/photos`, {recursive: true})
const photos = (await Bun.file('site/photos.json').json()) as Array<Photo>
async function download(id: string, file: string) {
  for (let attempt = 1; ; attempt++) {
    try {
      const res = await fetch(`https://images.unsplash.com/photo-${id}?w=1000&q=72&auto=format&fm=jpg`)
      if (!res.ok) throw new Error(`Photo ${id}: ${res.status}`)
      return await Bun.write(file, await res.arrayBuffer())
    } catch (e) {
      if (attempt === 3) throw e
    }
  }
}

const sized: Array<Photo & {src: string; width: number; height: number}> = []
// A few at a time: the CDN resets connections when flooded
for (let i = 0; i < photos.length; i += 8)
  sized.push(
    ...(await Promise.all(
      photos.slice(i, i + 8).map(async p => {
        const file = `${out}/photos/${p.id}.jpg`
        if (!existsSync(file)) await download(p.id, file)
        const {width, height} = await sharp(file).metadata()
        return {...p, src: `photos/${p.id}.jpg`, width: width!, height: height!}
      })
    ))
  )

const result = await Bun.build({entrypoints: ['site/main.ts'], minify: true, target: 'browser', format: 'iife'})
if (!result.success) {
  for (const log of result.logs) console.error(log)
  process.exit(1)
}
const js = (await result.outputs[0]!.text()).replaceAll('</script', '<\\/script')
const html = (await Bun.file('site/index.html').text())
  .replace('/*PHOTOS*/', () => JSON.stringify(sized))
  .replace('/*SCRIPT*/', () => js)
await Bun.write(`${out}/index.html`, html)
console.log(`${out}/index.html ${(html.length / 1024).toFixed(1)} kB, ${sized.length} photos`)

if (process.argv.includes('--serve')) {
  const labels = 'site/labels.json'
  Bun.serve({
    port: 4517,
    async fetch(req) {
      const path = new URL(req.url).pathname
      if (path === '/labels' && req.method === 'POST') {
        const data = (await req.json()) as Record<string, {x: number; y: number}>
        const sorted = Object.fromEntries(Object.entries(data).sort(([a], [b]) => a.localeCompare(b)))
        await Bun.write(labels, JSON.stringify(sorted, null, 1) + '\n')
        return new Response('ok')
      }
      if (path === '/labels.json') return new Response(Bun.file(labels))
      const file = Bun.file(`${out}${path === '/' ? '/index.html' : path}`)
      return (await file.exists()) ? new Response(file) : new Response('Not found', {status: 404})
    }
  })
  console.log('http://localhost:4517')
}
