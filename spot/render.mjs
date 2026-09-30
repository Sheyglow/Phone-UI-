// Renders spot/kinly-spot.html frame by frame and encodes an H.264 MP4.
// Usage: node spot/render.mjs [--fps 30] [--out output/kinly-launch-sample.mp4] [--stills]
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_PATH || 'playwright');

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const fps = Number(opt('--fps', 30));
const out = path.resolve(root, opt('--out', 'output/kinly-launch-sample.mp4'));
const ffmpeg = process.env.FFMPEG || 'ffmpeg';
const stills = args.includes('--stills');

mkdirSync(path.dirname(out), { recursive: true });

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: ['--force-color-profile=srgb', '--disable-lcd-text'],
});
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
await page.goto(pathToFileURL(path.join(here, 'kinly-spot.html')).href + '?render');
await page.evaluate(async () => {
  await document.fonts.ready;
  const urls = [...document.querySelectorAll('[style*="url("]')].map(el => el.style.backgroundImage.slice(5, -2));
  await Promise.all(urls.map(u => new Promise(r => { const i = new Image(); i.onload = i.onerror = r; i.src = u; })));
});
const duration = await page.evaluate(() => window.DURATION);
const stage = page.locator('#stage');

if (stills) {
  const dir = path.join(root, 'output', 'stills');
  mkdirSync(dir, { recursive: true });
  for (const t of [2.4, 5.6, 9.2, 13.4, 16.8, 21.5, 26.8]) {
    await page.evaluate(t => window.renderAt(t), t);
    await stage.screenshot({ path: path.join(dir, `frame-${String(t).replace('.', '_')}s.png`) });
  }
  console.log('stills written to', dir);
  await browser.close();
  process.exit(0);
}

const enc = spawn(ffmpeg, [
  '-y', '-f', 'image2pipe', '-framerate', String(fps), '-i', '-',
  '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-pix_fmt', 'yuv420p',
  '-tune', 'grain', '-maxrate', '8M', '-bufsize', '16M', '-movflags', '+faststart', out,
], { stdio: ['pipe', 'inherit', 'inherit'] });

const total = Math.round(duration * fps);
for (let f = 0; f < total; f++) {
  await page.evaluate(t => window.renderAt(t), f / fps);
  const buf = await stage.screenshot({ type: 'png' });
  if (!enc.stdin.write(buf)) await new Promise(r => enc.stdin.once('drain', r));
  if (f % fps === 0) process.stdout.write(`\rframe ${f}/${total}`);
}
enc.stdin.end();
await new Promise(r => enc.on('close', r));
await browser.close();
console.log(`\nwrote ${out}`);
