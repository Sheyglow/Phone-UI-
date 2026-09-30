// Renders on-screen titles as transparent 1920x1080 PNG layers for spot/edit.py.
// Usage: node spot/supers.mjs
import { createRequire } from 'node:module';
import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_PATH || 'playwright');
const here = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.join(here, 'build', 'supers');
mkdirSync(outDir, { recursive: true });

// id -> { pos: CSS for the block, html }
const SUPERS = {
  'open-1': { pos: 'left:150px; top:360px', html: '<div class="big">Date.</div>' },
  'open-2': { pos: 'left:150px; top:470px', html: '<div class="big">Make friends.</div>' },
  'open-3': { pos: 'left:150px; top:580px', html: '<div class="big">Or both.</div>' },
  'meet':   { pos: 'left:130px; bottom:130px; width:760px', html: '<div class="label">Meet Kinly</div><div class="mid">Dating and friends, in one app.</div>' },
  'choose': { pos: 'left:130px; bottom:130px; width:760px', html: '<div class="label">You choose</div><div class="mid">Dating, friends, or both.</div>' },
  'match':  { pos: 'left:130px; top:120px; width:820px', html: '<div class="label">It\'s a match</div><div class="mid">Matched on what you both love.</div>' },
  'circle': { pos: 'left:130px; top:120px; width:760px', html: '<div class="label">Friends</div><div class="mid">Build your circle.</div>' },
  'plans':  { pos: 'left:130px; top:130px; width:640px', html: '<div class="label">Make plans</div><div class="mid">From first message to Friday night.</div>' },
};

const css = `
  html, body { margin: 0; background: transparent; }
  .block { position: absolute; color: #fff4ed; text-shadow: 0 2px 30px rgba(10,6,16,.55), 0 1px 3px rgba(10,6,16,.35); }
  .block > * + * { margin-top: 18px; }
  .label { font: 700 22px/1 Figtree, Arial, sans-serif; letter-spacing: .28em; text-transform: uppercase; color: #ffb47b; }
  .big { font: 500 italic 104px/1 Fraunces, Georgia, serif; letter-spacing: -.01em; }
  .mid { font: 500 italic 78px/1.05 Fraunces, Georgia, serif; text-wrap: balance; }
`;

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
for (const [id, s] of Object.entries(SUPERS)) {
  await page.setContent(`<!doctype html><html><head>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@1,9..144,500&family=Figtree:wght@700&display=swap">
    <style>${css}</style></head><body><div class="block" style="${s.pos}">${s.html}</div></body></html>`);
  await page.evaluate(() => document.fonts.ready);
  await page.waitForFunction(() => document.fonts.check('italic 500 40px Fraunces'));
  await page.screenshot({ path: path.join(outDir, id + '.png'), omitBackground: true });
}
await browser.close();
console.log('supers written to', outDir);
