/* Export existing scenes with Excalidraw itself, using build-only browser tools. */
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const {createRequire} = require('node:module');
const root = path.resolve(__dirname, '..');
const requireBuild = createRequire(path.join(process.env.COSMOSAI_LEARNING_TOOLS || path.join(root, 'tools/learning-site'), 'package.json'));
const {chromium} = requireBuild('playwright');
const bundle = requireBuild.resolve('@excalidraw/utils');
const site = path.resolve(process.argv[2] || path.join(root, 'apps/learning-content/site'));
const licenses = ['NOTICE.txt', 'Excalidraw-MIT.txt', 'Virgil-OFL.txt', 'Cascadia-Code-OFL.txt']
  .map(name => fs.readFileSync(path.join(root, 'apps/api-gateway/static/vendor/learning-licenses', name), 'utf8')).join('\n\n');

(async () => {
  const server = http.createServer((req, res) => {
    if (req.url === '/exporter.js') {
      res.writeHead(200, {'Content-Type': 'text/javascript'}); fs.createReadStream(bundle).pipe(res);
    } else {
      res.writeHead(200, {'Content-Type': 'text/html'});
      res.end('<!doctype html><script type="module">import {exportToSvg} from "/exporter.js"; window.exportScene=exportToSvg;</script>');
    }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  let browser;
  try {
    browser = await chromium.launch({headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage']});
    const page = await browser.newPage();
    // Font bytes are embedded by the official exporter. No external fetches.
    await page.route('**/*', route => route.request().url().startsWith(origin) || route.request().url().startsWith('data:') ? route.continue() : route.abort());
    await page.goto(origin);
    await page.waitForFunction(() => typeof window.exportScene === 'function');
    const catalog = JSON.parse(fs.readFileSync(path.join(site, 'diagrams/catalog.json'), 'utf8'));
    for (const item of catalog) {
      const scene = JSON.parse(fs.readFileSync(path.join(site, 'diagrams/scenes', item.id + '.excalidraw'), 'utf8'));
      const result = await page.evaluate(async ({scene, title, licenses}) => {
        const svg = await window.exportScene({data: scene, config: {canvasBackgroundColor: '#ffffff', padding: 24, theme: 'light'}});
        for (const [tag, text] of [['title', title], ['metadata', licenses]]) {
          const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
          node.textContent = text; svg.prepend(node);
        }
        svg.querySelectorAll('a').forEach(a => a.setAttribute('target', '_top'));
        svg.setAttribute('role', 'img');
        const box = svg.getAttribute('viewBox').split(/\s+/).map(Number);
        return {svg: svg.outerHTML, width: box[2], height: box[3]};
      }, {scene, title: item.title, licenses});
      if (/obsidian:\/\/|vscode:\/\/|\/home\/|<script\b/i.test(result.svg)) throw new Error(`Unsafe export: ${item.id}`);
      fs.writeFileSync(path.join(site, 'diagrams/svg', item.id + '.svg'), result.svg);
      Object.assign(item, {width: result.width, height: result.height});
      console.log(`${item.id}: ${Math.round(item.width)} x ${Math.round(item.height)}`);
    }
    fs.writeFileSync(path.join(site, 'diagrams/catalog.json'), JSON.stringify(catalog, null, 2) + '\n');
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
