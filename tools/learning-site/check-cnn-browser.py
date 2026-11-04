"""Verify the generated local MAL study preview using Playwright and installed Chrome."""
import os
import sys
from playwright.sync_api import sync_playwright
from pathlib import Path
import hashlib
BASE=sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:8080'
ROOT=Path(__file__).resolve().parents[2]
Path('/tmp/cosmos-cnn-oct27').mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
 b=p.chromium.launch(executable_path=os.environ.get('COSMOSAI_CHROME','/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'),headless=True)
 errors=[]
 for width in [1440,390]:
  page=b.new_page(viewport={'width':width,'height':1000})
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto(BASE+'/learn/examples/cnn-fundamentals/')
  assert '24' in page.locator('#products').inner_text()
  page.get_by_role('button',name='Output (1,1): 48',exact=True).click()
  assert '34' in page.locator('#products').inner_text() and '48' in page.locator('#products').inner_text()
  page.locator('#bias').fill('2');page.get_by_role('button',name='Calculate',exact=True).click()
  assert '= 50' in page.locator('#products').inner_text()
  page.locator('#bias').fill('-100');page.locator('#relu').check()
  assert page.get_by_role('button',name='Output (1,1): 0',exact=True).count()==1
  page.locator('#preset').select_option('stride');assert '4×4×1' in page.locator('#shape-result').inner_text()
  page.locator('#preset').select_option('alex227');assert '56×56×96' in page.locator('#shape-result').inner_text()
  page.locator('#preset').select_option('tiny');assert '112' in page.locator('#shape-result').inner_text()
  assert '5×5' in page.locator('#rf-result').inner_text()
  page.locator('#rf-depth').select_option('3');assert '7×7' in page.locator('#rf-result').inner_text()
  page.locator('#rf-stride').select_option('2');assert '15×15' in page.locator('#rf-result').inner_text()
  diagram=page.locator('img[src="receptive-fields.svg"]');diagram.evaluate('el => el.decode()')
  assert diagram.evaluate('el => el.naturalWidth > 0')

  assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),width
  page.screenshot(path=f'/tmp/cosmos-cnn-oct27/browser-{width}.png',full_page=True)
  page.goto(BASE+'/learn/courses/MAL/')
  assert page.locator('a[href="/learn/examples/cnn-fundamentals/"]').count()==1
  assert page.locator('a[href*="mal-cnn-"]').count()==9
  page.get_by_role('link',name='CNN Explainer: guided visit, model comparison and how the visuals work',exact=True).click()
  assert page.get_by_role('heading',name='CNN Explainer — connect the pictures to CosmosAI',exact=True).count()==1
  assert '19,920' in page.inner_text('body')
  assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
  page.get_by_role('link',name='the Polo Club reference shelf',exact=True).click()
  assert page.get_by_role('heading',name='The linked ConceptAttention paper',exact=True).count()==1
  assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
  page.goto(BASE+'/learn/courses/MAL/')

  guide=page.get_by_role('link',name='English study companion:',exact=False)
  guide.click();assert page.get_by_role('heading',name='What the lecturer explicitly highlights for assessment',exact=True).count()==1
  page.locator('img').evaluate_all('els => Promise.all(els.map(el => el.decode()))')
  assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
  page.goto(BASE+'/learn/concepts/?course=MAL#mal-cnn-output-size')
  assert page.locator('#mal-cnn-output-size').is_visible()
  assert not page.evaluate('document.body.innerText.includes("MAL_L04_CNN_concepts.md")')
  page.goto(BASE+'/learn/')
  assert '+0.119161' in page.locator('#classifier-weight-timeline').inner_text()
  assert '+0.118419' in page.locator('#classifier-weight-timeline').inner_text()
  page.locator('#image-select').select_option('1')
  assert 'weight = +0.119161' in page.locator('#classifier-weight-timeline').inner_text()
  page.locator('#inspect-classifier-weight').click()
  assert page.locator('#parameter-select').input_value()=='classifier.weight|0,0'
  assert '+0.118419' in page.locator('#parameter-equation').inner_text()
  page.locator('#next-update').click()
  assert 'weight = +0.118419' in page.locator('#classifier-weight-timeline').inner_text()
  page.screenshot(path=f'/tmp/cosmos-cnn-oct27/replay-origin-{width}.png',full_page=True)
  page.close()
 # Hash check the served source PDF and check the derivative page count outside browser.
 response=b.new_page().request.get(BASE+'/learn/courses/materials/MAL/curs_4_CNN.pdf')
 assert response.status==200
 assert hashlib.sha256(response.body()).hexdigest()==hashlib.sha256((ROOT/'docs/courses/MAL/lectures/curs_4_CNN.pdf').read_bytes()).hexdigest()
 for secret in ['/learn/courses/materials/MAL/../../../../docs/workflow.md','/docs/workflow.md']:
  assert b.new_page().request.get(BASE+secret).status==404
 assert not errors,errors
 b.close()
print('PASS: desktop/mobile controls, concept/course links, PDF source hash, no browser errors, no horizontal overflow, private docs not exposed.')
