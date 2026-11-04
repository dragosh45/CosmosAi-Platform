"""Reproduce the exact SVG teaching diagram for stride and stacked receptive fields."""
from pathlib import Path
out=Path(__file__).resolve().parents[2]/'apps/learning-examples/cnn-fundamentals/receptive-fields.svg'
s=['<svg xmlns="http://www.w3.org/2000/svg" width="980" height="660" viewBox="0 0 980 660" role="img" aria-labelledby="title desc">',
'<title id="title">Stride one and two stacked 3 by 3 convolutions</title>',
'<desc id="desc">Stride one moves a window one input cell per step; stride two moves it two cells. With two consecutive valid 3 by 3 stride-one layers, the second layer reads a 3 by 3 first-layer map, whose nine receptive fields together cover a 5 by 5 original input.</desc>',
'<rect width="980" height="660" fill="#f6fafc"/>',
'<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#255872"/></marker></defs>',
'<g font-family="Arial,sans-serif" fill="#17344b">']
def text(x,y,t,size=19,weight='normal'):s.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}">{t}</text>')
def grid(x,y,n,cell,selected=None,nums=False):
 for r in range(n):
  for c in range(n):
   fill='#d7eaf6' if selected is None else ('#ffe3a3' if selected(r,c) else '#ffffff')
   s.append(f'<rect x="{x+c*cell}" y="{y+r*cell}" width="{cell}" height="{cell}" fill="{fill}" stroke="#829eaf"/>')
   if nums:text(x+c*cell+12,y+r*cell+27,str(r*n+c+1),16)
def line(x,y,x2,y2):s.append(f'<line x1="{x}" y1="{y}" x2="{x2}" y2="{y2}" stroke="#255872" stroke-width="3" marker-end="url(#arrow)"/>')
text(30,38,'Stride = how far the window moves on its current input',24,'bold')
for x,start,label in [(35,1,'Stride 1: next start = 1'),(510,2,'Stride 2: next start = 2')]:
 for c in range(7):
  s.append(f'<rect x="{x+c*55}" y="80" width="55" height="45" fill="white" stroke="#829eaf"/>');text(x+c*55+22,109,str(c),18)
 s.append(f'<rect x="{x}" y="75" width="165" height="55" fill="none" stroke="#268074" stroke-width="4"/>')
 s.append(f'<rect x="{x+start*55}" y="135" width="165" height="12" fill="#e39e25"/>')
 line(x+8,160,x+start*55+8,160);text(x,192,label,18,'bold')
text(35,220,'Green: current 3-cell window. Gold: next window. The 2D rule also applies vertically.',17)
s.append('<line x1="30" y1="245" x2="950" y2="245" stroke="#a8bbc6"/>')
text(30,281,'Stacking = feeding one convolution layer into the next',24,'bold')
text(35,315,'Original input: 5 x 5',19,'bold');grid(35,335,5,42,lambda r,c:1<=r<=3 and 1<=c<=3,True)
text(35,570,'Gold: one first-layer cell sees 3 x 3.',16)
text(350,315,'Layer 1 output: 3 x 3',19,'bold');grid(350,375,3,46,lambda r,c:r==1 and c==1)
text(362,408,'63',16);text(408,408,'72',16);text(454,408,'81',16)
text(358,454,'108',16);text(404,454,'117',16);text(450,454,'126',16)
text(358,500,'153',16);text(404,500,'162',16);text(450,500,'171',16)
text(650,315,'Layer 2 output: 1 x 1',19,'bold');grid(715,400,1,74);text(728,444,'1053',18,'bold')
line(260,440,330,440);line(515,440,685,440)
text(268,379,'3 x 3',17);text(268,402,'stride 1',16);text(536,379,'3 x 3',17);text(536,402,'stride 1',16)
text(350,570,'Layer 2 reads all nine Layer 1 cells.',16)
text(35,610,'Together their input patches cover 5 x 5: 3 + 2 = 5, not 3 x 3 = 9.',20,'bold')
text(35,640,'Teaching calculation: both kernels contain ones; bias 0; valid convolution; no dilation or activation.',15)
s.append('</g></svg>');out.write_text('\n'.join(s)+'\n')
print(out)
