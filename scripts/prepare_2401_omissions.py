"""Prepare source crops for omissions found during the nine-page coverage audit."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/2401-delivery/audit'

# Coordinates selected on 1888px-wide overview renders, then verified in the
# full-resolution crops. Pages are one-based in this handwritten audit list.
SPECS=[
 (4,(726,532,65,54),'equipment','unclassified_equipment','', 'Unlabelled compressor mechanical assembly'),
 (5,(584,497,68,54),'equipment','unclassified_equipment','', 'Unlabelled compressor mechanical assembly'),
 (5,(322,697,18,19),'opc','boundary','空气','Filled inlet boundary circle'),
 (5,(1223,443,17,23),'equipment','valve','PSV 001','Safety valve body and spring'),
 (5,(1296,863,35,28),'equipment','valve','FV 00301','Pneumatic control-valve assembly'),
 (6,(834,353,22,55),'equipment','filter','','Inline filter, same source glyph as named filters on page 10'),
 (6,(907,353,23,55),'equipment','filter','','Inline filter'),
 (6,(1200,350,23,57),'equipment','filter','','Inline filter'),
 (6,(834,733,22,55),'equipment','filter','','Inline filter'),
 (6,(907,733,23,55),'equipment','filter','','Inline filter'),
 (6,(1200,731,23,57),'equipment','filter','','Inline filter'),
 (6,(1112,301,79,28),'instrument','controller','PLC控制器','Controller frame and printed label'),
 (6,(1112,690,79,28),'instrument','controller','PLC控制器','Controller frame and printed label'),
 (7,(712,361,17,23),'equipment','valve','PSV 002','Safety valve body and spring'),
 (7,(1142,449,31,26),'equipment','valve','PV 00502','Pneumatic control-valve assembly'),
 (8,(726,351,65,53),'equipment','unclassified_equipment','','Unlabelled compressor mechanical assembly'),
 (8,(186,1076,23,22),'opc','boundary','循环水回水','Filled return boundary circle'),
 (9,(584,429,68,54),'equipment','unclassified_equipment','','Unlabelled compressor mechanical assembly'),
 (9,(322,629,18,19),'opc','boundary','空气','Filled inlet boundary circle'),
 (9,(1223,374,17,23),'equipment','valve','PSV 003','Safety valve body and spring'),
 (9,(1296,794,35,28),'equipment','valve','FV 00801','Pneumatic control-valve assembly'),
 (10,(349,342,34,26),'equipment','valve','PV 00801','Pneumatic control-valve assembly'),
 (10,(471,426,17,38),'equipment','filter','HO-0250F','Printed filter model'),
 (10,(531,426,17,38),'equipment','filter','HT-0250F','Printed filter model'),
 (10,(993,426,17,38),'equipment','filter','HA-0250F','Printed filter model'),
 (10,(638,476,83,74),'equipment','dryer','TZH-25FH','Printed packaged dryer model'),
 (10,(913,374,82,25),'instrument','controller','PC控制系统','Controller frame and printed label'),
 (10,(476,546,8,13),'equipment','valve','','Filter drain valve'),
 (10,(536,546,8,13),'equipment','valve','','Filter drain valve'),
 (10,(633,546,8,13),'equipment','valve','','Dryer drain valve'),
 (10,(997,546,8,13),'equipment','valve','','Filter drain valve'),
 (10,(471,982,17,38),'equipment','filter','HO-0250F','Printed filter model'),
 (10,(531,982,17,38),'equipment','filter','HT-0250F','Printed filter model'),
 (10,(993,982,17,38),'equipment','filter','HA-0250F','Printed filter model'),
 (10,(638,1033,83,74),'equipment','dryer','TZH-25FH','Printed packaged dryer model'),
 (10,(913,931,82,25),'instrument','controller','PC控制系统','Controller frame and printed label'),
 (10,(997,1102,8,13),'equipment','valve','','Filter drain valve'),
 (10,(1523,429,17,23),'equipment','valve','PSV 004','Safety valve body and spring'),
 (11,(919,331,76,25),'instrument','controller','PLC控制器','Controller frame and printed label'),
 (11,(919,859,76,25),'instrument','controller','PLC控制器','Controller frame and printed label'),
 (11,(1271,443,25,67),'equipment','filter','','Inline filter at nitrogen package outlet'),
 (11,(1271,967,25,67),'equipment','filter','','Inline filter at nitrogen package outlet'),
 (11,(1363,416,14,22),'instrument','flow_meter','流量计','Printed flow-meter label'),
 (11,(1363,943,14,22),'instrument','flow_meter','流量计','Printed flow-meter label'),
 (11,(1279,515,10,14),'equipment','valve','','Outlet filter drain valve'),
 (11,(1279,1040,10,14),'equipment','valve','','Outlet filter drain valve'),
 (11,(464,1223,32,26),'equipment','valve','FV 00901','Pneumatic control-valve assembly'),
 (12,(809,479,17,23),'equipment','valve','PSV 005','Safety valve body and spring'),
 (12,(1238,570,32,26),'equipment','valve','PV 00502','Printed PV numbering differs from surrounding loop; preserve source text'),
]


def main():
    scale=6000/1888
    records=[]
    font=ImageFont.truetype('/System/Library/Fonts/STHeiti Light.ttc',15)
    for i,(page,b,kind,cls,label,note) in enumerate(SPECS):
        box=dict(zip(('x','y','w','h'),[round(v*scale) for v in b]))
        records.append(dict(id=f'omission-{i:03d}',page_index=page-1,bbox=box,kind=kind,
                            symbol_class=cls,label=label,note=note))
    for start in range(0,len(records),16):
        canvas=Image.new('RGB',(1800,1200),'white');draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(records[start:start+16]):
            b=r['bbox'];pad=75;page=Image.open(OUT/f'page-{r["page_index"]+1:02d}.png')
            crop=page.crop((b['x']-pad,b['y']-pad,b['x']+b['w']+pad,b['y']+b['h']+pad))
            ImageDraw.Draw(crop).rectangle((pad,pad,pad+b['w'],pad+b['h']),outline='blue',width=3)
            crop.thumbnail((440,250));x=j%4*450;y=j//4*300;canvas.paste(crop,(x+5,y+40))
            draw.text((x+5,y+4),f'{r["id"]} p{r["page_index"]+1}: {r["label"] or r["symbol_class"]}',font=font,fill='black')
        canvas.save(OUT/f'omissions-{start//16+1:02d}.png')
    (OUT/'omissions-proposed.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    print(len(records),'omission source crops prepared')


if __name__=='__main__':main()
