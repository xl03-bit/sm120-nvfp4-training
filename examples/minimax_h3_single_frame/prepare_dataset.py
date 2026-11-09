from bootstrap import ROOT
from PIL import Image,ImageOps,ImageDraw
import pyarrow.parquet as pq
import io,json,random,hashlib

rows=pq.read_table(ROOT/'dataset/data/train-00000-of-00001.parquet').to_pylist()
order=list(range(len(rows)));random.Random(20261109).shuffle(order)
out=ROOT/'data';out.mkdir(exist_ok=True)
manifest=[]
sheet=Image.new('RGB',(8*180,5*205),'#eeeeee');draw=ImageDraw.Draw(sheet)
for j,i in enumerate(order[:40]):
    row=rows[i]
    im=Image.open(io.BytesIO(row['image']['bytes'])).convert('RGBA')
    bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im);im=bg.convert('RGB')
    im=ImageOps.pad(im,(512,512),color='white',method=Image.Resampling.LANCZOS)
    im.save(out/f'{i:03d}.png')
    caption=row['blip_caption']
    record=dict(id=i,split='train' if j<32 else 'heldout',path=f'data/{i:03d}.png',caption=caption,
        prompt=f'integrated_multimodal_description: [Shot 1] A still illustration in txmnstyle. {caption}. Clean colored cartoon monster illustration. overall_soundscape: Silent. non_diegetic_music: None.')
    manifest.append(record)
    tile=ImageOps.contain(im,(172,172));x=(j%8)*180+4;y=(j//8)*205+4
    sheet.paste(tile,(x,y));draw.text((x,y+176),f'{i:03d} {record["split"]}',fill='black')
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
sheet.save(out/'contact-sheet.png')
print('Prepared 32 train + 8 heldout images, seed 20261109; 512x512 white padding',flush=True)
print('dataset SHA256',hashlib.sha256((ROOT/'dataset/data/train-00000-of-00001.parquet').read_bytes()).hexdigest(),flush=True)
