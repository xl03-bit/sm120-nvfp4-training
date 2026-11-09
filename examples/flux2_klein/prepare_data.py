"""Download the pinned five-image subject set, verify hashes, and crop it."""
from bootstrap import ROOT
from pathlib import Path
import hashlib,json,shutil,urllib.request
from PIL import Image,ImageOps

HERE=Path(__file__).resolve().parent
def main():
    protocol=ROOT/'protocol.json'
    if not protocol.exists():shutil.copyfile(HERE/'protocol.json',protocol)
    cfg=json.loads(protocol.read_text())
    manifest=json.loads((HERE/'subject-sources.json').read_text())
    original=ROOT/'data/monster_toy/original';original.mkdir(parents=True,exist_ok=True)
    crops=ROOT/'data/monster_toy'/f'crops{cfg["resolution"]}';crops.mkdir(exist_ok=True)
    for row in manifest['files']:
        path=original/row['file']
        data=path.read_bytes() if path.exists() else urllib.request.urlopen(row['source'],timeout=90).read()
        if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Source checksum mismatch: '+row['file'])
        if not path.exists():path.write_bytes(data)
        im=ImageOps.exif_transpose(Image.open(path)).convert('RGB')
        ImageOps.fit(im,(cfg['resolution'],cfg['resolution']),method=Image.Resampling.BICUBIC,centering=(.5,.5)).save(crops/(path.stem+'.png'))
    print('Verified and prepared the four training images and one heldout image.')
if __name__=='__main__':main()
