
from PIL import Image
import os.path
import glob
def convertjpg(pngfile, outdir, width=64, height=64):
    img=Image.open(pngfile)
    try:
        new_img=img.resize((width, height), Image.BILINEAR)
        new_img.save(os.path.join(outdir, os.path.basename(pngfile)))
    except Exception as e:
        print(e)


# /Users/KentPeng/Documents/medical/malaria/cell_images/Parasitized/*.png
for pngfile in glob.glob('/Users/KentPeng/Documents/medical/malaria/cell_images/Uninfected/*.png'):
    convertjpg(pngfile, '/Users/KentPeng/Documents/medical/malaria/cell_images/Uninfected1/')

