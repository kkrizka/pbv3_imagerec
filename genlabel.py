#!/usr/bin/env python

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

import random

import argparse

import os

parser = argparse.ArgumentParser(description='Generate images for training/testing')
parser.add_argument('outDir', help='Output directory')
parser.add_argument('--font', default='fonts/DroidSansMono.ttf', help='Font for test rendering')


args=parser.parse_args()

if not os.path.exists(args.outDir):
    os.mkdir(args.outDir)

HEIGHT=374
WIDTH=650

def generate_image(label,outDir='.'):
    #
    # Check label format
    if len(label)!=7:
        print('Only support 7-character labels!')
        return

    texttop=label[:3]
    textbot=label[3:]

    #
    # Make test image
    im=Image.new('1',(WIDTH,HEIGHT),color=(1,))

    # Berkeley logo
    lbl_logo = Image.open('lbl_logo.png')

    lbl_w=WIDTH/2*0.9
    lbl_h=lbl_logo.size[1]*lbl_w/lbl_logo.size[0]
    lbl_logo=lbl_logo.resize((int(lbl_w),int(lbl_h)))

    im.paste(lbl_logo, (int(WIDTH*1/4-lbl_w/2), int(HEIGHT/2-lbl_h/2)) )

    # Label
    font = ImageFont.truetype(args.font, 100)
    draw = ImageDraw.Draw(im)

    textx=WIDTH*3/4
    texty=HEIGHT/2

    w,h=draw.textsize(texttop,font=font)
    draw.text((textx-w/2,texty-h-10),texttop,fill=(0),font=font)
    w,h=draw.textsize(textbot,font=font)
    draw.text((textx-w/2,texty  +10),textbot,fill=(0),font=font)

    im.save('{}/{}.png'.format(outDir,label),'PNG')


for i in range(1000):
    generate_image('{:07d}'.format(random.randrange(9999999)),args.outDir)
