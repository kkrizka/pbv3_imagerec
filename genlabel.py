#!/usr/bin/env python
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

import random
import argparse
import os
import glob
import cv2
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description='Generate images for training/testing')
parser.add_argument('outDir', help='Output directory')
parser.add_argument('--font', default='fonts/DroidSansMono.ttf', help='Font for test rendering')


args = parser.parse_args()

if not os.path.exists(args.outDir):
    os.mkdir(args.outDir)

HEIGHT = 374
WIDTH = 650

def generate_image(label,outDir='.'):
    #
    # Check label format
    if len(label) != 7:
        print('Only support 7-character labels!')
        return

    texttop = label[:3]
    textbot = label[3:]

    #
    # Make test image
    im=Image.new('1',(WIDTH,HEIGHT),color=(1,))

    # Berkeley logo
    lbl_logo = Image.open('lbl_logo.png')

    lbl_w = WIDTH/2*0.8
    lbl_h = lbl_logo.size[1]*lbl_w/lbl_logo.size[0]
    lbl_logo = lbl_logo.resize((int(lbl_w),int(lbl_h)))

    im.paste(lbl_logo, (int(WIDTH*1/4-lbl_w/2), int(HEIGHT/2-lbl_h/2)) )

    # Label
    font = ImageFont.truetype(args.font, 135)
    draw = ImageDraw.Draw(im)

    textx = WIDTH*7/10
    texty = HEIGHT/2

    w,h = draw.textsize(texttop,font=font)
    draw.text((textx-w/2,texty-h-10),texttop,fill=(0),font=font)
    w,h = draw.textsize(textbot,font=font)
    draw.text((textx-w/2,texty  +10),textbot,fill=(0),font=font)

    #saving PIL image and reading as cv2 image to apply blur
    #outDir for me was train_var
    im.save('{}/{}.png'.format(outDir,label),'PNG')
    imcv = cv2.imread(f'/Users/ameyakunder/pbv3_imagerec/pbv3_compvision/train_var/{label}.png')
    i = random.randrange(4)
    #randomly blurs image
    if i == 0:
        imcv = cv2.GaussianBlur(imcv,(3,3),0)
    if i == 1:
        imcv = cv2.GaussianBlur(imcv,(5,5),0)

    write_add = f"/Users/ameyakunder/pbv3_imagerec/pbv3_compvision/train/trainimages/{label}.png"
    cv2.imwrite(write_add, imcv)

for i in range(1000):
    generate_image('{:07d}'.format(random.randrange(9999999)),args.outDir)

#image augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
rescale=1./255,
rotation_range=3,
width_shift_range=.1,
height_shift_range=.1,
horizontal_flip=False,
zoom_range=0.1)

#creating and saving the augmented images
i = 0
read_path = "/Users/ameyakunder/pbv3_imagerec/pbv3_compvision/train/trainimages/*.png"
for x in glob.glob(read_path):
    #getting the serial number
    y = x.split("/", 7)[7]
    print(f"serial number : {y}")
    pic = tf.keras.preprocessing.image.load_img(f"/Users/ameyakunder/pbv3_imagerec/pbv3_compvision/train/trainimages/{y}")
    pic_array = tf.keras.preprocessing.image.img_to_array(pic)
    pic_array = pic_array.reshape((1,) + pic_array.shape) # Converting into 4 dimension array
    count = 0
    for batch in train_datagen.flow(
        pic_array, 
        batch_size=5,
        save_to_dir="/Users/ameyakunder/pbv3_imagerec/pbv3_compvision/aug_train", 
        save_prefix=y, 
        save_format='png'):
        count += 1
        if count > 3:
            break
    

