"""

Author: Pedro F. Proenza

Original download script from TACO repo, but modified by me.
"""
from __future__ import annotations
import argparse
import sys
import os
import json
import numpy as np
import copy
from io import BytesIO
from PIL import Image, ExifTags
import requests
from pycocotools.coco import COCO

print('Note. If for any reason the connection is broken. Just call me again and I will start where I left.')

def download_images(args, dataset_dir):
    missing_images = []
    # Load annotations
    with open(args.dataset_path, 'r') as f:
        annotations = json.loads(f.read())

        nr_images = len(annotations['images'])
        i = 0
        while i < nr_images:
            try: 
                image = annotations['images'][i]

                file_name = image['file_name']
                url_original = image['flickr_url']
                url_resized = image['flickr_640_url']

                file_path = os.path.join(dataset_dir, file_name)

                # Create subdir if necessary
                subdir = os.path.dirname(file_path)
                if not os.path.isdir(subdir):
                    os.mkdir(subdir)

                if not os.path.isfile(file_path):
                    # Load and Save Image
                    response = requests.get(url_original)
                    img = Image.open(BytesIO(response.content))
                    if img._getexif():
                        img.save(file_path, exif=img.info["exif"])
                    else:
                        img.save(file_path)
            except Exception as e:
                print(f"Error downloading image {file_name}: {e}")
                missing_images.append(file_name)
               
            # Show loading bar
            bar_size = 30
            x = int(bar_size * i / nr_images)
            sys.stdout.write("%s[%s%s] - %i/%i\r" % ('Loading: ', "=" * x, "." * (bar_size - x), i, nr_images))
            sys.stdout.flush()
            i += 1
        sys.stdout.write('Finished\n')

    # save missing images to a file
    if missing_images:
        missing_path = os.path.join(dataset_dir, 'missing_images.txt')
        missing_images = list(set(missing_images))  # remove duplicates
        with open(missing_path, 'w') as f:
            for item in missing_images:
                f.write("%s\n" % item)
        print(f"Missing images saved to missing_images.txt")
        

download_images(args=argparse.Namespace(dataset_path='data/TACO/annotations.json'), dataset_dir='data/TACO')