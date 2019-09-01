from PIL import Image
import os
import argparse


def rotate_imgs(root, dir):
    img_count = 0
    for filename in os.listdir(os.path.join(root, dir)):
        print('processed images ', img_count)
        img_count+=1
        if filename.endswith(".jpg"):
            im = Image.open(os.path.join(os.path.join(root, dir), filename))
            width, height = im.size
            if height > width:
                transposed = im.transpose(Image.ROTATE_90)
                transposed.save(os.path.join(os.path.join(root, dir), filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess CLEF dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--root_dir', default=os.path.abspath(os.path.dirname(__file__)))

    parser.add_argument('--dir', default='')

    args = parser.parse_args()
    rotate_imgs(args.root_dir, args.dir)
