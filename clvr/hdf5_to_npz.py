import argparse, os, json
import numpy as np
import h5py


parser = argparse.ArgumentParser()
parser.add_argument('--image_list', required=True)
parser.add_argument('--input_h5', required=True)
parser.add_argument('--output_dir', required=True)


def main(args):
  if not os.path.isdir(args.output_dir):
    os.path.makedirs(args.output_dir)
  with open(args.image_list, 'r') as image_list_file:
    with h5py.File(args.input_h5, 'r') as feat_file:
      for i, image_path in enumerate(image_list_file):
        image_path = image_path.strip()
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        feat_path = os.path.join(args.output_dir, '%s.npz' % image_id)
        feats = feat_file['feats'][i]
        np.savez(feat_path, feats)
        if (i + 1) % 50 == 0:
          print 'processed %d images' % (i + 1)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
