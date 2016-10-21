import json, argparse, os

################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', default='data/shapes_100k_9_22_v2/images/')
parser.add_argument('--splits_json', default='data/shapes_100k_9_22_v2/splits.json')
parser.add_argument('--image_extension', default='png')
parser.add_argument('--output_dir', default='data/shapes_100k_9_22_v2/image_lists')


def main(args):
  with open(args.splits_json, 'r') as f:
    splits = json.load(f)
  if not os.path.isdir(args.output_dir):
    print 'Creating output directory'
    os.mkdir(args.output_dir)

  image_dir = args.image_dir
  ext = args.image_extension
  for split in splits:
    with open(os.path.join(args.output_dir, '%s.txt' % split), 'w') as fout:
      for image_id in splits[split]:
        path = os.path.join(image_dir, '%s.%s' % (image_id, ext))
        fout.write('%s\n' % path)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
