import argparse, json, os


"""
Convert my question format to VQA format; this is needed to interface with
existing codebases such as Neural Module Nets.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--input_questions',
        default='data/shapes_100k_9_22_v2/questions.json')
parser.add_argument('--splits_json',
        default='data/shapes_100k_9_22_v2/splits.json')
parser.add_argument('--output_question_dir',
        default='data/shapes_100k_9_22_v2/Questions')
parser.add_argument('--output_annotation_dir',
        default='data/shapes_100k_9_22_v2/Annotations')


def main(args):
  # Read input data
  with open(args.splits_json, 'r') as f:
    splits = json.load(f)
  with open(args.input_questions, 'r') as f:
    input_questions = json.load(f)

  image_id_to_split = {}
  for split, image_ids in splits.iteritems():
    for image_id in image_ids:
      image_id_to_split[image_id] = split

  split_to_questions = {}
  split_to_anns = {}

  for question in input_questions:
    image_id = question['image']
    split = image_id_to_split[image_id]
    if split not in split_to_questions:
      split_to_questions[split] = {'questions': []}
      split_to_anns[split] = {'annotations': []}
    split_to_questions[split]['questions'].append({
      'image_id': int(image_id),
      'question': question['text_question'],
      'question_id': question['question_id'],
    })
    split_to_anns[split]['annotations'].append({
      'question_type': question['structured_question'][-1]['type'],
      'answers': [
        {
          'answer': question['answer'],
          'answer_confidence': 'yes',
          'answer_id': 1,
        },
      ],
      'image_id': int(image_id),
      'answer_type': 'other',
      'question_id': question['question_id'],
    })

  if not os.path.isdir(args.output_question_dir):
    os.makedirs(args.output_question_dir)
  for split, questions in split_to_questions.iteritems():
    path = os.path.join(args.output_question_dir, '%s_questions.json' % split)
    with open(path, 'w') as f:
      json.dump(questions, f)
  
  if not os.path.isdir(args.output_annotation_dir):
    os.makedirs(args.output_annotation_dir)
  for split, anns in split_to_anns.iteritems():
    path = os.path.join(args.output_annotation_dir, '%s_annotations.json' % split)
    with open(path, 'w') as f:
      json.dump(anns, f)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
