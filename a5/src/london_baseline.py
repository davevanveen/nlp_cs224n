# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import utils

if __name__ == '__main__':
    
    # data on correct answers
    args_eval_corpus_path = "../birth_dev.tsv"

    # get correct answer length
    with open(args_eval_corpus_path) as fn:
        lines = [x.strip().split('\t') for x in fn]
    pred_len = len(lines)

    # calculate model correctness
    preds = ["London"] * pred_len 
    total, correct = utils.evaluate_places(args_eval_corpus_path, preds)
    print('Correct: {} out of {}: {}%'.format(correct, total, 100*(correct/total)))
