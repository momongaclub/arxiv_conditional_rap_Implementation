import argparse
import MeCab

DEL = '\n'
SPACE = ' '


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('verses',
                        help = 'verses')
    args = parser.parse_args()
    return args


def load_verses(fname):
    verses = []
    with open(fname, 'r') as fp:
        for verse in fp:
            verse = verse.rstrip(DEL)
            verse = verse.split(SPACE)
            verses.append(verse)
    return verses


class RhymeEnhancement():

    """
    input is lyrics verse V={l0, .., ln} consisting of N tokenized lines;
    number of BERT predictions K to consider
    output is modified V with enchanced rhyming.
    """

    def __init__(self, verses, k):
        # self.bert_path = bert_path
        self.verses = load_verses(verses)
        self.k = k

    def get_rhyming_replacement(self, verse, src_idx, tgt_idx, mask):
        # get last a word .
        src = verse[src_idx][-1]
        tgt = verse[tgt_idx][-1]
        preds = bert_predictions(mask, K)
        rl_orig = rhyme_length(src, tgt)
        for pred in preds:
            rl_new = rhyme_length(pred, tgt)
            if rl_new > rl_orig:
                return pred, rl_new
        return tgt, rl_orig

    def replace(self):
        # odd for
        for i in range(1, len(self.verses), 2):
            mask_1 = mask_text(verse, i)
            mask_2 = mask_text(verse, i+1)
            cand_1, rl_1 = get_rhyming_replacement(verse, i+1, i, mask_1)
            cand_2, rl_2 = get_rhyming_replacement(verse, i, i+1, mask_2)
            if rl_2 >= rl_1:
                verse[i+1][-1] = cand_2
            else:
                verse[i][-1] = cand_1
        return verse


def main():
    args = parse()
    rhyme_enchant = RhymeEnhancement(args.verses, k=4)
    rhyme_enchant.replace()
    print(rhyme_enchant.verses)


if __name__ == '__main__':
    main()
