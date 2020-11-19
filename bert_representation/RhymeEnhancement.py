import argparse
import copy
import Vowel

import torch
import sys
from transformers import BertForMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertJapaneseTokenizer

BERT_PATH = '/home/mibayashi/bert_text_generation/Japanese_L-24_H-1024_A-16_E-30_BPE_WWM_transformers'
SPACE = ' '
SEP = ','
DEL = '\n'
TAB = '\t'


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('verses',
                        help = 'verses')
    args = parser.parse_args()
    return args


class RhymeEnhancement():

    """
    input is lyrics verse V={l0, .., ln} consisting of N tokenized lines;
    number of BERT predictions K to consider
    output is modified V with enchanced rhyming.
    """

    def __init__(self, verses):
        self.tohoku_tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        # TODO Auto?
        self.tohoku_bert_model = AutoModelForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        self.bert_path = BERT_PATH
        self.lyrics = self.load_lyrics(verses)
        self.replaced_verses = []
        self.model = BertForMaskedLM.from_pretrained(BERT_PATH)
        self.vowel = Vowel.Vowel()

    def load_lyrics(self, fname):
        lyrics = []
        with open(fname, 'r') as fp:
            for lyric in fp:
                tokenized_verses = []
                lyric = lyric.rstrip(DEL)
                verses = lyric.split(TAB)
                for verse in verses:
                    print('verse', verse)
                    # TODO subwordがややこいので普通のmecabがいいかも？
                    tokenized_verse = self.tohoku_tokenizer.tokenize(verse)
                    tokenized_verses.append(tokenized_verse)
                lyrics.append(tokenized_verses)
        return lyrics

    def mask_verse(self, verse):
        verse[-1] = '[MASK]'
        return verse

    def bert_predictions(self, masked_text, k=5):
        self.model.eval()
        indexed_tokens = self.tohoku_tokenizer.convert_tokens_to_ids(masked_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        masked_index = len(masked_text) - 1
        # Predict
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            predictions = outputs[0]
        _, predicted_indexes = torch.topk(predictions[0, masked_index], k)
        predicted_tokens = self.tohoku_tokenizer.convert_ids_to_tokens(predicted_indexes.tolist())
        print(predicted_tokens)
        return predicted_tokens

    def rhyme_length(self, src, tgt):
        # TODO どこの一致で見るか つけかたはいくつかある
        src_yomi = self.vowel.word2yomi(src)
        tgt_yomi = self.vowel.word2yomi(tgt)
        score = 0
        for s, t in zip(src_yomi[:-1], tgt_yomi[:-1]):
            if s == t:
                score += 1
        return score

    def get_rhyming_replacement(self, lyric, src_idx, tgt_idx, mask):
        # get last a word .
        src = lyric[src_idx][-1]
        tgt = lyric[tgt_idx][-1]
        preds = self.bert_predictions(mask, k=20)
        rl_orig = self.rhyme_length(src, tgt)
        for pred in preds:
            rl_new = self.rhyme_length(pred, tgt)
            if rl_new > rl_orig:
                return pred, rl_new
        return tgt, rl_orig

    def replace(self):
        replaced_lyrics = []
        for l_num, lyric in enumerate(self.lyrics):
            for i in range(0, len(lyric), 2):
                masked_verse_1 = self.mask_verse(lyric[i].copy())
                masked_verse_2 = self.mask_verse(lyric[i+1].copy())
                print('mask_1', masked_verse_1)
                print('mask_2', masked_verse_2)
                cand_1, rl_1 = self.get_rhyming_replacement(lyric, i+1, i, masked_verse_1)
                cand_2, rl_2 = self.get_rhyming_replacement(lyric, i, i+1, masked_verse_2)
                print('cand_1', cand_1)
                print('cand_2', cand_2)
                if rl_2 >= rl_1:
                    lyric[i+1][-1] = cand_2
                    self.lyrics[l_num][i+1][-1] = cand_2
                else:
                    lyric[i][-1] = cand_1
                    self.lyrics[l_num][i][-1] = cand_1


def main():
    args = parse()
    rhyme_enchant = RhymeEnhancement(args.verses)
    rhyme_enchant.replace()
    print(rhyme_enchant.lyrics)


if __name__ == '__main__':
    main()
