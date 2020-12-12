import argparse
import copy
import Vowel

import torch
import sys
from transformers import BertForMaskedLM
from transformers import AutoModelForMaskedLM, BertJapaneseTokenizer

SPACE = ' '
SEP = ','
DEL = '\n'
TAB = '\t'
PRED_NUM = 20


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('verses', help='verses')
    args = parser.parse_args()
    return args


class RhymeEnhancement():

    """
    input is lyrics verse V={l0, .., ln} consisting of N tokenized lines;
    number of BERT predictions K to consider
    output is modified V with enchanced rhyming.
    """

    def __init__(self, verses):
        self.tohoku_tokenizer = BertJapaneseTokenizer.from_pretrained(
                    'cl-tohoku/bert-base-japanese-whole-word-masking')
        self.tohoku_bert_model = AutoModelForMaskedLM.from_pretrained(
                    'cl-tohoku/bert-base-japanese-whole-word-masking')
        self.lyrics = self.load_lyrics(verses)
        self.replaced_verses = []
        self.vowel = Vowel.Vowel()
        # self.model = BertForMaskedLM.from_pretrained(BERT_PATH)
        # self.bert_path = BERT_PATH

    def load_lyrics(self, fname):
        lyrics = []
        with open(fname, 'r') as fp:
            for lyric in fp:
                tokenized_verses = []
                lyric = lyric.rstrip(DEL)
                verses = lyric.split(TAB)
                for verse in verses:
                    # TODO fine-tuning方法に合わせるsubword
                    tokenized_verse = self.tohoku_tokenizer.tokenize(verse)
                    tokenized_verses.append(tokenized_verse)
                lyrics.append(tokenized_verses)
        return lyrics

    def mask_verse(self, verse):
        verse[-1] = '[MASK]'
        return verse

    def tohoku_bert_predictions(self, masked_text, k=5):
        self.tohoku_bert_model.eval()
        indexed_tokens = self.tohoku_tokenizer.convert_tokens_to_ids(
                                                          masked_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        masked_index = len(masked_text) - 1
        # Predict
        with torch.no_grad():
            outputs = self.tohoku_bert_model(tokens_tensor)
            predictions = outputs[0]
        _, predicted_indexes = torch.topk(predictions[0, masked_index], k)
        predicted_tokens = self.tohoku_tokenizer.convert_ids_to_tokens(
                                             predicted_indexes.tolist())
        # print(predicted_tokens)
        return predicted_tokens

    def rhyme_length(self, src, tgt):
        # TODO どこの一致で見るか つけかたはいくつかある
        src_vowel = self.vowel.word2vowel(src)
        tgt_vowel = self.vowel.word2vowel(tgt)
        # print('src_vowel', src_vowel, 'tgt_vowel', tgt_vowel)
        score = 0
        reverse_s_vowel = src_vowel[::-1]
        reverse_t_vowel = tgt_vowel[::-1]
        """
        print('reverse_s_vowel', reverse_s_vowel,
              'reverse_t_vowel', reverse_t_vowel)
        """
        for s, t in zip(reverse_s_vowel, reverse_t_vowel):
            # print('s', s, 't', t)
            if s == t:
                score += 1
            else:
                break
        # print('src_pred', src, 'tgt_pred', tgt, 'new_score', score)
        # print('\n')
        return score

    def get_rhyming_replacement(self, lyric, src_idx, tgt_idx, mask):
        # get last a word .
        src = lyric[src_idx][-1]
        tgt = lyric[tgt_idx][-1]
        preds = self.tohoku_bert_predictions(mask, k=20)
        rl_orig = self.rhyme_length(src, tgt)
        # TODO 既存手法はすぐ返しちゃう
        for pred in preds:
            rl_new = self.rhyme_length(pred, tgt)
            if rl_new > rl_orig:
                return pred, rl_new
        return tgt, rl_orig

    def get_rhyming_original_replacement(self, mask1, mask2):
        # get last a word .
        src_preds = self.tohoku_bert_predictions(mask1, k=PRED_NUM)
        tgt_preds = self.tohoku_bert_predictions(mask2, k=PRED_NUM)
        max_list = []
        score = 0
        # すぐ返しちゃう
        for src_pred in src_preds:
            for tgt_pred in tgt_preds:
                new_score = self.rhyme_length(src_pred, tgt_pred)
                if new_score >= score:
                    max_list = [src_pred, tgt_pred]
                    score = new_score
                    # print('max_list', max_list)
        return max_list[0], max_list[1]

    def replace_original(self):
        replaced_lyrics = []
        for l_num, lyric in enumerate(self.lyrics):
            for i in range(0, len(lyric)-1, 2):
                masked_verse_1 = self.mask_verse(lyric[i].copy())
                masked_verse_2 = self.mask_verse(lyric[i+1].copy())
                # print('mask_1', masked_verse_1)
                # print('mask_2', masked_verse_2)
                src, tgt = self.get_rhyming_original_replacement(
                                    masked_verse_1, masked_verse_2)
                self.lyrics[l_num][i][-1] = src
                self.lyrics[l_num][i+1][-1] = tgt


def main():
    args = parse()
    rhyme_enchant = RhymeEnhancement(args.verses)
    rhyme_enchant.replace_original()
    for lyric in rhyme_enchant.lyrics:
        print(lyric)


if __name__ == '__main__':
    main()
