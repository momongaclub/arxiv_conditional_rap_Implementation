import argparse
import MeCab
import sys
import re
from sklearn.model_selection import train_test_split


TAB = '\t'
TITLE = 0
DATA_DIR = './training_data/'


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', help='corpus')
    parser.add_argument('stopwords', help='stopword from slothlib')
    args = parser.parse_args()
    return args


def load_verses(fname):
    verses = []
    with open(fname, 'r') as fp:
        for verse in fp:
            verse = verse.rstrip('\n')
            verse = verse.rstrip(TAB)
            verse = verse.split(TAB)
            title = verse.pop(TITLE)
            if remove_minlyric(verse):
                verses.append(verse)
    return verses


def load_stopwords(fname):
    stopwords = []
    with open(fname, 'r') as fp:
        for stopword in fp:
            stopword = stopword.rstrip('\n')
            stopwords.append(stopword)
    return stopwords


def remove_minlyric(verse):
    if len(verse) <= 4:
        return False
    else:
        return True


class ContextWords():

    def __init__(self, verses_fname, stopwords_fname):
        self.original_verses = load_verses(verses_fname)
        self.stopwords = load_stopwords(stopwords_fname)
        self.preprocessed_verses = []
        self.all_context_words = []
        self.mecab = MeCab.Tagger("-Ochasen")
        self.mecab_wakati = MeCab.Tagger("-Owakati")
        self.target_pos = ['名詞','動詞','形容詞']

    def mecab_tokenizer(self, sentence):
        self.mecab.parse("")
        tokenized_words = []
        node = self.mecab.parseToNode(sentence)
        while node:
            # 指定品詞のみを取得
            word = node.surface
            pos = node.feature.split(',')[0]
            if pos in self.target_pos:
                tokenized_words.append(word)
            node = node.next
        return tokenized_words

    def remove_noise(self, line):
        p = re.compile('[a-zA-Zａ-ｚＡ-Ｚ0-9０-９!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」｢｣〔〕“”〈〉『』【】＆＊・（）)＄＃＠。、？！｀＋￥％❤]+')
        removed_line = []
        for word in line:
            flag = p.fullmatch(word)
            if flag is None:
                removed_line.append(word)
        return removed_line

    def remove_stopwords(self, wakati_line):
        removed_line = []
        for word in wakati_line:
            if word not in self.stopwords:
                removed_line.append(word)
        return removed_line

    def remove_numbers(self, line):
        return 0

    def remove_punctuation(self):
        # TODO []()も全角含め除く
        return 0

    def preprocessing(self):
        for verse in self.original_verses:
            target_verse = []
            context_words = []
            for line in verse:
                target_tokenized_line = self.mecab_wakati.parse(line)
                target_tokenized_line = target_tokenized_line.split(' ')
                target_tokenized_line = target_tokenized_line[:-1]
                target_verse.extend(target_tokenized_line)
                # 品詞を限定する
                # remove stopwords, numbers and punctuation
                source_tokenized_line = self.mecab_tokenizer(line)
                source_tokenized_line = self.remove_stopwords(source_tokenized_line)
                source_tokenized_line = self.remove_noise(source_tokenized_line)
                context_words.extend(source_tokenized_line)
            # TODO setにするとランダムになっちゃうけどいいのかな
            # setにすべき
            # self.all_context_words.append(set(context_words))
            if len(context_words) != 0:
                self.all_context_words.append(context_words)
                self.preprocessed_verses.append(target_verse)

    def shuffle():
        # sentence levelでのシャッフル。 なので、センテンスを保持する必要性がある
        # つまる話がextendはまずい
        return 0

    def drop():
        return 0

    def synonym():
        return 0

    def write_data(self, src_fname, tgt_fname, src_data, tgt_data):
        # TODO この組み合わせであってる?
        with open(src_fname, 'w') as sfp, open(tgt_fname, 'w') as tfp:
            for src, tgt in zip(src_data, tgt_data):
                src_line = ''
                tgt_line = ''
                for s in src:
                    src_line += s
                    src_line += ' '
                for t in tgt:
                    tgt_line += t
                    tgt_line += ' '
                # print('src:', src_line, 'tgt', tgt_line)
                sfp.write(src_line+'\n')
                tfp.write(tgt_line+'\n')


def main():
    # TODO まず、文が長すぎるので、分割。そして、ターゲッtのバグを直す。
    args = parse()
    context_words = ContextWords(args.corpus, args.stopwords)
    context_words.preprocessing()
    src_train, src_test, \
    tgt_train, tgt_test = train_test_split(
                                    context_words.all_context_words,
                                    context_words.preprocessed_verses,
                                    test_size=0.2,
                                    random_state=0)
    src_valid, src_test, \
    tgt_valid, tgt_test = train_test_split(
                                    src_test,
                                    tgt_test,
                                    test_size=0.5,
                                    random_state=0)
    print('len', len(src_train), len(tgt_train))
    print('len', len(src_valid), len(tgt_valid))
    print('len', len(src_test), len(tgt_test))
    context_words.write_data(DATA_DIR+'src_train', DATA_DIR+'tgt_train', src_train, tgt_train)
    context_words.write_data(DATA_DIR+'src_valid', DATA_DIR+'tgt_valid', src_valid, tgt_valid)
    context_words.write_data(DATA_DIR+'src_test', DATA_DIR+'tgt_test', src_test, src_test)
    # print(context_words.preprocessed_verses)


if __name__ == '__main__':
    main()
