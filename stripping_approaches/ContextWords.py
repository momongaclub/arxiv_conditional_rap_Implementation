import argparse
import MeCab
import sys


TAB = '\t'
TITLE = 0


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
            verses.append(verse)
    return verses


def load_stopwords(fname):
    stopwords = []
    with open(fname, 'r') as fp:
        for stopword in fp:
            stopword = stopword.rstrip('\n')
            stopwords.append(stopword)
    return stopwords


class ContextWords():

    def __init__(self, verses_fname, stopwords_fname):
        self.original_verses = load_verses(verses_fname)
        self.stopwords = load_stopwords(stopwords_fname)
        self.preprocessed_verses = []
        self.wakati_verses = []
        self.mecab = MeCab.Tagger("-O wakati")

    def remove_stopwords(self, wakati_line):
        removed_line = []
        for word in wakati_line:
            if word not in self.stopwords:
                removed_line.append(word)
        return removed_line

    def remove_numbers(self):
        return 0

    def remove_punctuation(self):
        return 0

    def preprocessing(self):
        # remove stop words, numbers and punctuation
        # for verse in self.original_rap_verses:
        for verse in self.original_verses:
            wakati_verse = []
            for line in verse:
                wakati_line = self.mecab.parse(line)
                wakati_line = wakati_line.split(' ')
                wakati_line = wakati_line[:-1]
                removed_line = self.remove_stopwords(wakati_line)
                wakati_verse.append(removed_line)
            self.preprocessed_verses.append(wakati_verse)

    def shuffle():
        return 0

    def drop():
        return 0

    def synonym():
        return 0


def main():
    args = parse()
    context_words = ContextWords(args.corpus, args.stopwords)
    context_words.preprocessing()
    print(context_words.preprocessed_verses)


if __name__ == '__main__':
    main()
