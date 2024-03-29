# import required libraries
from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.reduction import ReductionSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import pandas as pd
from tqdm import tqdm
import os


class Summarizer:
    def __init__(self, language):
        super(Summarizer, self).__init__()
        self.language = language
        self.stemmer = Stemmer(language)

    def setup_summarizer(self, summarizer_name):
        """
        Initialize summarizer.

        Args:
            summarizer_name (str): summarizer name. Must be one of "lexrank", "textrank", "reduction".

        Raises:
            ValueError: invalid summarizer name.
        """
        stemmer = Stemmer(self.language)
        if summarizer_name == "lexrank":
            self.summarizer = LexRankSummarizer(stemmer)
        elif summarizer_name == "textrank":
            self.summarizer = TextRankSummarizer(stemmer)
        elif summarizer_name == "reduction":
            self.summarizer = ReductionSummarizer(stemmer)
        else:
            raise ValueError("Invalid summarizer name")
        self.summarizer.stop_words = get_stop_words(self.language)

    def summarize(self, text, num_sentences):
        """
        Summarize a peice of text.

        Args:
            text (str): text to be summarized.
            num_sentences (int): number of sentences in the summarised text.

        Returns:
            summary (str): summary of the text.
        """
        parser = PlaintextParser.from_string(text, Tokenizer(self.language))
        summary = ' '.join([str(sentence) for sentence in self.summarizer(parser.document, num_sentences)])
        
        while (len(summary.split(' '))>512) and (num_sentences>1):
            num_sentences-=1
            summary = ' '.join([str(sentence) for sentence in self.summarizer(parser.document, num_sentences)])

        return summary
    
    def sum_column(self, column, num_sentences):
        """
        Summarize a pandas column.

        Args:
            column (pandas df): pandas column containing text to be summarized.
            num_sentences (int): number of sentences in the summarised text.

        Returns:
            column: pandas column containing the summarised text.
        """
        tqdm.pandas()
        column = column.progress_apply(lambda x: self.summarize(text = x, num_sentences = num_sentences))
        return column
    
if __name__ == "__main__":
    # setup summarizer
    summarizer = Summarizer("english")

    for algo in ["lexrank", "textrank", "reduction"]:
        if not os.path.exists(f'data/echr/{algo}'):
            os.mkdir(f'data/echr/{algo}')
        for set in ["valid", "train", "test"]:
            for name in ["anon", "non-anon"]:
                # choose algorithm
                summarizer.setup_summarizer(algo)
                # print what is running
                print(f'{algo} {set} {name}')
                # read pickle
                df = pd.read_pickle(f'data/echr/{name}_{set}.pkl')
                # summarise text
                df['summary'] = summarizer.sum_column(df['text'], 10)
                # save pickle
                df.to_pickle(f'data/echr/{algo}/{name}_{set}.pkl')