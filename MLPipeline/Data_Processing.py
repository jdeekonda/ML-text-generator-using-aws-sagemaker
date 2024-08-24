import io
import pandas as pd
import re


class Data_Processing:

    def load_data(self, obj):
        # read file
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        data = df[["Text"]]
        # data.head()
        data["Text"] = [re.sub("[^a-z' ]", "", i.lower()) for i in data["Text"]]
        return data

    def create_seq(self, text, seq_len=10):
        sequences = []
        # if the number of tokens in 'text' is greater than 5
        if len(text.split()) > seq_len:
            for i in range(seq_len, len(text.split())):
                # select sequence of tokens
                seq = text.split()[i - seq_len:i + 1]
                # add to the list
                sequences.append(" ".join(seq))
            return sequences
        # if the number of tokens in 'text' is less than or equal to 10
        else:
            return [text]

    def splitting(self, seq):
        x = []
        y = []
        for s in seq:
            if len(s.split()) == 11:
                x.append(" ".join(s.split()[:-1]))
                y.append(" ".join(s.split()[1:]))
        return x, y
