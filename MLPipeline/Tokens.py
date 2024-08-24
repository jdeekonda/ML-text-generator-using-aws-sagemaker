class Tokens:

    def tokens(self, seq):
        int2token = {}
        cnt = 0
        for w in set(" ".join(seq).split()):
            int2token[cnt] = w
            cnt += 1
        # create token-to-integer mapping
        token2int = {t: i for i, t in int2token.items()}

        return int2token, token2int


    def get_integer_seq(self, seq):
        from Engine import token2int
        return [token2int[w] for w in seq.split()]