class WordVector(object):
    def __init__(self, filename, window=2, dim=10):
        self.filename = filename
        self.window = window
        self.dim = dim
        self.vocab, self.inv_vocab, self.cooccur_matrix = self.q21_cooccur_matrix()
        self.word_vectors = self.q22_word_vectors()

    def __getitem__(self, word):
        return self.word_vectors[self.vocab[word]]

    def q21_cooccur_matrix(self):
        from collections import defaultdict
        import numpy as np
        '''
        Arguments
        filename: the filename of an English article
        window: context window，定義「上下文」的範圍
        Returns
        vocab: 將單字對應到 id
        inv_vocab: 將 id 對應到單字
        cooccur_matrix: NxN np.ndarray，代表 co-occurrence matrix
        '''
        correlationDict = defaultdict(list)
        vocab , inv_vocab= {}, {}
        vocID = 0

        for i in open(self.filename, 'r'):
            sentence = i.split()
            for index, word in enumerate(sentence):
                if word not in vocab:
                    vocab[word] = vocID
                    inv_vocab[vocID] = word
                    vocID += 1

                for corrIndex in range(index-2 if index-2>=0 else 0, index+self.window+1 if index+self.window<=len(sentence)-1 else len(sentence)-1):
                    correlationDict[word].append(sentence[corrIndex])

        cooccur_matrix = np.zeros((vocID, vocID))
        for word, correlationList in correlationDict.items():
            for cor in correlationList:
                cooccur_matrix[vocab[word]][vocab[cor]] = 1

        # turn diagonal to zero
        # cause we all know a word is correlated to itself
        for i in range(0, vocID):
            cooccur_matrix[i][i] = 0
        return vocab, inv_vocab, cooccur_matrix

    def q22_word_vectors(self):
        import numpy as np
        from sklearn.decomposition import PCA
        '''
        Arguments
        cooccur_matrix: co-occurrence matrix with shape (N, N)
        dim: PCA的維度，預設10維
        Returns
        word_vectors: word vector matrix with shape (N, 10)
        '''
        pca = PCA(n_components=self.dim)  
        return pca.fit_transform(self.cooccur_matrix) 

    def most_similar(self, word):
        '''
        Arguments
            word: 要找的單字
            wv: word vector matrix with shape (N, 10)
            vocab: vocabulary
            inv_vocab: inverse vocabulary
        Returns
            ret: 長度為3的list，每個元素都是 tuple (word, similarity)
        '''
        from scipy.spatial.distance import cosine

        table = {}
        u = self.word_vectors[self.vocab[word]]

        for vocID, v in enumerate(self.word_vectors):
            if vocID != self.vocab[word]:
                table[self.inv_vocab[vocID]] = (-cosine(u, v)+1)
        return sorted(table.items(), key=lambda x:-x[1])[:3]

wv = WordVector('raw_sentences.txt')
print(wv['No'])
for word, sim in wv.most_similar('No'):
    print(word, sim)