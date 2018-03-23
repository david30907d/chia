def q21_cooccur_matrix(filename='raw_sentences.txt', window=2):
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

	for i in open(filename, 'r'):
		sentence = i.split()
		for index, word in enumerate(sentence):
			if word not in vocab:
				vocab[word] = vocID
				inv_vocab[vocID] = word
				vocID += 1

			for corrIndex in range(index-2 if index-2>=0 else 0, index+window+1 if index+window<=len(sentence)-1 else len(sentence)-1):
				correlationDict[word].append(sentence[corrIndex])

	cooccur_matrix = np.zeros((vocID, vocID))
	for word, correlationList in correlationDict.items():
		for cor in correlationList:
			cooccur_matrix[vocab[word]][vocab[cor]] = 1
	return vocab, inv_vocab, cooccur_matrix

vocab, inv_vocab, cooccur_matrix = q21_cooccur_matrix()

def q22_word_vectors(cooccur_matrix, dim=10):
	import numpy as np
	from sklearn.decomposition import PCA
	'''
	Arguments
	cooccur_matrix: co-occurrence matrix with shape (N, N)
	dim: PCA的維度，預設10維
	Returns
	word_vectors: word vector matrix with shape (N, 10)
	'''
	pca = PCA(n_components=dim)  
	word_vectors = pca.fit_transform(cooccur_matrix) 
	return word_vectors

word_vectors = q22_word_vectors(cooccur_matrix)
print(word_vectors)