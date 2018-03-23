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

	# turn diagonal to zero
	# cause we all know a word is correlated to itself
	for i in range(0, vocID):
		cooccur_matrix[i][i] = 0
	return vocab, inv_vocab, cooccur_matrix

vocab, inv_vocab, cooccur_matrix = q21_cooccur_matrix()
print(vocab, inv_vocab, cooccur_matrix)

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
# print(word_vectors)

def q23_similarity(word, wv=word_vectors, vocab=vocab, inv_vocab=inv_vocab):
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
	u = word_vectors[vocab[word]]

	for vocID, v in enumerate(word_vectors):
		if vocID != vocab[word]:
			table[inv_vocab[vocID]] = (-cosine(u, v)+1)
	return sorted(table.items(), key=lambda x:-x[1])[:3]

word = 'No'
for word, sim in q23_similarity(word):
	print(word, sim)