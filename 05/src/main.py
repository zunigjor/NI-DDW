import copy
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sentence_transformers import SentenceTransformer


TOP_N = 15
OUT_FILE = "../results/results.txt"


def loadData(path):
    res = []
    for i in range(1, len(os.listdir(path)) + 1):
        with open(os.path.join(path, f"{i}.txt")) as f:
            res.append(f.read())
    return res


def printList(list):
    return '[' + ', '.join(map(str, list)) + ']'


def printTermAndFile(string):
    print(string)
    with open(OUT_FILE, 'a') as f:
        print(string, file=f)


def printRes(name, euc, cos, relevance_list):
    printTermAndFile(f"{name}:")
    printTermAndFile(f"  Euclidian:    {printList(euc)}")
    printTermAndFile(f"    - Precision:    {getPrecision(euc, relevance_list)}")
    printTermAndFile(f"    - Recall:       {getRecall(euc, relevance_list)}")
    printTermAndFile(f"    - F-Measure:    {getFMeasure(euc, relevance_list)}")
    printTermAndFile(f"  Cosine:       {printList(cos)}")
    printTermAndFile(f"    - Precision:    {getPrecision(cos, relevance_list)}")
    printTermAndFile(f"    - Recall:       {getRecall(cos, relevance_list)}")
    printTermAndFile(f"    - F-Measure:    {getFMeasure(cos, relevance_list)}")
    printTermAndFile("-" * 60)

def printMean(name, euc_precision, euc_recall, euc_FMeasure, cos_precision, cos_recall, cos_FMeasure):
    printTermAndFile(f"MEAN {name}:")
    printTermAndFile(f"  Euclidian:")
    printTermAndFile(f"    - Mean Precision:    {np.mean(euc_precision)}")
    printTermAndFile(f"    - Mean Recall:       {np.mean(euc_recall)}")
    printTermAndFile(f"    - Mean F-Measure:    {np.mean(euc_FMeasure)}")
    printTermAndFile(f"  Cosine:")
    printTermAndFile(f"    - Mean Precision:    {np.mean(cos_precision)}")
    printTermAndFile(f"    - Mean Recall:       {np.mean(cos_recall)}")
    printTermAndFile(f"    - Mean F-Measure:    {np.mean(cos_FMeasure)}")
    printTermAndFile("-" * 60)


def getPrecision(retrieved_list, relevance_list):
    intersection = len(set(retrieved_list).intersection(set(relevance_list)))
    return intersection / len(retrieved_list)


def getRecall(retrieved_list, relevance_list):
    intersection = len(set(retrieved_list).intersection(set(relevance_list)))
    return intersection / len(relevance_list)


def getFMeasure(retrieved_list, relevance_list):
    precision = getPrecision(retrieved_list, relevance_list)
    recall = getRecall(retrieved_list, relevance_list)
    if precision == 0 and recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def getWeightingResults(data_matrix):
    # Returns sorted similarity between query and all docs.
    data_vectors = data_matrix[0:data_matrix.shape[0] - 1]
    query_vector = data_matrix[data_matrix.shape[0] - 1]
    euc_sim = np.array(euclidean_distances(query_vector, data_vectors)[0]).argsort()[::-1] + 1
    cos_sim = np.array(cosine_similarity(query_vector, data_vectors)[0]).argsort()[::-1] + 1
    return euc_sim[:TOP_N], cos_sim[:TOP_N]


def Binary(documents, query):
    corpus = copy.deepcopy(documents)
    corpus.append(query)
    vectorizer = CountVectorizer(binary=True)
    binary_matrix = vectorizer.fit_transform(corpus)
    return getWeightingResults(binary_matrix)


def TermFrequency(documents, query):
    corpus = copy.deepcopy(documents)
    corpus.append(query)
    vectorizer = CountVectorizer()
    tf_matrix = vectorizer.fit_transform(corpus)
    return getWeightingResults(tf_matrix)


def TF_IDF(documents, query):
    corpus = copy.deepcopy(documents)
    corpus.append(query)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    return getWeightingResults(tfidf_matrix)


def distilbert(documents, query):
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    corpus = copy.deepcopy(documents)
    corpus.append(query)
    corpus_encode = model.encode(corpus, convert_to_tensor=True)
    query_encoding = corpus_encode[-1]
    doc_encodings = corpus_encode[:-1]
    cosine_sims = np.dot(doc_encodings, query_encoding.T).squeeze()
    euclidean_dists = euclidean_distances(doc_encodings, query_encoding.reshape(1, -1)).squeeze()
    cosine_top_relevant = cosine_sims.argsort()[-10:][::-1] + 1
    euclidean_top_relevant = euclidean_dists.argsort()[:10] + 1
    return cosine_top_relevant, euclidean_top_relevant


if __name__ == "__main__":
    with open(OUT_FILE, 'w') as f:
        print("", file=f)
    printTermAndFile("=" * 60)
    documents = loadData("./data/d")
    queries = loadData("./data/q")
    relevances = loadData("./data/r")
    printTermAndFile(f"Documents: {len(documents)}")
    printTermAndFile(f"Queries: {len(queries)}")
    printTermAndFile(f"Relevances: {len(relevances)}")
    # stats
    binary_euc_precision_sum = []
    binary_euc_recall_sum = []
    binary_euc_FMeasure_sum = []
    binary_cos_precision_sum = []
    binary_cos_recall_sum = []
    binary_cos_FMeasure_sum = []

    TF_euc_precision_sum = []
    TF_euc_recall_sum = []
    TF_euc_FMeasure_sum = []
    TF_cos_precision_sum = []
    TF_cos_recall_sum = []
    TF_cos_FMeasure_sum = []

    TF_IDF_euc_precision_sum = []
    TF_IDF_euc_recall_sum = []
    TF_IDF_euc_FMeasure_sum = []
    TF_IDF_cos_precision_sum = []
    TF_IDF_cos_recall_sum = []
    TF_IDF_cos_FMeasure_sum = []

    distilbert_euc_precision_sum = []
    distilbert_euc_recall_sum = []
    distilbert_euc_FMeasure_sum = []
    distilbert_cos_precision_sum = []
    distilbert_cos_recall_sum = []
    distilbert_cos_FMeasure_sum = []
    # Loop queries and find rel. documents
    for i, query in enumerate(queries):
        printTermAndFile("=" * 60)
        relevance_list = np.array([int(x) for x in relevances[i].split(sep='\n')[:-1]])
        printTermAndFile(f"Query {i + 1}")
        printTermAndFile(f"Relevance list: {printList(relevance_list)}")
        printTermAndFile("-" * 60)

        # Binary
        binary_euc, binary_cos = Binary(documents, query)
        # stats
        binary_euc_precision_sum.append(getPrecision(binary_euc, relevance_list))
        binary_euc_recall_sum.append(getRecall(binary_euc, relevance_list))
        binary_euc_FMeasure_sum.append(getFMeasure(binary_euc, relevance_list))
        binary_cos_precision_sum.append(getPrecision(binary_cos, relevance_list))
        binary_cos_recall_sum.append(getRecall(binary_cos, relevance_list))
        binary_cos_FMeasure_sum.append(getFMeasure(binary_cos, relevance_list))
        # print
        printRes("Binary", binary_euc, binary_cos, relevance_list)

        # TF
        TF_euc, TF_cos = TermFrequency(documents, query)
        # stats
        TF_euc_precision_sum.append(getPrecision(TF_euc, relevance_list))
        TF_euc_recall_sum.append(getRecall(TF_euc, relevance_list))
        TF_euc_FMeasure_sum.append(getFMeasure(TF_euc, relevance_list))
        TF_cos_precision_sum.append(getPrecision(TF_cos, relevance_list))
        TF_cos_recall_sum.append(getRecall(TF_cos, relevance_list))
        TF_cos_FMeasure_sum.append(getFMeasure(TF_cos, relevance_list))
        # print
        printRes("TF", TF_euc, TF_cos, relevance_list)

        # TF_IDF
        TF_IDF_euc, TF_IDF_cos = TF_IDF(documents, query)
        # stats
        TF_IDF_euc_precision_sum.append(getPrecision(TF_IDF_euc, relevance_list))
        TF_IDF_euc_recall_sum.append(getRecall(TF_IDF_euc, relevance_list))
        TF_IDF_euc_FMeasure_sum.append(getFMeasure(TF_IDF_euc, relevance_list))
        TF_IDF_cos_precision_sum.append(getPrecision(TF_IDF_cos, relevance_list))
        TF_IDF_cos_recall_sum.append(getRecall(TF_IDF_cos, relevance_list))
        TF_IDF_cos_FMeasure_sum.append(getFMeasure(TF_IDF_cos, relevance_list))
        # print
        printRes("TF_IDF", TF_IDF_euc, TF_IDF_cos, relevance_list)
        
        # distilbert-base-nli-stsb-mean-tokens
        distilbert_euc, distilbert_cos = distilbert(documents, query)
        distilbert_euc_precision_sum.append(getPrecision(distilbert_euc, relevance_list))
        distilbert_euc_recall_sum.append(getRecall(distilbert_euc, relevance_list))
        distilbert_euc_FMeasure_sum.append(getFMeasure(distilbert_euc, relevance_list))
        distilbert_cos_precision_sum.append(getPrecision(distilbert_cos, relevance_list))
        distilbert_cos_recall_sum.append(getRecall(distilbert_cos, relevance_list))
        distilbert_cos_FMeasure_sum.append(getFMeasure(distilbert_cos, relevance_list))
        # print
        printRes("distilbert-base-nli-stsb-mean-tokens", distilbert_euc, distilbert_cos, relevance_list)
    printTermAndFile("=" * 60)
    printMean("Binary",
              binary_euc_precision_sum, binary_euc_recall_sum, binary_euc_FMeasure_sum,
              binary_cos_precision_sum, binary_cos_recall_sum, binary_cos_FMeasure_sum)
    printMean("TF",
              TF_euc_precision_sum, TF_euc_recall_sum, TF_euc_FMeasure_sum,
              TF_cos_precision_sum, TF_cos_recall_sum, TF_cos_FMeasure_sum)
    printMean("TF_IDF",
              TF_IDF_euc_precision_sum, TF_IDF_euc_recall_sum, TF_IDF_euc_FMeasure_sum,
              TF_IDF_cos_precision_sum, TF_IDF_cos_recall_sum, TF_IDF_cos_FMeasure_sum)
    printMean("distilbert-base-nli-stsb-mean-tokens",
              distilbert_euc_precision_sum, distilbert_euc_recall_sum, distilbert_euc_FMeasure_sum,
              distilbert_cos_precision_sum, distilbert_cos_recall_sum, distilbert_cos_FMeasure_sum)
    printTermAndFile("=" * 60)
    exit(0)
