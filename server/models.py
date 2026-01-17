import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class TFIDFModel:
    """TF-IDF реализация на чистом numpy"""
    
    def fit_transform(self, texts, max_features=100):
        # 1. Построение словаря
        word_freq = defaultdict(int)
        for text in texts:
            words = text.lower().split()
            for word in set(words):
                word_freq[word] += 1
        
        # Top N слов
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_features]
        vocab = {word: i for i, (word, _) in enumerate(top_words)}
        
        # 2. TF матрица
        n_docs = len(texts)
        n_words = len(vocab)
        tf = np.zeros((n_docs, n_words))
        
        for doc_idx, text in enumerate(texts):
            words = text.lower().split()
            total_words = len(words)
            
            for word in words:
                if word in vocab:
                    tf[doc_idx, vocab[word]] += 1
            
            if total_words > 0:
                tf[doc_idx] /= total_words
        
        # 3. IDF вычисление
        doc_freq = np.zeros(n_words)
        for word, idx in vocab.items():
            count = sum(1 for text in texts if word in text.lower().split())
            doc_freq[idx] = count
        
        idf = np.log((n_docs + 1) / (doc_freq + 1)) + 1
        
        # 4. TF-IDF матрица
        tfidf = tf * idf
        
        # Нормализация
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        tfidf_norm = np.divide(tfidf, norms, where=norms!=0)
        
        # Инвертируем словарь для вывода
        inv_vocab = {i: word for word, i in vocab.items()}
        
        return tfidf_norm, inv_vocab

class BOWModel:
    """Bag of Words реализация на чистом numpy"""
    
    def fit_transform(self, texts, max_features=100):
        # Построение словаря
        word_freq = defaultdict(int)
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freq[word] += 1
        
        # Top N слов
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_features]
        vocab = {word: i for i, (word, _) in enumerate(top_words)}
        
        # Создание матрицы BOW
        n_docs = len(texts)
        n_words = len(vocab)
        bow = np.zeros((n_docs, n_words))
        
        for doc_idx, text in enumerate(texts):
            words = text.lower().split()
            for word in words:
                if word in vocab:
                    bow[doc_idx, vocab[word]] += 1
        
        # Инвертируем словарь для вывода
        inv_vocab = {i: word for word, i in vocab.items()}
        
        return bow, inv_vocab

class LSAModel:
    """LSA через sklearn"""
    
    def fit_transform(self, texts, max_features=100, n_components=5):
        # TF-IDF через sklearn для простоты
        vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # LSA (SVD)
        lsa = TruncatedSVD(n_components=n_components, random_state=42)
        transformed = lsa.fit_transform(tfidf_matrix)
        
        return {
            "components": lsa.components_,
            "transformed": transformed,
            "variance": lsa.explained_variance_ratio_
        }
