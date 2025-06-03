from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

from agent.constants import DEFAULT_LANG, DEFAULT_TOP_K, SENTENCE_TRANSFORMER_MODEL


class TextPreprocessor:

    @classmethod
    def preprocess(cls, text: str, lang=DEFAULT_LANG) -> str:
        if isinstance(text, tuple): text = ' '.join(text)
        text = text.lower()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words(lang))
        tokens = [t for t in tokens if t not in stop_words]

        return ' '.join(tokens)


class Retriever(ABC):

    def __init__(self, name='abstract_retriever'):
        self.name = name

    def get_name(self):
        return self.name

    """
    Este método recibe un conjunto de documentos y los indexa para poder realizar búsquedas posteriores
    """

    @abstractmethod
    def build_index(self, documents: List[str], lang: str = DEFAULT_LANG):
        pass

    """
        Este método búsca los documentos relevantes para una query.
        Devuelve una lista con el la posición (index) del documento encontrado y su score de relevancia.
    """

    @abstractmethod
    def search(self, query: str, top_k: int = DEFAULT_TOP_K, lang: str = DEFAULT_LANG) -> List[Tuple[int, float]]:
        pass

    """
        Este método búsca los documentos relevantes para una query.
        Devuelve los documentos que considera relevantes.
    """

    @abstractmethod
    def search_documents(self, query: str, top_k: int = DEFAULT_TOP_K, lang: str = DEFAULT_LANG) -> List[str]:
        pass


class SparseRetrieverNM(Retriever):

    def __init__(self):
        super().__init__("sparse_retriever_nm")
        self.vectorizer = TfidfVectorizer()
        self.nn_model = NearestNeighbors(n_neighbors=5, metric="cosine", algorithm="auto")

    """
    Construye el índice usando TF-IDF
    """

    def build_index(self, documents: List[str], lang: str = DEFAULT_LANG):
        self.documents = documents
        # Limpiar tokens innecesarios
        processed_docs = [TextPreprocessor.preprocess(doc, lang) for doc in self.documents]
        # Generar embeddings dispersos TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_docs)
        # Construir un modelo de búsqueda eficiente
        self.nn_model.fit(self.tfidf_matrix)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, lang: str = DEFAULT_LANG) -> List[Tuple[int, float]]:
        # Vectorizar la consulta
        processed_query = TextPreprocessor.preprocess(query, lang)
        query_vector = self.vectorizer.transform([processed_query])

        # Encontrar los vecinos más cercanos
        distances, indices = self.nn_model.kneighbors(query_vector, n_neighbors=top_k)
        # Retornar resultados como documentos y distancias inversas (para similitud)
        return [(idx, score) for idx, score in zip(indices[0], distances[0])][::-1]

    def search_documents(self, query: str, top_k: int = DEFAULT_TOP_K, lang: str = DEFAULT_LANG) -> List[str]:
        relevant_documents = self.search(query, top_k, lang)
        return [self.documents[idx] for idx, score in relevant_documents]


class SparseRetriever(Retriever):
    def __init__(self):
        super().__init__('sparse_retriever')

    def build_index(self, documents: List[str], lang: str = DEFAULT_LANG):
        self.documents = documents
        processed_docs = [TextPreprocessor.preprocess(doc, lang) for doc in self.documents]
        tokenized_docs = [doc.split() for doc in processed_docs]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, lang: str = DEFAULT_LANG) -> List[Tuple[int, float]]:
        processed_query = TextPreprocessor.preprocess(query, lang)
        query_tokens = processed_query.split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(idx, scores[idx]) for idx in top_indices]

    def search_documents(self, query: str, top_k: int = DEFAULT_TOP_K, lang: str = DEFAULT_LANG) -> List[str]:
        relevant_documents = self.search(query, top_k, lang)
        return [self.documents[idx] for idx, score in relevant_documents]


class DenseRetriever(Retriever):

    def __init__(self, model=SENTENCE_TRANSFORMER_MODEL):
        super().__init__('dense_retriever' + model)
        # Cargar modelo de embeddings multilingüe
        self.model = SentenceTransformer(model)

    def build_index(self, documents: List[str], lang: str = DEFAULT_LANG):
        # Guardar documentos originales
        self.documents = documents
        #  Procesar texto eliminando tokens no relevantes
        processed_docs = [TextPreprocessor.preprocess(doc, lang) for doc in self.documents]
        # Generar y almacenar embeddings
        self.embeddings = self.model.encode(processed_docs, show_progress_bar=True)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, lang: str = DEFAULT_LANG) -> List[Tuple[int, float]]:
        # Realiza búsqueda por similitud de embeddings.
        processed_query = TextPreprocessor.preprocess(query, lang)
        # Generar embedding de la query
        query_embedding = self.model.encode([processed_query])
        # Calcular similitud
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        # Obtener top_k resultados
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(idx, similarities[idx]) for idx in top_indices]

    def search_documents(self, query: str, top_k: int = DEFAULT_TOP_K, lang: str = DEFAULT_LANG) -> List[str]:
        relevant_documents = self.search(query, top_k, lang)
        return [self.documents[idx] for idx, score in relevant_documents]


class HybridRetriever(Retriever):

    def __init__(self, weight_sparse: float = 0.3,
                 weight_dense: float = 0.7, model=SENTENCE_TRANSFORMER_MODEL):
        super().__init__('hybrid_retriever' + model)
        self.model = model
        self.weight_sparse = weight_sparse
        self.weight_dense = weight_dense

    def build_index(self, documents: List[str], lang: str = DEFAULT_LANG):
        self.sparse_retriever = SparseRetriever()
        self.dense_retriever = DenseRetriever(self.model)
        self.sparse_retriever.build_index(documents)
        self.dense_retriever.build_index(documents)
        self.documents = documents

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, lang: str = DEFAULT_LANG) -> List[Tuple[int, float]]:
        # Obtener resultados de ambos retrievers
        sparse_results = self.sparse_retriever.search(query, top_k=top_k, lang=lang)
        dense_results = self.dense_retriever.search(query, top_k=top_k, lang=lang)

        # Combinar scores
        combined_scores = {}
        for idx, score in sparse_results:
            combined_scores[idx] = score * self.weight_sparse

        for idx, score in dense_results:
            if idx in combined_scores:
                combined_scores[idx] += score * self.weight_dense
            else:
                combined_scores[idx] = score * self.weight_dense

        # Ordenar resultados finales
        sorted_results = sorted(combined_scores.items(),
                                key=lambda x: x[1],
                                reverse=True)[:top_k]
        # Preparar resultados
        return [(idx, score) for idx, score in sorted_results]

    def search_documents(self, query: str, top_k: int = DEFAULT_TOP_K, lang: str = DEFAULT_LANG) -> List[str]:
        relevant_documents = self.search(query, top_k, lang)
        return [self.documents[idx] for idx, score in relevant_documents]

