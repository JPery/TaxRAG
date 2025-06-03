import os
import glob
import pickle

from agent.constants import DATA_FOLDER, DEFAULT_LANG
from agent.retrievers import HybridRetriever, SparseRetriever, DenseRetriever


def load_and_preprocess_data(path):
    texts = []
    for item in sorted(glob.glob(f"{path}/*.txt"), key=lambda x: (x.rsplit("_", 1)[0], int(x.rsplit(".", 1)[0].rsplit("_", 1)[-1]))):
        with open(item, "r", encoding="utf8") as f:
            text = f.read()
            texts.append(text)
    return texts


def save_retriever(retriever, save_dir='saved_retriever'):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the documents
    with open(os.path.join(save_dir, 'documents.pkl'), 'wb') as f:
        pickle.dump(retriever.documents, f)

    # Save sparse retriever data
    with open(os.path.join(save_dir, 'sparse_bm25.pkl'), 'wb') as f:
        pickle.dump(retriever.sparse_retriever.bm25, f)

    # Save dense retriever embeddings
    with open(os.path.join(save_dir, 'dense_embeddings.pkl'), 'wb') as f:
        pickle.dump(retriever.dense_retriever.embeddings, f)

    # Save model name and weights
    config = {
        'model_name': retriever.model,
        'weight_sparse': retriever.weight_sparse,
        'weight_dense': retriever.weight_dense
    }
    with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    print(f"Retriever saved in {save_dir}")


def load_retriever(save_dir='saved_retriever'):
    # Load config
    with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)

    # Initialize retriever with saved config
    retriever = HybridRetriever(
        weight_sparse=config['weight_sparse'],
        weight_dense=config['weight_dense'],
        model=config['model_name']
    )

    retriever.sparse_retriever = SparseRetriever()
    retriever.dense_retriever = DenseRetriever(config['model_name'])

    # Load documents
    documents_path = os.path.join(save_dir, 'documents.pkl')
    if os.path.exists(documents_path):
        with open(documents_path, 'rb') as f:
            retriever.documents = pickle.load(f)
    else:
        retriever.documents = []

    # Initialize index if does not exist the path
    if not os.path.exists(os.path.join(save_dir, 'sparse_bm25.pkl')) or not os.path.exists(
            os.path.join(save_dir, 'dense_embeddings.pkl')):
        retriever.build_index(load_and_preprocess_data(DATA_FOLDER), lang=DEFAULT_LANG)

    # Load sparse retriever data
    with open(os.path.join(save_dir, 'sparse_bm25.pkl'), 'rb') as f:
        retriever.sparse_retriever.bm25 = pickle.load(f)

    # Load dense embeddings
    with open(os.path.join(save_dir, 'dense_embeddings.pkl'), 'rb') as f:
        retriever.dense_retriever.embeddings = pickle.load(f)

    print(f"Retriever loaded from {save_dir}")
    return retriever
