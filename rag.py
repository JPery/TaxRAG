import os
import nltk
import torch
import streamlit as st
from huggingface_hub import login

from agent.chatbot import Chatbot
from agent.constants import APP_NAME, DEFAULT_LANG, DATA_FOLDER, HF_TOKEN, DEFAULT_TOP_K, WELCOME_MESSAGE
from agent.retrievers import HybridRetriever

torch.classes.__path__ = []  # add this line to manually set it to empty.

from agent.utils import load_and_preprocess_data, load_retriever, save_retriever

login(HF_TOKEN)
st.set_page_config(page_title=APP_NAME, page_icon="💸")


@st.cache_resource()
def load_chatbot():
    device_setup = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using: ", device_setup)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Descarga de recursos necesarios
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    # Carga el dataset
    dataset = load_and_preprocess_data(DATA_FOLDER)

    # Carga el retriever (o lo crea y guarda si no existe)
    if os.path.exists('saved_retriever'):
        print(f"Loading embedings")
        retriever = load_retriever()
    else:
        print(f"Creating embedings")
        retriever = HybridRetriever()
        retriever.build_index(dataset, lang=DEFAULT_LANG)
        save_retriever(retriever)  # Lo guardamos para la próxima
        retriever = retriever

    # Inicializa el chatbot
    chatbot = Chatbot(retriever, device_map=device_setup)
    return chatbot


def main():
    st.title(APP_NAME)
    st.write(WELCOME_MESSAGE)

    # --- Carga de datos y modelo (una sola vez) ---
    chatbot = load_chatbot()

    # --- Barra lateral (opciones) ---
    with st.sidebar:
        st.header("Opciones")
        top_k = st.slider("Número de documentos a recuperar (top_k)", min_value=1, max_value=25, value=DEFAULT_TOP_K)
        if st.button("Limpiar historial"):
            chatbot.clear_history()
            st.success("Historial limpiado.")
        st.write("---")
        st.write("**Advertencia:** Este es un prototipo. La información proporcionada no debe considerarse una consulta fiscal real. Consulta siempre a un profesional.")

    # --- Interfaz principal (chat) ---

    # Muestra el historial de la conversación
    for message in chatbot.conversation_history.split("</s>\n"):  # Separamos por los tokens de fin
        if message.strip():  # Evita mensajes vacíos
            if "<|user|>" in message:
                with st.chat_message('user'):
                    st.write(message.replace("<|user|>", "").replace("Consulta del usuario:", "").strip())
            elif "<|assistant|>" in message:
                with st.chat_message('assistant'):
                    st.write(message.replace("<|assistant|>", "").strip())

    # Input del usuario
    user_query = st.chat_input("Escribe tu consulta aquí...")

    if user_query:  # Si el usuario ha escrito algo
        with st.chat_message("user"):
            st.write(user_query)  # Mostramos su pregunta
        answer = chatbot.chat_remote(user_query, top_k=top_k, device_setup="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))  # Obtenemos la respuesta

        with st.chat_message("assistant"):
            st.write(answer)  # Mostramos la respuesta


if __name__ == '__main__':
    main()

