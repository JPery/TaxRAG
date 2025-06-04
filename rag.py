import os
import torch
torch.classes.__path__ = [] # add this line to manually set it to empty.
import nltk
import streamlit as st
from agent.chatbot import Chatbot
from agent.constants import APP_NAME, DEFAULT_LANG, DATA_FOLDER, DEFAULT_TOP_K, WELCOME_MESSAGE, MAX_TOP_K
from agent.retrievers import HybridRetriever

from agent.utils import load_and_preprocess_data, load_retriever, save_retriever

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
        top_k = st.slider("Número de documentos a recuperar (top_k)", min_value=1, max_value=MAX_TOP_K, value=DEFAULT_TOP_K)
        if st.button("Limpiar historial"):
            chatbot.clear_history()
            st.success("Historial limpiado.")
        st.write("---")
        st.write("**Advertencia:** Este es un prototipo. La información proporcionada no debe considerarse una consulta fiscal real. Consulta siempre a un profesional.")

    # --- Interfaz principal (chat) ---

    # Muestra el historial de la conversación
    for message in chatbot.conversation_history:  # Separamos por los tokens de fin
        if message['role'] == "user":
            with st.chat_message('user'):
                st.write(message['content'][0]['text'].strip())
        elif message['role'] == "assistant":
            with st.chat_message('assistant'):
                st.write(message['content'][0]['text'].strip())

    # Input del usuario
    user_query = st.chat_input("Escribe tu consulta aquí...")

    if user_query:  # Si el usuario ha escrito algo
        with st.chat_message("user"):
            st.write(user_query)  # Mostramos su pregunta

        with st.chat_message("assistant"):
            with st.spinner('Generando respuesta...'):  # Muestra un spinner mientras genera
                response_container = st.empty()  # Contenedor para la respuesta
                answer = chatbot.chat(user_query, response_container, top_k=top_k)  # Obtenemos la respuesta
        print(answer)


if __name__ == '__main__':
    main()

