import gc
import threading
from collections import defaultdict
from threading import Thread
from typing import List, Dict
import torch
from huggingface_hub import login
from openai import OpenAI
from torch._inductor import cudagraph_trees
from transformers import AutoTokenizer, Gemma3ForCausalLM, TextIteratorStreamer

from agent.constants import SYSTEM_PROMPT, DEFAULT_LANG, DEFAULT_TOP_K, CONTEXT_PROMPT, OPENAI_API_KEY, \
    USE_ONLINE_AGENTS, LLM_MAX_TOKENS, HUGGINGFACE_API_KEY, DEFAULT_TEMPERATURE, DEFAULT_TOP_P

client = OpenAI(
    api_key=OPENAI_API_KEY,
)
MODEL_NAME = "google/gemma-3-1b-it"


class Chatbot:
    def __init__(self, retriever, model_name=MODEL_NAME, device_map="auto"):
        login(HUGGINGFACE_API_KEY)
        self.retriever = retriever
        self.conversation_history = []
        self.device_map = device_map

        if not USE_ONLINE_AGENTS:
            print(f"Loading local models")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = Gemma3ForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,  # Importante:  usar device_map
                cache_dir="models/" + model_name,
            )
        else:
            print("Using OpenAI API")

        self.relevant_docs = None

    def generate_prompt(self, query: str, context_documents: List[str]) -> List[List[Dict]]:
        context = ""
        for i, doc in enumerate(context_documents):
            context += f"## Documento {i + 1}:\n{doc}\n"
        system_prompt = f"{CONTEXT_PROMPT}\n\n{context}\n{SYSTEM_PROMPT}"
        prompt = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}, ]
                },
                *self.conversation_history,
                {
                    "role": "user",
                    "content": [{"type": "text", "text": query}, ]
                },
            ],
        ]
        return prompt

    def chat_remote(self, prompt, response_container):
        completion = client.chat.completions.create(
            messages=prompt[0],
            model="gpt-4.1-nano",
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            stream=True
        )
        # Stream the response
        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
                if response_container:
                    response_container.markdown(response)

        return response

    def chat_local(self, prompt, response_container):
        """Función principal de chat (adaptada para streaming)."""

        # Usamos .generate() directamente para tener más control (streaming y length_penalty)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        #  Generamos en un hilo separado para no bloquear la interfaz
        def inference():
            cudagraph_trees.local.tree_manager_containers = {}
            cudagraph_trees.local.tree_manager_locks = defaultdict(threading.Lock)
            torch._C._stash_obj_in_tls("tree_manager_containers", cudagraph_trees.local.tree_manager_containers)
            torch._C._stash_obj_in_tls("tree_manager_locks", cudagraph_trees.local.tree_manager_locks)
            inputs = self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            with torch.inference_mode():
                self.model.generate(
                    **inputs,
                    do_sample=True,
                    length_penalty=1.2,
                    repetition_penalty=1.2,
                    temperature=DEFAULT_TEMPERATURE,
                    max_new_tokens=LLM_MAX_TOKENS,
                    top_p=DEFAULT_TOP_P,
                    streamer=streamer
                )
            del inputs
            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()
        thread = Thread(target=inference)
        thread.start()
        response = ""
        for new_text in streamer:
            response += new_text
            if response_container:
                response_container.markdown(response)
        return response

    def chat(self, query: str, response_container, lang: str = DEFAULT_LANG, top_k: int = DEFAULT_TOP_K):
        if self.relevant_docs is None:
            self.relevant_docs = self.retriever.search_documents(query, top_k=top_k, lang=lang)
        prompt = self.generate_prompt(query, self.relevant_docs)
        if not USE_ONLINE_AGENTS:
            response = self.chat_local(prompt, response_container)
        else:
            response = self.chat_remote(prompt, response_container)
        # Update conversation history
        self.conversation_history.extend([
            {
                "role": "user",
                "content": [{"type": "text", "text": query}, ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}, ]
            },
        ])
        return response

    def clear_history(self):
        """Limpia el historial de la conversación."""
        self.conversation_history = []
        self.relevant_docs = None

