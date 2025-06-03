from typing import List
import torch
import streamlit as st
from openai import OpenAI
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TextStreamer, pipeline

from agent.constants import SYSTEM_PROMPT, DEFAULT_LANG, DEFAULT_TOP_K, CONTEXT_PROMPT, OPEN_AI_API_KEY

client = OpenAI(
    api_key=OPEN_AI_API_KEY,
)
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

class Chatbot:
    def __init__(self, retriever, model_name=MODEL_NAME, device_map="auto", load_in_8bit=False, load_in_4bit=True, ):
        self.retriever = retriever
        self.conversation_history = ""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"Loading Model")

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="fp8",  # Menos agresivo que nf4
            bnb_8bit_use_double_quant=False  # Sin doble cuantización para más estabilidad
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device_map,  # Importante:  usar device_map
            cache_dir="models/" + model_name,
            quantization_config=bnb_config,  # Aquí aplicamos la cuantización
        )

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=400,  # Ajusta según necesidad
            do_sample=True,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
            streamer=TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True),
            # Para ver la respuesta en tiempo real
            use_cache=True,
        )
        self.relevant_docs = None

    def generate_prompt(self, query: str, context_documents: List[str]) -> str:
        """Genera el prompt para DeepSeek R1."""
        context = ""

        for i, doc in enumerate(context_documents):
            context += f"<|document{i + 1}|>\n{doc}\n"

        rag_context = f"<|context|>\n{CONTEXT_PROMPT}\n\n{context}\n"

        prompt = SYSTEM_PROMPT + rag_context
        if self.conversation_history != "":
            prompt += '<|history|>\n' + self.conversation_history
        prompt += f"<|user|>\n{query}\n"
        return prompt

    def chat_remote(self, query: str, lang: str = DEFAULT_LANG, top_k: int = DEFAULT_TOP_K, device_setup="auto"):
        if self.relevant_docs is None:
            self.relevant_docs = self.retriever.search_documents(query, top_k=top_k, lang=lang)

        prompt = self.generate_prompt(query, self.relevant_docs)

        with st.spinner('Generando respuesta...'):
            completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="gpt-4.1-nano",
                temperature=0.5,
                top_p=0.95,
                stream=True
            )

            # Stream the response
            #response_container = st.empty()
            collected_messages = []

            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    collected_messages.append(chunk.choices[0].delta.content)
                    #response_container.markdown("".join(collected_messages))

        full_response = "".join(collected_messages)
        print(f"Response: {full_response}")

        # Update conversation history
        self.conversation_history += f"<|user|>\n{query}\n<|assistant|>{full_response}\n"

        return full_response

    def chat_local(self, query: str, lang: str = DEFAULT_LANG, top_k: int = DEFAULT_TOP_K, device_setup="auto"):
        """Función principal de chat (adaptada para streaming)."""
        relevant_docs = self.retriever.search_documents(query, top_k=top_k, lang=lang)
        prompt = self.generate_prompt(query, relevant_docs)

        # Usamos .generate() directamente para tener más control (streaming y length_penalty)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device_setup)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        #  Generamos en un hilo separado para no bloquear la interfaz
        with st.spinner('Generando respuesta...'):  # Muestra un spinner mientras genera
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,  # ¡Ajusta! Valor inicial bajo
                repetition_penalty=1.2,
                # length_penalty=1.5,   #  Opcional: descomenta para usar length_penalty
                do_sample=True,
                top_k=50,
                top_p=0.95,
                streamer=streamer,  # Usa el streamer
            )

        #  Extraemos la respuesta completa (ya que el streamer la imprimió)
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        #  Aislar la respuesta del chatbot (quitando el prompt)
        response = full_response.replace(prompt, "").strip()

        # ---  ACTUALIZA EL HISTORIAL  ---
        self.conversation_history += f"<|user|>\n{query}\n<|assistant|>{response}\n"

        return response

    def clear_history(self):
        """Limpia el historial de la conversación."""
        self.conversation_history = ""
        self.relevant_docs = None

