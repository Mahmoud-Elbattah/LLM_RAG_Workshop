from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

class ScienceRAGAssistant:
    def __init__(self, faiss_path="faiss_Sci", model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.embedding_model = HuggingFaceEmbeddings()
        
        self.db = FAISS.load_local(
            faiss_path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
        
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name
        )

    def retrieve_docs(self, query, k=3):
        return self.db.similarity_search(query, k=k)

    def format_docs(self, docs):
        return "\n\n".join([doc.page_content for doc in docs])

    def build_prompt(self, context, question, allow_model_knowledge=False):
        if allow_model_knowledge==False:
            instruction = """
First, check if the context contains the answer.
If yes, answer using the context.
If not, return ONLY: I don't know.
Do not attempt to answer before checking.
"""
        else:
            instruction = """
Use the context below as your main source of information.
You may use your own knowledge to explain and clarify the answer.

Do not invent references.
"""

        prompt = f"""
You are a scientific assistant.

{instruction}

Context:
{context}

Question:
{question}

Answer:
"""
        return prompt

    def generate_answer(self, prompt, max_new_tokens=200):
        response = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        
        full_text = response[0]["generated_text"]
        
        if "Answer:" in full_text:
            answer = full_text.split("Answer:")[-1].strip()
        else:
            answer = full_text[len(prompt):].strip()
        
        return answer

    def ask(self, question, k=3, allow_model_knowledge=False):
        docs = self.retrieve_docs(question, k=k)
        context = self.format_docs(docs)
        prompt = self.build_prompt(
            context,
            question,
            allow_model_knowledge=allow_model_knowledge
        )
        answer = self.generate_answer(prompt)
        return answer