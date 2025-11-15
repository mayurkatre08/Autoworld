import os
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import re

load_dotenv()

class RAGState(TypedDict):
    query: str
    chunks: List[Dict]
    retrieved_chunks: List[Dict]
    response: str
    
class LangGraphCarRAG:
    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = [
            """Maruti Suzuki Fronx 2024 Complete Review: The Maruti Suzuki Fronx is a compact SUV that was launched in 2024 with great fanfare. It features a 1.2L Petrol engine that delivers excellent performance and fuel efficiency of 22.89 kmpl. The vehicle is priced competitively between Rs 7.51-12.91 Lakh, making it accessible to a wide range of customers. Key features include a premium sunroof that enhances the cabin experience, a 360-degree camera system for parking assistance, wireless charging capability for modern convenience, and a large 9-inch touchscreen infotainment system. The Fronx has achieved a prestigious 5-star Global NCAP safety rating, establishing it as one of the safest vehicles in its segment. The exterior design is modern and appealing, while the interior offers comfortable seating for five passengers.""",
            """Tata Nexon EV 2024 Electric Revolution: The Tata Nexon EV represents India's electric vehicle revolution and is currently the best-selling electric SUV in the country. This electric SUV is powered by an advanced electric motor that provides instant torque and smooth acceleration. With an impressive range of 465 km on a single charge, it addresses range anxiety effectively. The vehicle is priced between Rs 14.74-19.94 Lakh, positioning it as a premium electric option. Notable features include fast charging capability that can charge the battery from 10% to 80% in just 56 minutes, cutting-edge connected car technology, premium interior materials and design, and an advanced air purifier system. The Nexon EV has earned a 5-star Global NCAP safety rating and offers zero-emission driving with state-of-the-art battery technology and regenerative braking.""",
            """Hyundai Creta 2024 Feature Analysis: The Hyundai Creta continues to be one of India's most popular mid-size SUVs with its 2024 iteration bringing enhanced features and performance. It offers both 1.5L Petrol and Diesel engine options to cater to different customer preferences. The vehicle delivers a respectable fuel efficiency of 17.4 kmpl and is priced between Rs 11.00-20.15 Lakh across various variants. The Creta stands out with its panoramic sunroof that creates an airy cabin atmosphere, Advanced Driver Assistance Systems (ADAS) for enhanced safety, wireless charging pad for smartphones, and a premium sound system for entertainment. The vehicle has achieved a 4-star Global NCAP safety rating and offers a perfect blend of style, comfort, and technology for urban and highway driving.""",
            """Electric Vehicle Market in India 2024: The electric vehicle market in India is experiencing unprecedented growth with government incentives and improving charging infrastructure. Among electric cars under 20 lakhs, the Tata Nexon EV emerges as the clear leader with its 465km range and comprehensive fast charging network. Alternative options include the MG ZS EV which offers premium features and the Mahindra XUV400 which provides rugged electric performance. The charging infrastructure has expanded significantly with both AC and DC fast charging stations becoming more common in major cities. Government subsidies under the FAME II scheme make electric vehicles more affordable, while lower running costs compared to petrol vehicles provide long-term savings. The future of electric mobility in India looks promising with more manufacturers launching electric variants."""
        ]
        self.chunks = []
        self._process_documents()
        self.graph = self._create_graph()
    
    def _chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    def _process_documents(self):
        for doc_id, doc in enumerate(self.documents):
            chunks = self._chunk_text(doc)
            for chunk_id, chunk in enumerate(chunks):
                self.chunks.append({
                    'text': chunk,
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'car_name': doc.split(':')[0].strip()
                })
        
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        embeddings = self.encoder.encode(chunk_texts)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
    
    def retrieve_node(self, state: RAGState) -> RAGState:
        query_embedding = self.encoder.encode([state["query"]])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding.astype('float32'), 3)
        
        retrieved_chunks = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(scores[0][i])
                retrieved_chunks.append(chunk)
        
        return {"retrieved_chunks": retrieved_chunks}
    
    def generate_node(self, state: RAGState) -> RAGState:
        context_parts = []
        for i, chunk in enumerate(state["retrieved_chunks"]):
            context_parts.append(f"Source {i+1} ({chunk['car_name']}): {chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        system_prompt = """You are a car expert assistant. Provide accurate, helpful answers about cars based on the given context. 
        - Be specific with prices, features, and specifications
        - If comparing cars, highlight key differences
        - Always mention the car name when providing information
        - Keep responses concise but informative"""
        
        user_prompt = f"""Context Information:
{context}

User Question: {state['query']}

Provide a helpful answer based on the context above:"""
        
        response = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=400
        )
        
        return {"response": response.choices[0].message.content}
    
    def _create_graph(self):
        workflow = StateGraph(RAGState)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_node)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def query(self, question: str) -> str:
        result = self.graph.invoke({"query": question})
        return result["response"]

def main():
    rag = LangGraphCarRAG()
    
    print("🚗 Enhanced LangGraph Car RAG System")
    print(f"📚 Processed {len(rag.chunks)} chunks from {len(rag.documents)} documents")
    print("Ask about cars! Type 'quit' to exit.\n")
    
    while True:
        question = input("You: ").strip()
        if question.lower() in ['quit', 'exit']:
            break
        
        if question:
            try:
                result = rag.graph.invoke({"query": question})
                print(f"Assistant: {result['response']}")
                print(f"[Retrieved {len(result['retrieved_chunks'])} chunks]\n")
            except Exception as e:
                print(f"Error: {e}\n")

if __name__ == "__main__":
    main()