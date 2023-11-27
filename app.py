import os
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from pydantic import BaseModel
from langchain.callbacks import AsyncIteratorCallbackHandler
import uvicorn
from fastapi import FastAPI

# Create the app object
app = FastAPI()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DB_FAISS_PATH = 'vectorstore/db_faiss'

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    config = {'max_new_tokens': 1000, 'context_length':1000, 'repetition_penalty': 1.1}
    llm = CTransformers(
        model = "/home/aniket/Chaabi/model.bin",
        model_type="llama",
        config=config,
        temperature = 0.5
    )
    return llm


loader = CSVLoader(file_path='/home/aniket/Chaabi/.venv/bigBasketProducts.csv', encoding="utf-8", csv_args={
            'delimiter': ','})
data = loader.load()

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                    model_kwargs={'device': 'cuda'})

db = FAISS.from_documents(data, embeddings)
db.save_local(DB_FAISS_PATH)
llm = load_llm()
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

def conversational_chat(query):
    result = chain({"question": query, "chat_history": []})
    return result["answer"]

class Item(BaseModel):
    text: str

@app.post("/ask_query")
def root(Data: Item):
    return {"Answer": conversational_chat(Data.text)}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
    
