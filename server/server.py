from flask import Flask, request, jsonify
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import DirectoryLoader
import time
import shutil

def format_docs(docs):
   return "\n\n".join(doc.page_content for doc in docs)

def test(docs):
    if not docs:
        return []
    contexts = []
    for doc in docs:
        context = {
            "page_content": doc.page_content,
            "source": doc.metadata,
        }
        contexts.append(context)
    return contexts

def embeddings(name):
   return HuggingFaceEmbeddings(
            model_name='name',
            model_kwargs={'device': 'cuda:0'}
        )

t0 = time.time()

# loader = TextLoader("../data/all_texts_clean.txt")
loader = DirectoryLoader('../corpus_txt_agg')

data = loader.load()

# Division du texte
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

local_embeddings = OllamaEmbeddings(model="zylonai/multilingual-e5-large")

# db_name = "/mnt/diskSustainability/GOM/RAG/db_e5-new"
db_name = "../db_e5"

def load_or_create(string):
   if string == "create":
      vectorstore = Chroma.from_documents(
         documents=all_splits,
         embedding=local_embeddings,
         persist_directory=db_name
      )
   else:
      vectorstore = Chroma(persist_directory=db_name, embedding_function=local_embeddings)
   return vectorstore

vectorstore = load_or_create("import")

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# model = ChatOllama(
#     model = ChatOllama(model="mixtral:8x7b")
#     # model = ChatOllama(model="llama3.1:latest")
#     # model = ChatOllama(model="mistral:latest")
# )

model = ChatOllama(
    # model="gemma2:latest"
    # model="tinyllama:latest"
    # model="mixtral:8x22b"
    # model="mistral:latest"
    model="openchat:latest" #vrm rapide
    # model="llama3.3:latest"
    ) 
# Optimisation du prompt
template = """
           En te basant uniquement sur le contexte suivant, réponds à la question de manière claire et concise:

          {context}

          Question : {input}

          Réponse :
          """

template2 = """
           Un paysan parle comme ça: "Eh bien, ma foi, faut qu'j'aille aux champs de bon matin. La terre, elle s'laboure pas toute seule, vous savez. Si le temps s'met au beau, on aura une belle récolte c't'année. Et pis, faut qu'j'aille m'occuper des bêtes avant qu'le soleil soit trop haut. Allez, au boulot, y'a pas d'temps à perdre !" Reponds à la question en parlant comme un paysan et en te basant uniquement sur le contexte suivant:
           {contexte}
           Question: {input]
           Réponse:
           """ 

prompt = ChatPromptTemplate.from_template(template)
question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


app = Flask(__name__)

@app.route("/get-response", methods=['POST'])
def getResponse():
    data = request.get_json()
    question = data.get("question")
    if question:
        result = rag_chain.invoke({"input": question})
        response = {
            "input": question,
            "contexte": test(result["context"]),
            "answer": result["answer"],
        }
        print("\n\n")
        print(question)
        print("\n\n")
        contexts = test(result["context"])
        for context in contexts:
            print("les contextes et les sources", context)
            print("\n\n")
        return jsonify(response), 200
    else:
        return jsonify({"error": "bad request"}), 400
     
if __name__ == "__main__":
    app.run(debug=True)
