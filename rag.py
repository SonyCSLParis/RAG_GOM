from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import time
import shutil

def format_docs(docs):
   return "\n\n".join(doc.page_content for doc in docs)

def embeddings(name):
   return HuggingFaceEmbeddings(
            model_name='name',
            model_kwargs={'device': 'cuda:0'}
        )
        

t0 = time.time()

loader = TextLoader("data/all_texts_clean.txt")
data = loader.load()

# Division du texte
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

#name = 'dangvantuan/sentence-camembert-large'
#db_name = 'db_dangvantuan'
#name='gilf/french-camembert-large-sentence-embedding'
#db_name = 'db_gilf'
#name = 'hugorosen/flaubert_base_uncased-xnli-sts'
#db_name = 'db_LaBSE'
#local_embeddings = embeddings(name)

local_embeddings = OllamaEmbeddings(model="zylonai/multilingual-e5-large")

db_name = "db_e5"

vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=local_embeddings,
            persist_directory=db_name
        )

#vectorstore = Chroma(persist_directory=db_name, embedding_function=local_embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

model = ChatOllama(
            model="mixtral:8x7b",
        )

# Optimisation du prompt
template = """
           En te basant uniquement sur le contexte suivant, réponds à la question de manière claire et concise :

          {context}

          Question : {input}

          Réponse :
          """

template2 ="""
           Un paysan parle comme ça: "Eh bien, ma foi, faut qu'j'aille aux champs de bon matin. La terre, elle s'laboure pas toute seule, vous savez. Si le temps s'met au beau, on aura une belle récolte c't'année. Et pis, faut qu'j'aille m'occuper des bêtes avant qu'le soleil soit trop haut. Allez, au boulot, y'a pas d'temps à perdre !" Reponds à la question en parlant comme un paysan et en te basant uniquement sur le contexte suivant:
           {contexte}
           Question: {input]
           Réponse:
           """ 


prompt = ChatPromptTemplate.from_template(template)
question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
result = rag_chain.invoke({"input": "Pourquoi faut-il arroser les salades le soir?"})

print("--------------------")
print(result["input"])
print("--------------------")
print(result["answer"])
print("--------------------")
print("Basé sur les sources suivantes :")
print("***")

for doc in result["context"]:
    print(doc.page_content)
    print("***")

print(f"Temps d'exécution : {time.time() - t0} secondes")
