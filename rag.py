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

# Chargement des données
loader = TextLoader("data/all_texts_clean.txt")
data = loader.load()

# Division du texte
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Choix des embeddings adaptés au français

#name = 'dangvantuan/sentence-camembert-large'
#db_name = 'db_dangvantuan'
#name='gilf/french-camembert-large-sentence-embedding'
#db_name = 'db_gilf'
#name = 'hugorosen/flaubert_base_uncased-xnli-sts'
#db_name = 'db_LaBSE'
#local_embeddings = embeddings(name)

local_embeddings = OllamaEmbeddings(model="zylonai/multilingual-e5-large")

db_name = "db_e5"
# Création du vectorstore avec les nouveaux embeddings
#vectorstore = Chroma.from_documents(
#            documents=all_splits,
#            embedding=local_embeddings,
#            persist_directory=db_name
#        )

vectorstore = Chroma(persist_directory=db_name, embedding_function=local_embeddings)

# Configuration du retriever pour renvoyer plus de documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Initialisation du modèle de langage
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

prompt = ChatPromptTemplate.from_template(template)

# Création de la chaîne de question-réponse
question_answer_chain = create_stuff_documents_chain(model, prompt)

# Création de la chaîne RAG complète
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Exécution de la chaîne RAG
result = rag_chain.invoke({"input": "Pourquoi faut-il arroser les salades le soir?"})

# Affichage des résultats
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
