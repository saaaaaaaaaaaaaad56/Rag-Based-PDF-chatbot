from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()


Docs = PyPDFLoader('C:\\Users\\muham\\OneDrive\\Desktop\\Rag-Based_project\\Rag-Based-PDF-chatbot\\transformers_bert_gpt_guide.pdf')

docs  = Docs.load()



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=250)


documents = text_splitter.split_documents(docs)



embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vector_store = FAISS.from_documents(documents, embeddings)


retriever = vector_store.as_retriever(search_kwargs={"k":3},search_type="similarity")


retrieved_docs = retriever.invoke("What is the difference between BERT and GPT?")



llm = HuggingFaceEndpoint(
    repo_id = 'Qwen/Qwen2.5-1.5B-Instruct',
    task = 'text-generation',
    temperature = 0.4,

)

model = ChatHuggingFace(
    llm = llm
)



prompt = PromptTemplate(
       
    template = "you are a really helpful assistant and will answer the question based on the context provided. If you don't know the answer, just say that you don't know, don't try to make up an answer. Context: {context} Question: {question} Answer:",

    input_variables = ['context','question']
)


question = "What is the difference between BERT and GPT?"


final_prompt = prompt.invoke({"context": retrieved_docs, "question": question})


parser = StrOutputParser()


def formate_doc(retrieved_doc):
    context_text = " ".join(Doc.page_content for Doc in retrieved_doc)
    return context_text


parallel_chain = RunnableParallel({
    'context' : retriever | RunnableLambda(formate_doc),
    'question' : RunnablePassthrough()
}
)



main_chain = parallel_chain | prompt | model | parser

result = main_chain.invoke(question)
print(result)
