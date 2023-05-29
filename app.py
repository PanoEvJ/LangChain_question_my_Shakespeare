import os
import shutil
from glob import glob
from transformers import AutoTokenizer
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import BSHTMLLoader, DirectoryLoader

bshtml_dir_loader = DirectoryLoader('./data/', loader_cls=BSHTMLLoader)
data = bshtml_dir_loader.load()

bloomz_tokenizer = AutoTokenizer.from_pretrained('bigscience/bloomz-1b7')
text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(bloomz_tokenizer, chunk_size=100, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(data)

embeddings = HuggingFaceEmbeddings()

persist_directory = "vector_db"
vectordb = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_directory)
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
vectordb.persist()
vectordb = None
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

llm = HuggingFacePipeline.from_model_id(
    model_id="bigscience/bloomz-1b7", 
    task="text-generation", 
    model_kwargs={"temperature" : 0, "max_length" : 500})

doc_retriever = vectordb.as_retriever()
shakespeare_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=doc_retriever)

def make_inference(query):
    # docs = docsearch.get_relevant_documents(query)
    # return(chain.run(input_documents=docs, question=query))
    return(shakespeare_qa.run(query))

if __name__ == "__main__":
    # make a gradio interface
    import gradio as gr

    gr.Interface(
        make_inference,
        [
            gr.inputs.Textbox(lines=2, label="Query"),
        ],
        gr.outputs.Textbox(label="Response"),
        title="🗣️TalkToMyShakespeare📄",
        description="🗣️TalkToMyShakespeare📄 is a tool that allows you to ask questions about Shakespeare literature work.",
    ).launch()