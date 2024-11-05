from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings
from typing import Optional, List
import os
from dotenv import load_dotenv
from pathlib import Path

import cohere                                                                                                                                                                                                                                                           

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

class SimpleRAGPipeline:
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the RAG pipeline.
        
        Args:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
            openai_api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY in environment.
        """
        load_dotenv()
        #self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        # if not self.openai_api_key:
        #     raise ValueError("OpenAI API key must be provided or set in environment as OPENAI_API_KEY")
                                                                                                                                                
        # Initialize components
        self.embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
        self.vector_store = None
        self.qa_chain = None
        
        # Default chunk settings
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        
        # Default prompt template
        self.default_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        {context}
        
        Question: {question}
        Answer: """
        
        self.prompt = PromptTemplate(
            template=self.default_template,
            input_variables=["context", "question"]
        )

    def upload_pdf(self, 
                   pdf_path: str, 
                   chunk_size: Optional[int] = None,
                   chunk_overlap: Optional[int] = None) -> None:
        """
        Upload and process a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Optional custom chunk size for text splitting
            chunk_overlap: Optional custom chunk overlap for text splitting
        """
        # Convert to Path object and validate
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
            
        # Update text splitter if custom parameters provided
        if chunk_size or chunk_overlap:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size or self.chunk_size,
                chunk_overlap=chunk_overlap or self.chunk_overlap,
                length_function=len,
            )
            
        print(f"Loading PDF from {pdf_path}...")
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        
        print("Splitting documents into chunks...")
        splits = self.text_splitter.split_documents(documents)
        
        print("Creating vector store...")
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        # Initialize QA chain
        self._initialize_qa_chain()
        print("PDF processed and ready for querying!")
        
    def set_custom_prompt(self, template: str) -> None:
        """
        Set a custom prompt template.
        
        Args:
            template: Custom template string. Must include {context} and {question} variables.
        """
        if "{context}" not in template or "{question}" not in template:
            raise ValueError("Template must include {context} and {question} variables")
            
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Reinitialize QA chain with new prompt if vector store exists
        if self.vector_store:
            self._initialize_qa_chain()

    def _initialize_qa_chain(self) -> None:
        """Initialize the QA chain with current prompt and vector store."""
        self.qa_chain = RetrievalQA.from_chain_type(
            llm= ChatOpenAI(),
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={"prompt": self.prompt}
        )

    def query(self, question: str) -> str:
        """
        Query the RAG system.
        
        Args:
            question: Question to ask about the uploaded document(s)
        Returns:
            Answer from the system
        """
        if not self.qa_chain:
            return "Please upload a PDF document first using upload_pdf()"
            
        return self.qa_chain.run(question)

