from typing import List, Dict, Any, Optional, Union
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document, AgentAction, AgentFinish
from langchain.prompts import PromptTemplate
from langchain.agents import tool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad.log import format_log_to_str
from langchain.tools.render import render_text_description
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class RAGAgent:
    def __init__(self):
        """
        Initialize RAG agent with OpenAI API key.
        """
        load_dotenv()
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.llm = ChatOpenAI(
            temperature=0,
            stop=["\nObservation"],
            #openai_api_key=openai_api_key
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        
        @tool
        def search_documents(query: str) -> str:
            """Search through the documents using the query"""
            if self.vector_store is None:
                return "No documents have been indexed yet."
            results = self.vector_store.similarity_search(query, k=2)
            return "\n".join(doc.page_content for doc in results)

        self.tools = [search_documents]
        
        self.prompt = PromptTemplate.from_template("""
            Answer the following questions as best you can using the provided documents. You have access to the following tools:
            {tools}

            Use the following format:
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!
            Question: {input}
            Thought: {agent_scratchpad}
                    """
        ).partial(
            tools=render_text_description(self.tools),
            tool_names=", ".join([t.name for t in self.tools]),
        )

    def ingest_pdf(self, 
                   pdf_path: Union[str, Path], 
                   collection_name: str = "documents",
                   chunk_size: int = 1000,
                   chunk_overlap: int = 200) -> None:
        """
        Ingest a PDF file into the vector store.
        
        Args:
            pdf_path: Path to the PDF file
            collection_name: Name for the Chroma collection
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        # Convert string path to Path object if necessary
        pdf_path = Path(pdf_path) if isinstance(pdf_path, str) else pdf_path
        
        # Validate PDF exists
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
            
        # Load PDF
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        
        # Update text splitter settings if different from defaults
        if chunk_size != 1000 or chunk_overlap != 200:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
        
        # Split documents into chunks
        splits = self.text_splitter.split_documents(documents)
        
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory="./chroma_db"
            )
        else:
            # Add to existing vector store
            self.vector_store.add_documents(splits)
            
    def ingest_documents(self, 
                        documents: Union[List[Document], List[str], str, Path], 
                        collection_name: str = "documents") -> None:
        """
        Ingest documents into the vector store. Accepts various input types:
        - List of Document objects
        - List of strings
        - Single string
        - Path to PDF file
        
        Args:
            documents: Documents to ingest
            collection_name: Name for the Chroma collection
        """
        # Handle PDF file
        if isinstance(documents, (str, Path)) and str(documents).lower().endswith('.pdf'):
            self.ingest_pdf(documents, collection_name)
            return
            
        # Handle list of strings
        if isinstance(documents, list) and all(isinstance(doc, str) for doc in documents):
            documents = [Document(page_content=text) for text in documents]
            
        # Handle single string
        elif isinstance(documents, str):
            documents = [Document(page_content=documents)]
            
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory="./chroma_db"
            )
        else:
            self.vector_store.add_documents(splits)

    def query(self, question: str) -> str:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
        Returns:
            The final answer from the agent
        """
        if self.vector_store is None:
            return "No documents have been indexed yet. Please add documents first."
            
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
            }
            | self.prompt
            | self.llm
            | ReActSingleInputOutputParser()
        )

        intermediate_steps = []
        agent_step = ""
        
        while not isinstance(agent_step, AgentFinish):
            agent_step = agent.invoke(
                {
                    "input": question,
                    "agent_scratchpad": intermediate_steps,
                }
            )
            
            if isinstance(agent_step, AgentAction):
                tool_name = agent_step.tool
                tool_to_use = next(t for t in self.tools if t.name == tool_name)
                observation = tool_to_use.func(str(agent_step.tool_input))
                intermediate_steps.append((agent_step, str(observation)))
                
        return agent_step.return_values["output"]

# Example usage
if __name__ == "__main__":
    # Initialize RAG agent
    rag_agent = RAGAgent()
    
    # Example 1: Ingest a PDF file
    rag_agent.ingest_documents("1811.12808.pdf")
    
    # Example 2: Ingest text directly
    texts = [
        "The cat sat on the mat.",
        "The dog chased the ball."
    ]
    rag_agent.ingest_documents(texts)
    
    # Query the system
    question = "What did the dog do?"
    answer = rag_agent.query(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")