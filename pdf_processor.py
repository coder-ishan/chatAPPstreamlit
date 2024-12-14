import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import chromadb
from chromadb.config import Settings
import os
import logging
from typing import Dict, List, Any, Tuple

from dotenv import load_dotenv
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        """
        Initialize the PDFProcessor with necessary components for text processing,
        embedding generation, and database management.
        """
        # Initialize text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        # Initialize OpenAI embeddings
        try:
            self.embeddings = OpenAIEmbeddings()
            logger.info("Successfully initialized OpenAI embeddings")
        except Exception as e:
            logger.error(f"Error initializing OpenAI embeddings: {e}")
            raise

        # Set up ChromaDB
        self.persist_directory = "db"
        self._setup_chroma()

        # Define QA template
        self.qa_template = """
        You are a helpful assistant that provides accurate information based on the PDF content.
        Your response must always be in JSON format with exactly two keys: 'answer' and 'confidence'.

        Rules:
        1. Answer must be based only on the provided context
        2. Confidence score must be between 0 and 1:
        - 1.0: Answer is directly stated in context
        - 0.7-0.9: Answer can be clearly inferred from context
        - <= 0.6: Question is not relevant with the given context and can't be answered. I am using a fallback with threshold 0.6.  

        Context: {context}

        Question: {question}

        Provide your response in this exact format:
        {{"answer": "your answer here", "confidence": 0.X}}

        Remember: Always return a JSON object, no matter what. """

    def _setup_chroma(self) -> None:
        """
        Set up ChromaDB with persistent storage.
        """
        try:
            # Create persistence directory if it doesn't exist
            if not os.path.exists(self.persist_directory):
                os.makedirs(self.persist_directory)

            # Initialize ChromaDB client
            self.chroma_client = chromadb.Client(
                Settings(
                    persist_directory=self.persist_directory,
                    chroma_db_impl="duckdb+parquet",
                    anonymized_telemetry=False
                )
            )
            logger.info("Successfully set up ChromaDB")
        except Exception as e:
            logger.error(f"Error setting up ChromaDB: {e}")
            raise

    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_file: The uploaded PDF file object
            
        Returns:
            str: Extracted text from the PDF
        """
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            # Extract text from each page
            for page in pdf_reader.pages:
                text += page.extract_text()
                
            if not text.strip():
                raise ValueError("No text extracted from PDF")
                
            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    def create_knowledge_base(self, text: str, pdf_name: str) -> Chroma:
        """
        Create a vector store from the PDF text.
        
        Args:
            text (str): Extracted text from PDF
            pdf_name (str): Name of the PDF file
            
        Returns:
            Chroma: Vector store containing the processed text
        """
        try:
            # Split text into chunks
            texts = self.text_splitter.split_text(text)
            logger.info(f"Split text into {len(texts)} chunks")
            
            # Create sanitized collection name
            collection_name = f"pdf_{pdf_name.replace('.pdf', '').replace(' ', '_')}"
            
            # Create vector store with persistence
            vectorstore = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=collection_name
            )
            
            # Persist the data
            vectorstore.persist()
            logger.info(f"Successfully created and persisted vector store: {collection_name}")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error creating knowledge base: {e}")
            raise

    def load_knowledge_base(self, pdf_name: str) -> Chroma:
        """
        Load an existing knowledge base for a PDF if it exists.
        
        Args:
            pdf_name (str): Name of the PDF file
            
        Returns:
            Chroma: Existing vector store or None if not found
        """
        try:
            collection_name = f"pdf_{pdf_name.replace('.pdf', '').replace(' ', '_')}"
            
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
            
            # Verify the collection exists and has data
            if vectorstore._collection.count() > 0:
                logger.info(f"Successfully loaded existing knowledge base: {collection_name}")
                return vectorstore
            return None
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return None

    def create_conversation_chain(self, vectorstore: Chroma) -> ConversationalRetrievalChain:
        """Create a conversation chain using the vector store."""
        try:
            llm = ChatOpenAI(
                temperature=0.1,
                model_name='gpt-3.5-turbo',
                max_tokens=500
            )
            
            # Create prompt template
            prompt = PromptTemplate(
                template=self.qa_template,
                input_variables=["context", "question"]
            )
            
            # Create the chain with specific configuration
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                ),
                return_source_documents=False,
                combine_docs_chain_kwargs={"prompt": prompt},
                verbose=True
            )
            
            logger.info("Successfully created conversation chain")
            return chain
            
        except Exception as e:
            logger.error(f"Error creating conversation chain: {e}")
            raise

    def get_response(self, chain: ConversationalRetrievalChain, question: str, chat_history: List) -> str:
        """Get response from the chain."""
        try:
            # Get response from chain
            response = chain({
                "question": question, 
                "chat_history": chat_history
            })
            
            # Extract the raw answer
            raw_answer = response.get('answer', '') if isinstance(response, dict) else str(response)
            
            try:
                # Try to parse as JSON
                import json
                # Clean the response if needed (remove any leading/trailing text)
                raw_answer = raw_answer.strip()
                if raw_answer.startswith("```json"):
                    raw_answer = raw_answer[7:-3]  # Remove ```json and ``` if present
                
                parsed_response = json.loads(raw_answer)
                
                # Extract answer and confidence
                answer = parsed_response.get('answer', '')
                confidence = float(parsed_response.get('confidence', 0))
                
                # Apply fallback if confidence is low
                if confidence <= 0.62:
                    final_answer = (
                        "Sorry, I didn’t understand your question. Do you want to connect with a live agent?"
                    )
                else:
                    final_answer = answer
                
                logger.info(f"Generated response for question: {question[:50]}... (confidence: {confidence})")
                return final_answer
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {raw_answer}")
                # Attempt to provide a reasonable response even if JSON parsing fails
                if raw_answer:
                    return f"{raw_answer}\n\n(Note: Confidence score unavailable)"
                return "I encountered an error processing the response. Please try again."
                
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return "I encountered an error while processing your question. Please try again."