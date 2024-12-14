import streamlit as st
from pdf_processor import PDFProcessor
import os
import logging
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
PAGE_CONFIG = {
    "page_title": "PDF Chatbot",
    "page_icon": "📚",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

def initialize_session_state() -> None:
    """Initialize all session state variables."""
    session_vars = {
        'processor': PDFProcessor(),
        'chain': None,
        'chat_history': [],
        'current_pdf': None,
        'last_question': None
    }
    
    for var, value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = value

def reset_conversation() -> None:
    """Reset all conversation-related session states."""
    st.session_state.chain = None
    st.session_state.chat_history = []
    st.session_state.current_pdf = None
    st.session_state.last_question = None
    logger.info("Conversation reset successful")

def handle_pdf_upload(pdf_file) -> Optional[bool]:
    """
    Handle PDF file upload and processing.
    
    Args:
        pdf_file: The uploaded PDF file
        
    Returns:
        Optional[bool]: True if successful, False if failed, None if no processing needed
    """
    try:
        # Check if it's a new PDF
        if st.session_state.current_pdf != pdf_file.name:
            with st.spinner('Processing PDF... This may take a moment.'):
                logger.info(f"Processing new PDF: {pdf_file.name}")
                
                # Try to load existing knowledge base
                vectorstore = st.session_state.processor.load_knowledge_base(pdf_file.name)
                
                if vectorstore is None:
                    # Create new knowledge base
                    text = st.session_state.processor.extract_text_from_pdf(pdf_file)
                    if not text.strip():
                        st.error("The PDF appears to be empty or unreadable.")
                        return False
                    
                    vectorstore = st.session_state.processor.create_knowledge_base(text, pdf_file.name)
                
                # Create conversation chain
                st.session_state.chain = st.session_state.processor.create_conversation_chain(vectorstore)
                st.session_state.current_pdf = pdf_file.name
                
                logger.info(f"Successfully processed PDF: {pdf_file.name}")
                return True
        return None
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        st.error(f"Error processing PDF: {str(e)}")
        return False

def handle_question(question: str) -> None:
    """Handle user question and get response."""
    try:
        with st.spinner('Finding the answer...'):
            response = st.session_state.processor.get_response(
                st.session_state.chain,
                question,
                st.session_state.chat_history
            )
            
            if response:
                # Add to chat history
                st.session_state.chat_history.append((question, response))
                st.session_state.last_question = question
                logger.info(f"Successfully processed question: {question[:50]}...")
            else:
                st.error("No response generated. Please try rephrasing your question.")
            
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        st.error("An error occurred while processing your question. Please try again.")
        
def display_chat_interface():
    """Display and handle the chat interface."""
    st.subheader("Ask questions about your PDF")
    
    # Chat input area
    chat_container = st.container()
    
    with chat_container:
        # Create columns for input and button
        col1, col2 = st.columns([5, 1])
        
        with col1:
            question = st.text_input(
                "Enter your question:",
                key="question_input",
                placeholder="Ask a question about the PDF content..."
            )
        
        with col2:
            submit_button = st.button("Ask", type="primary")
        
        if question and submit_button:
            if st.session_state.chain is None:
                st.warning("Please upload a PDF first.")
                return
            
            if question != st.session_state.last_question:
                handle_question(question)

def display_chat_history():
    """Display the chat history."""
    if st.session_state.chat_history:
        st.subheader("Conversation History")
        
        chat_container = st.container()
        
        with chat_container:
            for i, (question, answer) in enumerate(reversed(st.session_state.chat_history)):
                # Question box
                st.markdown("**🙋 Question:**")
                st.markdown(f"{question}")
                
                # Answer box
                st.markdown("**🤖 Answer:**")
                st.markdown(f"{answer}")
                
                # Add separator
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown("---")

def create_sidebar():
    """Create and manage sidebar elements."""
    with st.sidebar:
        st.header("📚 PDF Chatbot")
        st.markdown("---")
        
        st.subheader("Instructions")
        st.markdown("""
        1. Upload your PDF document
        2. Wait for processing to complete
        3. Ask questions about the content
        
        The chatbot will help you understand and extract information from your PDF.
        """)
        
        st.markdown("---")
        
        if st.button("Clear Conversation", type="primary"):
            reset_conversation()
            st.success("✨ Conversation cleared successfully!")
        
        if st.session_state.current_pdf:
            st.markdown("---")
            st.markdown(f"**📄 Current PDF:** {st.session_state.current_pdf}")

def main():
    """Main application function."""
    # Configure the Streamlit page
    st.set_page_config(**PAGE_CONFIG)

    # Initialize session state
    initialize_session_state()

    # Create sidebar
    create_sidebar()

    # Main content area
    st.title("📚 PDF Chatbot")

    # File upload section
    pdf_file = st.file_uploader(
        "Upload your PDF",
        type="pdf",
        help="Upload a PDF file to start asking questions"
    )
    
    if pdf_file is not None:
        # Handle PDF upload
        upload_result = handle_pdf_upload(pdf_file)
        
        if upload_result is True:
            st.success('✅ PDF processed successfully!')
        elif upload_result is False:
            st.error('❌ Failed to process PDF. Please try again.')
        
        # Display chat interface and history
        st.markdown("---")
        display_chat_interface()
        st.markdown("---")
        display_chat_history()
    
    else:
        # Welcome message
        st.markdown("""
        👋 Welcome to PDF Chatbot!
        
        This application allows you to:
        - 📁 Upload and process PDF documents
        - 💬 Ask questions about the content
        - 🤖 Get accurate answers based on the document
        
        To get started, please upload a PDF using the file uploader above.
        """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An unexpected error occurred. Please refresh the page and try again.")