# PDF Chatbot

A Streamlit application that allows users to ask questions about PDF documents using OpenAI's language models.

## Quick Start

1. Open the project folder

2. Create virual environment
```bash
python -m venv virtual_env
source virtual_env/bin/activate
```
2. Install Dependencies

```bash
pip install -r requirements.txt
```

2. Create a .env file same as .env.template :
```
OPENAI_API_KEY=your-key-here
```

3. Run the app:
```bash
streamlit run app.py
```

## Features
- Upload and process PDF documents
- Ask questions about PDF content
- Get AI-powered responses
- Confidence-based fallback system
- Persistent chat history
- Fallback system if question is not relevant to pdf uploaded


## Usage
1. Upload a PDF using the file uploader
2. Wait for processing to complete
3. Ask questions about the document
4. View responses and chat history

## Security Note
- Keep API keys secure
- Use environment variables

## Structure
pdf-chatbot/
├── app.py              # Streamlit interface
├── pdf_processor.py    # Core logic
├── .env                # API keys (create this)
├── requirements.txt      # Dependencies
└── virtual_env 