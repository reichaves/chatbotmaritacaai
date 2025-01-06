---
title: Chatbot-with-MaritacaAI-for-PDFs
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# Chatbot with MaritacaAI for PDFs

[Leia este README em Português](README.pt.md)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/reichaves/Chatbot-with-MaritacaAI-for-PDFs)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbotmaritacaai-jkiyzmcjcuvduw5dzymwu5.streamlit.app/)

This project implements a conversational Retrieval-Augmented Generation (RAG) system using Streamlit, LangChain, and large language models from [MaritacaAI](https://www.maritaca.ai/) - a Brazilian startup focused on specializing language models for specific domains and languages - specialized in Brazilian Portuguese. The application allows users to upload PDF documents, ask questions about their content, and maintain a chat history for context in ongoing conversations.

## Author

Reinaldo Chaves (reichaves@gmail.com)

## Features

- Streamlit user interface with dark theme and responsive layout
- Upload and processing of multiple PDF files
- Document processing using LangChain and FAISS
- Answer generation using MaritacaAI's sabia-3 model specialized in Brazilian Portuguese
- Text embeddings using Hugging Face's all-MiniLM-L6-v2 model
- Persistent chat history to maintain conversation context
- Sidebar with important user guidelines
- Token count per response
- Special formatting for legal documents and FOI (Freedom of Information) requests

## Requirements

- Python 3.7+
- Streamlit
- LangChain
- FAISS
- PyPDF2
- MaritalkAI
- HuggingFace Embeddings
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/reichaves/chatbotmaritacaai.git
   cd chatbotmaritacaai
   ```

2. Install dependencies:
   ```
   pip install streamlit langchain langchain_huggingface maritalk faiss-cpu tenacity cachetools
   ```

3. Configure the necessary API keys:
   - Maritaca AI API key (https://plataforma.maritaca.ai/)
   - Hugging Face API token (https://huggingface.co/docs/hub/security-tokens)

## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open your browser and access the local address shown in the terminal.
3. Enter your API keys when prompted.
4. Upload one or more PDF files.
5. Ask questions about the documents' content in the text input box.

## How it Works

1. **Document Upload**: Users upload PDF files, which are processed and split into smaller chunks.
2. **Embedding Creation**: The text is converted into embeddings using Hugging Face's all-MiniLM-L6-v2 model.
3. **Vector Storage**: Embeddings are stored in a FAISS database for efficient retrieval.
4. **Question Processing**: User questions are contextualized based on chat history.
5. **Information Retrieval**: The system retrieves the most relevant text chunks based on the question.
6. **Answer Generation**: MaritacaAI's sabia-3 model generates an answer in Brazilian Portuguese based on the retrieved chunks and question.
7. **History Maintenance**: Chat history is maintained to provide context in ongoing conversations.

## Special Features

- Special formatting for legal document analysis
- Detailed processing of Freedom of Information (FOI) documents
- Cache system for better performance
- Robust error handling
- Adaptive interface that maintains conversation context

## Important Notices

- Do not share documents containing sensitive or confidential information
- AI-generated responses may contain errors or inaccuracies
- Always verify information with original sources
- This project is for educational and demonstration purposes
- Use responsibly and in compliance with API usage policies

## Contributions

Contributions are welcome! Please:
1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

[MIT License](LICENSE)

## Citation

If you use this project in your research or application, please cite:

```
@software{chatbot-maritacaai-pdfs,
  author = {Reinaldo Chaves},
  title = {Chatbot with MaritacaAI for PDFs},
  year = {2024},
  url = {https://github.com/reichaves/chatbotmaritacaai/}
}
```
