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

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/reichaves/Chatbot-with-MaritacaAI-for-PDFs)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbotmaritacaai-jkiyzmcjcuvduw5dzymwu5.streamlit.app/)

Este projeto implementa um sistema de Recuperação de Informações Aumentada por Geração (RAG) conversacional usando Streamlit, LangChain, e modelos de linguagem de grande escala da [MaritacaAI](https://www.maritaca.ai/) - startup brasileira focada em especializar modelos de linguagem para certos domínios e idiomas - especializada no Português Brasileiro. O aplicativo permite que os usuários façam upload de documentos PDF, façam perguntas sobre o conteúdo desses documentos, e mantenham um histórico de chat para contexto em conversas contínuas.

## Autor

Reinaldo Chaves (reichaves@gmail.com)

## Características

- Interface de usuário Streamlit com tema dark e layout responsivo
- Upload e processamento de múltiplos arquivos PDF
- Processamento de documentos usando LangChain e FAISS
- Geração de respostas usando o modelo sabia-3 da Maritaca AI especializado em Português do Brasil
- Embeddings de texto usando o modelo all-MiniLM-L6-v2 do Hugging Face
- Histórico de chat persistente para manter o contexto da conversa
- Barra lateral com orientações importantes para o usuário
- Contagem de tokens por resposta
- Formatação especial para documentos jurídicos e pedidos LAI

## Requisitos

- Python 3.7+
- Streamlit
- LangChain
- FAISS
- PyPDF2
- MaritalkAI
- HuggingFace Embeddings
- Outras dependências listadas em `requirements.txt`

## Instalação

1. Clone este repositório:
   ```
   git clone https://github.com/reichaves/chatbotmaritacaai.git
   cd chatbotmaritacaai
   ```

2. Instale as dependências:
   ```
   pip install streamlit langchain langchain_huggingface maritalk faiss-cpu tenacity cachetools
   ```

3. Configure as chaves de API necessárias:
   - Chave da API Maritaca AI (https://plataforma.maritaca.ai/)
   - Token da API Hugging Face (https://huggingface.co/docs/hub/security-tokens)

## Uso

1. Execute o aplicativo Streamlit:
   ```
   streamlit run app.py
   ```

2. Abra o navegador e acesse o endereço local mostrado no terminal.
3. Insira suas chaves de API quando solicitado.
4. Faça upload de um ou mais arquivos PDF.
5. Faça perguntas sobre o conteúdo dos documentos na caixa de entrada de texto.

## Como funciona

1. **Upload de Documentos**: Os usuários fazem upload de arquivos PDF, que são processados e divididos em chunks menores.
2. **Criação de Embeddings**: O texto é convertido em embeddings usando o modelo all-MiniLM-L6-v2 do Hugging Face.
3. **Armazenamento de Vetores**: Os embeddings são armazenados em um banco de dados FAISS para recuperação eficiente.
4. **Processamento de Perguntas**: As perguntas dos usuários são contextualizadas com base no histórico do chat.
5. **Recuperação de Informações**: O sistema recupera os chunks de texto mais relevantes com base na pergunta.
6. **Geração de Respostas**: O modelo sabia-3 da Maritaca AI gera uma resposta em Português do Brasil com base nos chunks recuperados e na pergunta.
7. **Manutenção do Histórico**: O histórico do chat é mantido para fornecer contexto em conversas contínuas.

## Funcionalidades Especiais

- Formatação especial para análise de documentos jurídicos
- Processamento detalhado de documentos da Lei de Acesso à Informação (LAI)
- Sistema de cache para melhor performance
- Tratamento de erros robusto
- Interface adaptativa que mantém o contexto da conversa

## Avisos Importantes

- Não compartilhe documentos contendo informações sensíveis ou confidenciais
- As respostas geradas pela IA podem conter erros ou imprecisões
- Sempre verifique as informações com as fontes originais
- Este projeto é para fins educacionais e de demonstração
- Use com responsabilidade e em conformidade com as políticas de uso das APIs

## Contribuições

Contribuições são bem-vindas! Por favor:
1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Licença

[MIT License](LICENSE)

## Citação

Se você usar este projeto em sua pesquisa ou aplicação, por favor cite:

```
@software{chatbot-maritacaai-pdfs,
  author = {Reinaldo Chaves},
  title = {Chatbot with MaritacaAI for PDFs},
  year = {2024},
  url = {https://github.com/reichaves/chatbotmaritacaai/}
}
```
