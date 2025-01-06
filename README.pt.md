---
title: Chatbot-com-MaritacaAI-para-PDFs
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# Chatbot com MaritacaAI para PDFs

[Read this README in English](README.md)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/reichaves/Chatbot-with-MaritacaAI-for-PDFs)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbotmaritacaai-jkiyzmcjcuvduw5dzymwu5.streamlit.app/)

Este projeto implementa um sistema de Geração Aumentada por Recuperação (RAG, na sigla em inglês) conversacional usando Streamlit, LangChain e modelos de linguagem grandes da [MaritacaAI](https://www.maritaca.ai/) - uma startup brasileira focada na especialização de modelos de linguagem para domínios e idiomas específicos - especializado em Português do Brasil. O aplicativo permite aos usuários fazer upload de documentos em PDF, fazer perguntas sobre seu conteúdo e manter um histórico de conversas para contexto em diálogos contínuos.

## Autor

Reinaldo Chaves (reichaves@gmail.com)

## Funcionalidades

- Interface de usuário em Streamlit com tema escuro e layout responsivo
- Upload e processamento de múltiplos arquivos PDF
- Processamento de documentos utilizando LangChain e FAISS
- Geração de respostas usando o modelo sabia-3 da MaritacaAI, especializado em Português do Brasil
- Criação de embeddings de texto utilizando o modelo all-MiniLM-L6-v2 da Hugging Face
- Histórico de chat persistente para manter o contexto das conversas
- Barra lateral com diretrizes importantes para o usuário
- Contagem de tokens por resposta
- Formatação especial para documentos jurídicos e solicitações baseadas na Lei de Acesso à Informação (LAI)

## Requisitos

- Python 3.7 ou superior
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
- Chave da API da MaritacaAI (https://plataforma.maritaca.ai/)
- Token de API da Hugging Face (https://huggingface.co/docs/hub/security-tokens)

## Uso

1. Execute o aplicativo Streamlit:
 ```
   streamlit run app.py
   ```


2. Abra o navegador e acesse o endereço local exibido no terminal.
3. Insira suas chaves de API quando solicitado.
4. Faça o upload de um ou mais arquivos PDF.
5. Faça perguntas sobre o conteúdo dos documentos no campo de entrada de texto.

## Como funciona

1. **Upload de documentos**: Os usuários fazem o upload de arquivos PDF, que são processados e divididos em partes menores.
2. **Criação de embeddings**: O texto é convertido em embeddings usando o modelo all-MiniLM-L6-v2 da Hugging Face.
3. **Armazenamento vetorial**: Os embeddings são armazenados em um banco de dados FAISS para recuperação eficiente.
4. **Processamento de perguntas**: As perguntas dos usuários são contextualizadas com base no histórico do chat.
5. **Recuperação de informações**: O sistema recupera os trechos de texto mais relevantes com base na pergunta.
6. **Geração de respostas**: O modelo sabia-3 da MaritacaAI gera uma resposta em Português do Brasil com base nos trechos recuperados e na pergunta.
7. **Manutenção do histórico**: O histórico do chat é mantido para fornecer contexto em conversas contínuas.

## Funcionalidades especiais

- Formatação especial para análise de documentos jurídicos
- Processamento detalhado de documentos relacionados à Lei de Acesso à Informação (LAI)
- Sistema de cache para melhor desempenho
- Tratamento robusto de erros
- Interface adaptável que mantém o contexto das conversas

## Avisos importantes

- Não compartilhe documentos que contenham informações sensíveis ou confidenciais
- As respostas geradas por IA podem conter erros ou imprecisões
- Sempre verifique as informações com as fontes originais
- Este projeto é para fins educacionais e de demonstração
- Use de forma responsável e em conformidade com as políticas de uso de API

## Contribuições

Contribuições são bem-vindas! Por favor:
1. Faça um fork do projeto
2. Crie um branch para sua funcionalidade (`git checkout -b feature/AmazingFeature`)
3. Commit suas alterações (`git commit -m 'Add some AmazingFeature'`)
4. Envie para o branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Licença

[Licença MIT](LICENSE)

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
