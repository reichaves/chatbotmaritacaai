# -*- coding: utf-8
# Reinaldo Chaves (reichaves@gmail.com)
# Este projeto implementa um sistema de Recuperação de Informações Aumentada por Geração (RAG) conversacional 
# usando Streamlit, LangChain, e modelos de linguagem de grande escala - para entrevistar PDFs
# Geração de respostas usando o modelo sabia-3 da Maritaca AI especializado em Português do Brasil
# Embeddings de texto usando o modelo all-MiniLM-L6-v2 do Hugging Face
##

"""
Chatbot com RAG (Retrieval Augmented Generation) para PDFs usando MaritacaAI

Este script implementa um chatbot que pode analisar documentos PDF usando:
- Streamlit para interface web
- LangChain para processamento de documentos e gerenciamento de chat
- Modelo sabia-3 da Maritaca AI para geração de respostas em Português
- Embeddings do Hugging Face para processamento de texto

"""

# Importação das bibliotecas principais
import streamlit as st  # Framework para interface web
import os  # Operações do sistema operacional
import tempfile  # Manipulação de arquivos temporários
from typing import List, Dict, Any, Optional  # Tipos para type hints
from tenacity import retry, stop_after_attempt, wait_exponential  # Gerenciamento de retentativas
from cachetools import TTLCache  # Cache com tempo de vida
import logging  # Sistema de logging
from datetime import datetime  # Manipulação de datas

# Configuração do sistema de logging para debug e monitoramento
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Desativa paralelismo dos tokenizers para evitar deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Importações do LangChain para processamento de documentos e chat
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import Runnable
from maritalk import MariTalk

# Cache para armazenar embeddings e melhorar performance
embeddings_cache = TTLCache(maxsize=100, ttl=3600)  # Cache por 1 hora

class MariTalkWrapper(Runnable):
    """
    Wrapper para integrar o modelo MaritacaAI com o LangChain.
    Gerencia a comunicação com a API e formata mensagens.
    """
    
    def __init__(self, maritalk_model: Any, max_retries: int = 3, timeout: int = 30):
        """
        Inicializa o wrapper com configurações de retry e timeout
        """
        self.maritalk_model = maritalk_model
        self.max_retries = max_retries
        self.timeout = timeout

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def invoke(self, input: Any, config: Optional[Dict] = None) -> str:
        """
        Processa entrada e gera resposta, com retries automáticos em caso de falha.
        Suporta diferentes formatos de entrada: ChatPromptValue, dict, string
        """
        try:
            # Processamento de ChatPromptValue (formato LangChain)
            if hasattr(input, "to_messages"):
                messages = input.to_messages()
                formatted_messages = self._format_messages(messages)
                response = self.maritalk_model.generate(formatted_messages)
                if isinstance(response, str):
                    return response
                elif isinstance(response, dict) and 'text' in response:
                    return response['text']
                else:
                    return response[0]['content'] if isinstance(response, list) else str(response)
            
            # Processamento de dicionário
            elif isinstance(input, dict):
                if "messages" in input:
                    messages = input["messages"]
                    formatted_messages = self._format_messages(messages)
                    response = self.maritalk_model.generate(formatted_messages)
                    if isinstance(response, str):
                        return response
                    elif isinstance(response, dict) and 'text' in response:
                        return response['text']
                    else:
                        return response[0]['content'] if isinstance(response, list) else str(response)
                elif "answer" in input:
                    return str(input["answer"])
                else:
                    return self._process_text(str(input))
            
            # Processamento de string simples
            elif isinstance(input, str):
                return self._process_text(input)
            
            else:
                raise TypeError(f"Tipo de input não suportado: {type(input)}")
        
        except Exception as e:
            logger.error(f"Erro na chamada à API MariTalk: {str(e)}")
            logger.error(f"Input recebido: {str(input)[:200]}")
            logger.error(f"Tipo do input: {type(input)}")
            raise

    def _format_messages(self, messages: List[Any]) -> List[Dict[str, str]]:
        """
        Formata mensagens para o formato esperado pela API da Maritaca
        Converte entre formatos LangChain e Maritaca
        """
        formatted = []
        for msg in messages:
            role = "user"
            if hasattr(msg, "type"):
                if msg.type == "human":
                    role = "user"
                elif msg.type in ["ai", "assistant"]:
                    role = "assistant"
                elif msg.type == "system":
                    role = "system"
                formatted.append({"role": role, "content": msg.content})
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                formatted.append({"role": role, "content": content})
            else:
                formatted.append({"role": "user", "content": str(msg)})
        return formatted

    def _process_text(self, text: str) -> str:
        """
        Processa texto simples através do modelo Maritaca
        Gerencia diferentes formatos de resposta possíveis
        """
        response = self.maritalk_model.generate([{"role": "user", "content": text}])
        if isinstance(response, str):
            return response
        elif isinstance(response, dict) and 'text' in response:
            return response['text']
        elif isinstance(response, list) and len(response) > 0:
            return response[0].get('content', str(response))
        else:
            return str(response)

def init_page_config():
    """
    Inicializa a configuração da página Streamlit
    Define título, layout e ícone
    """
    st.set_page_config(
        page_title="Chatbot com IA especializada em Português do Brasil - entrevista PDFs",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="📚"
    )

def apply_custom_css():
    """
    Aplica estilos CSS personalizados à interface
    Define cores, formatos e layout dos elementos
    """
    st.markdown("""
    <style>
    /* Estilo global e tema dark */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Formatação das mensagens do chat */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        width: 100%;
    }
    .user-message {
        background-color: #2e2e2e;
    }
    .assistant-message {
        background-color: #1e1e1e;
    }
    
    /* Elementos do chat */
    .chat-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .chat-content {
        margin-left: 1rem;
        white-space: pre-line;
    }
    .chat-content em {
        color: #888;
        font-size: 0.9em;
        display: block;
        margin-top: 10px;
        border-top: 1px solid #444;
        padding-top: 8px;
    }
    
    /* Elementos da interface */
    .main-title {
        color: #FFA500;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 30px;
    }
    .stButton > button {
        background-color: #262730;
        color: #4F8BF9;
        border-radius: 20px;
        padding: 10px 20px;
    }
    
    /* Esconde elementos desnecessários */
    div[data-testid="stToolbar"] {
        display: none;
    }
    .stDeployButton {
        display: none;
    }
    
    /* Informações de tokens */
    .token-info {
        font-style: italic;
        color: #888;
        margin-top: 10px;
        padding-top: 8px;
        border-top: 1px solid #444;
    }
    </style>
    """, unsafe_allow_html=True)

def create_sidebar():
    """
    Cria a barra lateral com instruções e informações importantes
    Inclui links para obtenção de API keys e avisos
    """
    st.sidebar.markdown("## Orientações")
    st.sidebar.markdown("""
    * Se encontrar erros de processamento, reinicie com F5. Utilize arquivos .PDF com textos não digitalizados como imagens.
    * Para recomeçar uma nova sessão pressione F5.

    **Obtenção de chaves de API:**
    * Você pode fazer uma conta na MaritacaAI e obter uma chave de API [aqui](https://plataforma.maritaca.ai/)
    * Você pode fazer uma conta no Hugging Face e obter o token de API Hugging Face [aqui](https://huggingface.co/docs/hub/security-tokens)

    **Atenção:** Os documentos que você compartilhar com o modelo de IA generativa podem ser usados pelo LLM para treinar o sistema. 
    Portanto, evite compartilhar documentos PDF que contenham:
    1. Dados bancários e financeiros
    2. Dados de sua própria empresa
    3. Informações pessoais
    4. Informações de propriedade intelectual
    5. Conteúdos autorais

    E não use IA para escrever um texto inteiro! O auxílio é melhor para gerar resumos, filtrar informações ou auxiliar a 
    entender contextos - que depois devem ser checados. Inteligência Artificial comete erros (alucinações, viés, baixa qualidade, 
    problemas éticos)!

    Este projeto não se responsabiliza pelos conteúdos criados a partir deste site.

    **Sobre este app**
    Este aplicativo foi desenvolvido por Reinaldo Chaves. Para mais informações, contribuições e feedback, visite o 
    [repositório](https://github.com/reichaves/chatbotmaritacaai)
    """)

def display_chat_message(message: str, is_user: bool):
    """
    Exibe uma mensagem no chat
    Formata diferentemente mensagens do usuário e do assistente
    Inclui contagem de tokens para respostas do assistente
    """
    class_name = "user-message" if is_user else "assistant-message"
    role = "Você" if is_user else "Assistente"
    
    # Formata resposta do assistente
    if not is_user:
        if isinstance(message, dict):
            content = message.get('answer', str(message))
            tokens = message.get('usage', {}).get('total_tokens', None)
            
            # Quebras de linha em HTML
            content = content.replace('\n', '<br>')
            
            if tokens:
                content = f"{content}<br><br><em>Total tokens: {tokens}</em>"
        else:
            content = str(message)
    else:
        content = message
    
    # Renderiza mensagem com HTML
    st.markdown(f"""
        <div class="chat-message {class_name}">
            <div class="chat-header">{role}:</div>
            <div class="chat-content">{content}</div>
        </div>
    """, unsafe_allow_html=True)

def setup_rag_chain(documents: List[Any], llm: Any, embeddings: Any) -> Any:
    """
    Configura a chain RAG (Retrieval Augmented Generation)
    Processa documentos e configura sistema de recuperação e resposta
    """
    # Divide documentos em chunks menores
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Tamanho de cada chunk
        chunk_overlap=200,  # Sobreposição entre chunks
        length_function=len,
        is_separator_regex=False
    )
    
    # Cria banco de vetores com FAISS
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()
    
    # Prompt para contextualização de perguntas
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Você é um assistente especializado em analisar documentos PDF com um contexto jornalístico, 
            como documentos da Lei de Acesso à Informação, contratos públicos e processos judiciais. 
            Sempre coloque no final das respostas: 'Todas as informações devem ser checadas com a(s) fonte(s) original(ais)'
            Responda em Português do Brasil a menos que seja pedido outro idioma
            Se você não sabe a resposta, diga que não sabe
            Siga estas diretrizes:\n\n
            1. Explique os passos de forma simples e mantenha as respostas concisas.\n
            2. Inclua links para ferramentas, pesquisas e páginas da Web citadas.\n
            3. Ao resumir passagens, escreva em nível universitário.\n
            4. Divida tópicos em partes menores e fáceis de entender quando relevante.\n
            5. Seja claro, breve, ordenado e direto nas respostas.\n
            6. Evite opiniões e mantenha-se neutro.\n
            7. Base-se nas classes processuais do Direito no Brasil conforme o site do CNJ.\n
            8. Se não souber a resposta, admita que não sabe.\n\n
            Ao analisar processos judiciais, priorize:\n
            - Identificar se é petição inicial, decisão ou sentença\n
            - Apresentar a ação e suas partes\n
            - Explicar os motivos do ajuizamento\n
            - Listar os requerimentos do autor\n
            - Expor o resultado das decisões\n
            - Indicar o status do processo\n\n
            Para licitações ou contratos públicos, considere as etapas do processo licitatório e as modalidades de licitação.\n\n
            Para documentos da Lei de Acesso à Informação (LAI), inclua:\n
            - Data\n
            - Protocolo NUP\n
            - Nome do órgão público\n
            - Nomes dos responsáveis pela resposta\n
            - Data da resposta\n
            - Se o pedido foi totalmente atendido, parcialmente ou negado\n\n
            Use o seguinte contexto para responder à pergunta: {context}\n\n
            Sempre termine as respostas com: 'Todas as informações precisam ser checadas com as fontes das informações'."
            )
        """),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Prompt para respostas
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Você é um assistente especializado em análise de documentos.
        
        Use este contexto para responder à pergunta: {context}
        
        Diretrizes:
        1. Use o contexto fornecido E o histórico do chat para suas respostas
        2. Mantenha consistência com respostas anteriores
        3. Se uma informação não estiver no contexto mas foi mencionada antes, você pode usá-la
        4. Seja conciso mas mantenha a coerência com o histórico
        5. Se houver contradição entre o histórico e o novo contexto, mencione isso
        
        Responda em Português do Brasil.
         
         Mais orientações:
         Você é um assistente especializado em analisar documentos PDF com um contexto jornalístico, 
            como documentos da Lei de Acesso à Informação, contratos públicos e processos judiciais. 
            Sempre coloque no final das respostas: 'Todas as informações devem ser checadas com a(s) fonte(s) original(ais)'
            Responda em Português do Brasil a menos que seja pedido outro idioma
            Se você não sabe a resposta, diga que não sabe
            Siga estas diretrizes:\n\n
            1. Explique os passos de forma simples e mantenha as respostas concisas.\n
            2. Inclua links para ferramentas, pesquisas e páginas da Web citadas.\n
            3. Ao resumir passagens, escreva em nível universitário.\n
            4. Divida tópicos em partes menores e fáceis de entender quando relevante.\n
            5. Seja claro, breve, ordenado e direto nas respostas.\n
            6. Evite opiniões e mantenha-se neutro.\n
            7. Base-se nas classes processuais do Direito no Brasil conforme o site do CNJ.\n
            8. Se não souber a resposta, admita que não sabe.\n\n
            Ao analisar processos judiciais, priorize:\n
            - Identificar se é petição inicial, decisão ou sentença\n
            - Apresentar a ação e suas partes\n
            - Explicar os motivos do ajuizamento\n
            - Listar os requerimentos do autor\n
            - Expor o resultado das decisões\n
            - Indicar o status do processo\n\n
            Para licitações ou contratos públicos, considere as etapas do processo licitatório e as modalidades de licitação.\n\n
            Para documentos da Lei de Acesso à Informação (LAI), inclua:\n
            - Data\n
            - Protocolo NUP\n
            - Nome do órgão público\n
            - Nomes dos responsáveis pela resposta\n
            - Data da resposta\n
            - Se o pedido foi totalmente atendido, parcialmente ou negado\n\n
            Use o seguinte contexto para responder à pergunta: {context}\n\n
            Sempre termine as respostas com: 'Todas as informações precisam ser checadas com as fontes das informações'."
            )
        """),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Cria chain com histórico
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

def process_documents(uploaded_files: List[Any]) -> List[Any]:
    """
    Processa arquivos PDF enviados
    Carrega documentos e mostra progresso
    """
    documents = []
    progress_bar = st.progress(0)
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Cria arquivo temporário
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name
            
            # Carrega PDF
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            documents.extend(docs)
            os.unlink(temp_file_path)  # Remove arquivo temporário
            
            # Atualiza barra de progresso
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        except Exception as e:
            logger.error(f"Erro ao processar {uploaded_file.name}: {str(e)}")
            st.error(f"Erro ao processar {uploaded_file.name}")
    
    progress_bar.empty()  # Remove barra de progresso
    return documents

def display_chat_interface():
    """
    Exibe a interface do chat com campo de entrada fixo
    Gerencia histórico e entrada do usuário
    """
    # Container para histórico do chat
    chat_container = st.container()
    
    # Container fixo para campo de entrada
    input_container = st.container()
    
    # Campo de entrada de texto
    with input_container:
        user_input = st.text_input("💭 Sua pergunta:", key=f"user_input_{len(st.session_state.get('messages', []))}")
    
    # Exibe histórico de mensagens
    with chat_container:
        if 'messages' in st.session_state:
            for msg in st.session_state.messages:
                display_chat_message(msg["content"], msg["role"] == "user")
    
    return user_input

def update_chat_history(user_input: str, assistant_response: Any):
    """
    Atualiza o histórico do chat com novas mensagens
    Mantém conversas no estado da sessão
    """
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Adiciona mensagens ao histórico
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

def main():
    """
    Função principal do aplicativo
    Gerencia todo o fluxo da aplicação
    """
    # Inicialização da interface
    init_page_config()
    apply_custom_css()
    create_sidebar()
    
    # Título principal
    st.markdown('<h1 class="main-title">Chatbot com modelo de IA especializado em Português do Brasil - entrevista PDFs 📚</h1>', unsafe_allow_html=True)
    
    # Campos de API em duas colunas
    col1, col2 = st.columns(2)
    with col1:
        maritaca_api_key = st.text_input("Chave API Maritaca:", type="password")
    with col2:
        huggingface_api_token = st.text_input("Token API Hugging Face:", type="password")
    
    # Verifica chaves de API
    if not (maritaca_api_key and huggingface_api_token):
        st.warning("⚠️ Insira as chaves de API para continuar")
        return
    
    # Configura ambiente com token Hugging Face
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_token
    
    try:
        # Inicializa modelos de IA
        maritalk_model = MariTalk(key=maritaca_api_key, model="sabia-3")
        llm = MariTalkWrapper(maritalk_model)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Erro ao inicializar modelos: {str(e)}")
        return
    
    # Inicializa estado da sessão
    if 'store' not in st.session_state:
        st.session_state.store = {}
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # ID da sessão e botão limpar
    col1, col2 = st.columns([3, 1])
    with col1:
        session_id = st.text_input("ID da Sessão:", value=datetime.now().strftime("%Y%m%d_%H%M%S"))
    with col2:
        if st.button("🗑️ Limpar Chat"):
            # Limpa todos os dados da sessão
            for key in ['messages', 'documents', 'documents_processed', 'rag_chain']:
                if key in st.session_state:
                    del st.session_state[key]
            
            if session_id in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            st.success("Chat limpo com sucesso!")
            st.rerun()
    
    # Upload de arquivos PDF
    uploaded_files = st.file_uploader(
        "Upload de PDFs:",
        type="pdf",
        accept_multiple_files=True,
        help="Selecione um ou mais arquivos PDF"
    )
    
    if not uploaded_files:
        st.info("📤 Faça upload de PDFs para começar")
        return
    
    # Processamento de documentos
    if uploaded_files:
        if 'documents_processed' not in st.session_state or not st.session_state.documents_processed:
            documents = process_documents(uploaded_files)
            if not documents:
                st.error("❌ Nenhum documento processado")
                return
            st.session_state.documents = documents
            st.session_state.documents_processed = True
            
            # Configura RAG chain
            try:
                rag_chain = setup_rag_chain(documents, llm, embeddings)
                st.session_state.rag_chain = rag_chain
                st.success(f"✅ {len(documents)} documentos processados")
            except Exception as e:
                logger.error(f"Erro ao configurar RAG chain: {str(e)}")
                st.error("Erro ao configurar o sistema")
                return
    
    try:
        # Configuração do histórico da sessão
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]
        
        # Criação da chain conversacional
        conversational_rag_chain = RunnableWithMessageHistory(
            st.session_state.rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
    except Exception as e:
        logger.error(f"Erro ao configurar RAG chain: {str(e)}")
        st.error("Erro ao configurar o sistema")
        return
    
    # Interface do chat
    user_input = display_chat_interface()
    
    # Processamento da entrada do usuário
    if user_input:
        with st.spinner("🤔 Pensando..."):
            try:
                # Gera resposta
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                
                # Logging da resposta
                logger.info(f"Tipo da resposta: {type(response)}")
                logger.info(f"Conteúdo da resposta: {str(response)[:200]}...")
                
                # Atualiza históricos
                update_chat_history(user_input, response)
                
                history = get_session_history(session_id)
                history.add_user_message(user_input)
                if isinstance(response, dict) and 'answer' in response:
                    history.add_ai_message(response['answer'])
                else:
                    history.add_ai_message(str(response))
                
                # Atualiza interface
                st.rerun()
                    
            except Exception as e:
                logger.error(f"Erro ao processar pergunta: {str(e)}", exc_info=True)
                st.error(f"❌ Erro ao processar sua pergunta: {str(e)}")

# Ponto de entrada do aplicativo
if __name__ == "__main__":
    main()
