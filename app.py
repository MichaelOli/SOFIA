import tempfile
import streamlit as st
import pandas as pd  
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import pytesseract
from langchain.memory import ConversationBufferMemory

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

from loaders import *

TIPOS_ARQUIVOS_VALIDOS = [
    'Site', 'Youtube', 'Pdf', 'Csv', 'Txt', 'Excel', 'Imagem' 
]

CONFIG_MODELOS = {
    'Groq': {
        'modelos': ['llama-3.1-70b-versatile', 'gemma2-9b-it', 'mixtral-8x7b-32768'],
        'chat': ChatGroq
    },
    'OpenAI': {
        'modelos': ['gpt-4o-mini', 'gpt-4o', 'o1-preview', 'o1-mini'],
        'chat': ChatOpenAI
    },
    'Google Gemini': {
        'modelos': ['gemini-1.5-pro', 'gemini-1.5-flash','gemini-1.0-pro'],
        'chat': ChatGoogleGenerativeAI
    }
}

MEMORIA = ConversationBufferMemory()

def carrega_arquivos(tipo_arquivo, arquivo):
    if tipo_arquivo == 'Site':
        documento = carrega_site(arquivo)
    elif tipo_arquivo == 'Youtube':
        documento = carrega_youtube(arquivo)
    elif tipo_arquivo == 'Pdf':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_pdf(nome_temp)
    elif tipo_arquivo == 'Csv':
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_csv(nome_temp)
    elif tipo_arquivo == 'Txt':
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_txt(nome_temp)
    elif tipo_arquivo == 'Excel':  # Nova opção para arquivos Excel
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_excel(nome_temp)  # Função de carregamento para Excel
    elif tipo_arquivo == 'Imagem':  # Nova opção para arquivos de imagem
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_imagem(nome_temp)  # Função de carregamento para imagens
    return documento

def carrega_excel(nome_arquivo):
    # Lê o arquivo Excel com Pandas
    dados = pd.read_excel(nome_arquivo)
    # Converte os dados para string para serem passados no sistema de prompt
    return dados.to_string(index=False)

def carrega_imagem(nome_arquivo):
    # Carregar a imagem usando PIL
    img = Image.open(nome_arquivo)
    # Usar OCR para extrair texto da imagem
    texto = pytesseract.image_to_string(img)
    return texto

def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):
    documento = carrega_arquivos(tipo_arquivo, arquivo)
    
    system_message = f'''Você é uma assistente amigável chamada SOFIA. SE ALGUEM PERGUNTAR POR QUE SE CHAMA 
                    SOFIA É POR QUE O SEU CRIADOR MICHAEL, GOSTA MUITO DE SOFTWARE E INTELIGENCIA ARTIFICIAL, como VOCÊ É UM MODELO DE IA, DAI O NOME 
                    SOF = SOFTWARE E IA = INTELIGENCIA ARTIFICIAL
                    Você possui acesso às seguintes informações vindas 
                    de um documento {tipo_arquivo}: 

                    ####
                    {documento}
                    ####

                    Utilize as informações fornecidas para basear as suas respostas.

                    Sempre que houver $ na sua saída, substitua por S. ou caso identifique caracteres especiais, 
                    tente tratá-los corretamente.

                    Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue" 
                    sugira ao usuário carregar novamente a Sofia!'''

    print(system_message)

    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])
    chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key)
    chain = template | chat

    st.session_state['chain'] = chain

def pagina_chat():
    st.header('🤖Bem-vindo a Consulta com a SOFIA', divider=True)

    chain = st.session_state.get('chain')
    if chain is None:
        st.error('Carrege a Sofia antes de comecar a conversar')
        st.stop()

    memoria = st.session_state.get('memoria', MEMORIA)
    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    input_usuario = st.chat_input('Digite o que voce deseja conversar com a Sofia, aqui')
    if input_usuario:
        chat = st.chat_message('human')
        chat.markdown(input_usuario)

        chat = st.chat_message('ai')
        resposta = chat.write_stream(chain.stream({
            'input': input_usuario, 
            'chat_history': memoria.buffer_as_messages
        }))
        
        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria

def sidebar():
    tabs = st.tabs(['Upload de Arquivos', 'Seleção de Modelos'])
    with tabs[0]:
        tipo_arquivo = st.selectbox('Selecione o tipo de arquivo', TIPOS_ARQUIVOS_VALIDOS)
        if tipo_arquivo == 'Site':
            arquivo = st.text_input('Digite a url do site')
        if tipo_arquivo == 'Youtube':
            arquivo = st.text_input('Digite a url do vídeo')
        if tipo_arquivo == 'Pdf':
            arquivo = st.file_uploader('Faça o upload do arquivo pdf', type=['.pdf'])
        if tipo_arquivo == 'Csv':
            arquivo = st.file_uploader('Faça o upload do arquivo csv', type=['.csv'])
        if tipo_arquivo == 'Txt':
            arquivo = st.file_uploader('Faça o upload do arquivo txt', type=['.txt'])
        if tipo_arquivo == 'Excel':  # Adiciona o upload para Excel
            arquivo = st.file_uploader('Faça o upload do arquivo Excel', type=['.xlsx', '.xls'])
        if tipo_arquivo == 'Imagem':
            arquivo = st.file_uploader('Faça o upload da imagem', type=['.png', '.jpg', '.jpeg'])
    with tabs[1]:
        provedor = st.selectbox('Selecione o provedor dos modelo', CONFIG_MODELOS.keys())
        modelo = st.selectbox('Selecione o modelo', CONFIG_MODELOS[provedor]['modelos'])
        api_key = st.text_input(
            f'Adicione a api key para o provedor {provedor}',
            value=st.session_state.get(f'api_key_{provedor}'))
        st.session_state[f'api_key_{provedor}'] = api_key
    
    st.warning('Lembre-se de adicionar a api key do provedor escolhido, antes de iniciar uma conversa com a Sofia', icon="⚠️")
    if st.button('Iniciar conversa com a Sofia', use_container_width=True):
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    if st.button('Apagar Histórico de Conversa', use_container_width=True):
        st.session_state['memoria'] = MEMORIA

def main():
    with st.sidebar:
        sidebar()
    pagina_chat()

if __name__ == '__main__':
    main()
