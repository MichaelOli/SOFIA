import os
from time import sleep
import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image
from langchain_community.document_loaders import (
    WebBaseLoader,
    YoutubeLoader, 
    CSVLoader, 
    PyPDFLoader, 
    TextLoader
)
from fake_useragent import UserAgent

def carrega_site(url):
    documento = ''
    for i in range(5):
        try:
            os.environ['USER_AGENT'] = UserAgent().random
            loader = WebBaseLoader(url, raise_for_status=True)
            lista_documentos = loader.load()
            documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
            break
        except:
            print(f'Erro ao carregar o site {i+1}')
            sleep(3)
    if documento == '':
        st.error('Não foi possível carregar o site')
        st.stop()
    return documento

def carrega_youtube(video_id):
    loader = YoutubeLoader(video_id, add_video_info=False, language=['pt'])
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

def carrega_csv(caminho):
    loader = CSVLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

def carrega_pdf(caminho):
    loader = PyPDFLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

def carrega_txt(caminho):
    loader = TextLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

def carrega_excel(caminho):
    try:
        # Lê o arquivo Excel usando o pandas
        excel_data = pd.read_excel(caminho, sheet_name=None)  # Carrega todas as abas

        # Concatena os dados de cada aba
        documento = ''
        for nome_aba, df in excel_data.items():
            documento += f'\n\n--- {nome_aba} ---\n\n'
            documento += df.to_string(index=False)  # Converte o DataFrame em texto sem o índice

        return documento
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo Excel: {e}")
        return ""
    
def carrega_imagem(nome_arquivo):
    try:
        imagem = Image.open(nome_arquivo)
        texto = pytesseract.image_to_string(imagem)
        return texto
    except Exception as e:
        st.error(f"Erro ao carregar imagem: {e}")
        return None
