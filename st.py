import streamlit as st
from  dados.conjunto_de_dados import *
from  modelos.vgg19 import *
st.sidebar.title('Transferencia de Aprendizado')

pagina = st.sidebar.selectbox('Página:',['Conjunto de dados','VGG19','InceptionV3'])

if pagina == 'Conjunto de dados':
    explicacao_dados(st)
elif pagina == 'VGG19':
    explicacao_vgg19(st)
elif pagina == 'InceptionV3':
    pass
