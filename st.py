import streamlit as st
from  dados.conjunto_de_dados import *
st.title('Transferencia de Aprendizado')

pagina = st.sidebar.selectbox('Página:',['Conjunto de dados','VGG19','InceptionV3'])

if pagina == 'Conjunto de dados':
    explicacao_dados(st)
elif pagina == 'VGG19':
    pass
elif pagina == 'InceptionV3':
    pass
