import os 
import plotly.graph_objs as go
from PIL import Image

def explicacao_dados(st):
    st.write('---')
    radio_dados = st.radio('',['Sobre os dados','Pneumonia X Normal','Viral X Bacteriana X Normal'],horizontal=True)
    if radio_dados =='Sobre os dados':
        sobre_os_dados(st)
    elif radio_dados =='Pneumonia X Normal':
        normal_x_pneumonia(st)
    elif radio_dados == 'Viral X Bacteriana X Normal':
        viral_bacteriana_normal(st)
def sobre_os_dados(st):
    st.write('Chest X-Ray Images (Pneumonia)')
    st.markdown('''O conjunto de dados é organizado em 2 pastas (train, test) 
        e contém subpastas para cada categoria de imagem (Pneumonia/Normal). 
        São 5.863 imagens de Raios X (JPEG) e 2 categorias (Pneumonia/Normal).
        **É possível tambem separar as imagens de Pneumonia em Viral ou Bacteriana**
    ''')
    st.image('img/img1.png')
    st.write('---')
def normal_x_pneumonia(st):
    col1_normal_x_pneumonia,col2_normal_x_pneumonia = st.columns(2)
    with col1_normal_x_pneumonia:
        select_normal_x_pneumonia = st.selectbox('PNEUMONIA',os.listdir('img/test/PNEUMONIA'))
        imagem = Image.open('img/test/PNEUMONIA/'+select_normal_x_pneumonia)
        layout = go.Layout(
            images=[go.layout.Image(
                source=imagem,
                x=0,
                y=1,
                xref="paper",
                yref="paper",
                sizex=1,
                sizey=1,
                opacity=0.8,
                layer="below")]
        )
        fig = go.Figure(layout=layout)
        st.plotly_chart(fig, use_container_width=True)
    with col2_normal_x_pneumonia:
        select_normal_x_pneumonia2 = st.selectbox('NORMAL',os.listdir('img/test/NORMAL'))
        imagem = Image.open('img/test/NORMAL/'+select_normal_x_pneumonia2)
        layout = go.Layout(
            images=[go.layout.Image(
                source=imagem,
                x=0,
                y=1,
                xref="paper",
                yref="paper",
                sizex=1,
                sizey=1,
                opacity=0.8,
                layer="below")]
        )
        fig = go.Figure(layout=layout)
        st.plotly_chart(fig, use_container_width=True)

def viral_bacteriana_normal(st):
    col1_normal_x_pneumonia_bacteria,col2_normal_x_pneumonia_virus,col3_normal_x_pneumonia_normal = st.columns(3)
    with col1_normal_x_pneumonia_bacteria:
        lista_de_arquivos = os.listdir('img/test/PNEUMONIA')
        substring_bacteria = 'bacteria'
        lista_bacteria = [arquivo for arquivo in lista_de_arquivos if substring_bacteria in arquivo]
        select_normal_x_pneumonia_bacteria = st.selectbox('PNEUMONIA(bacteria)',lista_bacteria)
        imagem_bacteria = Image.open('img/test/PNEUMONIA/'+select_normal_x_pneumonia_bacteria)
        layout = go.Layout(
            images=[go.layout.Image(
                source=imagem_bacteria,
                x=0,
                y=1,
                xref="paper",
                yref="paper",
                sizex=1,
                sizey=1,
                opacity=0.8,
                layer="below")]
        )
        fig = go.Figure(layout=layout)
        st.plotly_chart(fig, use_container_width=True)
    with col2_normal_x_pneumonia_virus:
        lista_de_arquivos = os.listdir('img/test/PNEUMONIA')
        substring_virus = 'virus'
        lista_virus = [arquivo for arquivo in lista_de_arquivos if substring_virus in arquivo]
        select_normal_x_pneumonia_virus = st.selectbox('PNEUMONIA(virus)',lista_virus)
        imagem_virus = Image.open('img/test/PNEUMONIA/'+select_normal_x_pneumonia_virus)
        layout = go.Layout(
            images=[go.layout.Image(
                source=imagem_virus,
                x=0,
                y=1,
                xref="paper",
                yref="paper",
                sizex=1,
                sizey=1,
                opacity=0.8,
                layer="below")]
        )
        fig = go.Figure(layout=layout)
        st.plotly_chart(fig, use_container_width=True)
    with col3_normal_x_pneumonia_normal:
        select_normal_x_pneumonia2 = st.selectbox('NORMAL',os.listdir('img/test/NORMAL'))
        imagem = Image.open('img/test/NORMAL/'+select_normal_x_pneumonia2)
        layout = go.Layout(
            images=[go.layout.Image(
                source=imagem,
                x=0,
                y=1,
                xref="paper",
                yref="paper",
                sizex=1,
                sizey=1,
                opacity=0.8,
                layer="below")]
        )
        fig = go.Figure(layout=layout)
        st.plotly_chart(fig, use_container_width=True)


