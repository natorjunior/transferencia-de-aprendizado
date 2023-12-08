def explicacao_dados(st):
    st.write('---')
    radio_dados = st.radio('',['Sobre os dados','Pneumonia X Normal','Viral X Bacteriana X Normal'],horizontal=True)
    if radio_dados =='Sobre os dados':
        sobre_os_dados(st)
def sobre_os_dados(st):
    st.write('Chest X-Ray Images (Pneumonia)')
    st.markdown('''O conjunto de dados é organizado em 2 pastas (train, test) 
        e contém subpastas para cada categoria de imagem (Pneumonia/Normal). 
        São 5.863 imagens de Raios X (JPEG) e 2 categorias (Pneumonia/Normal).
        **É possível tambem separar as imagens de Pneumonia em Viral ou Bacteriana**
    ''')
    st.image('img/img1.png')
    st.write('---')