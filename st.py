import streamlit as st
from  dados.conjunto_de_dados import *
from  modelos.vgg19 import *
from  modelos.inceptionV3 import *
st.set_page_config(layout="wide")
st.sidebar.title('Transferencia de Aprendizado')

pagina = st.sidebar.selectbox('Página:',['Sobre','Conjunto de dados','VGG19','InceptionV3'])

if pagina == 'Sobre':
    #st.title('SOBRE')
    st.markdown('''
## Resumo

A plataforma utiliza técnicas de **transferência de aprendizado** para classificar imagens de casos de pneumonia e imagens normais. Ao empregar um modelo de aprendizado pré-treinado.

Esta abordagem oferece vantagens significativas:
- **Conhecimento Prévio:** Aproveita o conhecimento adquirido durante o treinamento em grandes conjuntos de dados.
- **Eficiência na Detecção:** Proporciona resultados eficazes na identificação de padrões associados à presença de pneumonia.

#### Nesta abordagem usamos dois modelo de classificacao pre-treinados
                
## VGG19

A **VGG19** é uma arquitetura de rede neural convolucional (CNN) notável por sua simplicidade e eficácia. Desenvolvida pelo Visual Geometry Group (VGG) na Universidade de Oxford, esta arquitetura possui 19 camadas, incluindo convoluções 3x3 em todas as camadas. Sua estrutura uniforme facilita a compreensão e o treinamento. Apesar de mais profunda que suas variantes anteriores, a VGG19 é conhecida por sua habilidade em extrair características complexas de imagens.

## InceptionV3

A **InceptionV3**, também chamada de GoogLeNetV3, é uma arquitetura de CNN desenvolvida pelo Google. Sua característica distintiva é a abordagem inovadora usando módulos inception, que empregam convoluções de diferentes tamanhos simultaneamente. Isso possibilita a extração eficiente de características em várias escalas espaciais. Reconhecida pela eficácia em equilibrar desempenho e eficiência computacional, a InceptionV3 é amplamente utilizada em tarefas de classificação de imagens e reconhecimento visual em larga escala.

                
obs: Parte do conteudo descrito nessa página foi construido utilizando a **documentacao** do **Keras**, tensorflow e de ferramentas como: **Perplexity, google Bard e ChatGPT3.5-turbo;**
''')
elif pagina == 'Conjunto de dados':
    explicacao_dados(st)
elif pagina == 'VGG19':
    explicacao_vgg19(st)
elif pagina == 'InceptionV3':
    explicacao_inceptionV3(st)
