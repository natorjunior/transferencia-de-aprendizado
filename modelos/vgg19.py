import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import seaborn as sns
from tensorflow.keras.models import load_model
import random


# Função para inicializar o modelo se não existir no estado da sessão
def inicializar_modelo(st):
    if 'modelo_vgg19' not in st.session_state:
        st.session_state.modelo_vgg19 = VGG19(weights='imagenet')
    if 'caminhos_completos' not in st.session_state:
        caminhos_pneumonia = [os.path.join('img/test/PNEUMONIA', arquivo) for arquivo in os.listdir('img/test/PNEUMONIA')]
        caminhos_normal = [os.path.join('img/test/NORMAL', arquivo) for arquivo in os.listdir('img/test/NORMAL')]
        # Combinar as duas listas de caminhos
        caminhos_completos = caminhos_pneumonia + caminhos_normal
        random.shuffle(caminhos_completos)
        st.session_state.caminhos_completos = caminhos_completos

# Função para obter a saída da camada de convolução escolhida
def obter_saida_camada(modelo, camada, img_array):
    modelo_extracao = tf.keras.Model(inputs=modelo.input, outputs=camada.output)
    return modelo_extracao.predict(img_array)



def explicacao_vgg19(st):
    st.title('VGG19')
    radio_vgg = st.radio('',['Arquitetura','rede passo a passo','código','app'],horizontal=True)
    inicializar_modelo(st)
    # Carregar o modelo VGG19 pré-treinado
    modelo_vgg19 = st.session_state.modelo_vgg19
    if radio_vgg == 'Arquitetura':
        st.image('img/img2.png')
    elif radio_vgg == 'rede passo a passo':
        caminho_da_imagem = st.selectbox('PNEUMONIA',os.listdir('img/test/PNEUMONIA'))
        caminho_da_imagem = 'img/test/PNEUMONIA/'+caminho_da_imagem
        img = image.load_img(caminho_da_imagem, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Obter a lista de camadas disponíveis no modelo VGG19
        camadas_disponiveis = [camada.name for camada in modelo_vgg19.layers if 'conv' in camada.name]
        col1,col2 = st.columns(2)
        with col1:
            # Interface do Streamlit para seleção de camadas
            camada_selecionada = st.selectbox("Selecione a Camada de Convolução", camadas_disponiveis)
        # Obter a camada de convolução selecionada
        camada = modelo_vgg19.get_layer(camada_selecionada)
        # Obter a saída da camada de convolução escolhida
        saida_camada = obter_saida_camada(modelo_vgg19, camada, img_array)
        with col2:
            canal = st.selectbox("Selecione o canal", list(range(saida_camada.shape[3])))

        st.write(saida_camada.shape)

        col11,col22 = st.columns(2)
        with col11:
            # Redimensionar a exibição da imagem
            plt.figure(figsize=(3, 5))

            # Exibir a saída como uma imagem usando matplotlib
            plt.imshow(img, cmap='viridis')
            plt.title('Imagem original')
            plt.axis('off')

            # Exibir a figura usando Streamlit
            st.pyplot(plt)

        with col22:
            # Redimensionar a exibição da imagem
            plt.figure(figsize=(5, 6))

            # Exibir a saída como uma imagem usando matplotlib
            plt.imshow(saida_camada[0, :, :, canal], cmap='viridis')
            plt.title('Saída da Camada de Convolução Selecionada')
            plt.axis('off')

            # Exibir a figura usando Streamlit
            st.pyplot(plt)
    elif radio_vgg == 'código':
        st.write('---')
        radio_vgg_cod = st.radio('',['load','camada densa','compilado','treinando','código completo','resultados'],horizontal=True)
        if radio_vgg_cod == 'load':
            st.code('''from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
import numpy as np

num_classes = 2
#VGG19 pré-treinado
#VGG19 sem as camadas totalmente conectadas
base_model = VGG19(weights='imagenet', include_top=False)
        ''')
        elif radio_vgg_cod == 'camada densa':
            st.code('''#camadas totalmente conectadas para a nova tarefa de classificação
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
        ''')
        elif radio_vgg_cod == 'compilado':
            st.code('''# Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False
#Novo modelo combinando a base pré-treinada com as camadas personalizadas
model = Model(inputs=base_model.input, outputs=predictions)
# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        ''')
        elif radio_vgg_cod == 'treinando':
            st.code('''# Tranformando os rótulos em codificação one-hot
y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

# Treinando o modelo
model.fit(x_train, y_train_one_hot, epochs=10, batch_size=32, validation_data=(x_test, y_test_one_hot))

# Medindo o desempenho do modelo nos dados de teste
accuracy = model.evaluate(x_test, y_test_one_hot)[1]
print(f'Acurácia nos dados de teste: {accuracy * 100:.2f}%')
        ''')
        elif radio_vgg_cod == 'código completo':
            sss = st.selectbox('selecione:',['Pre-processamento de imagens','VGG19'])
            if sss == 'Pre-processamento de imagens':
                st.markdown('''## Imagens ''')
                st.code('''import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
path_test_NORMAL = '/content/drive/MyDrive/dados_pneumonia/test/NORMAL/'
path_test_PNEUMONIA = '/content/drive/MyDrive/dados_pneumonia/test/PNEUMONIA/'

# Lista de todas as imagens
list_img_normal = os.listdir(path_test_NORMAL)
list_img_pneumonia = os.listdir(path_test_PNEUMONIA)

x_data = []
y_data = []

# Lendo as imagens que tem normal
for img_name in list_img_normal:
    img_path = os.path.join(path_test_NORMAL, img_name)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    x_data.append(img_array)
    y_data.append(0)  # Rótulo 0 para NORMAL

# Lendo as imagens que tem pneumonia
for img_name in list_img_pneumonia:
    img_path = os.path.join(path_test_PNEUMONIA, img_name)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    x_data.append(img_array)
    y_data.append(1)  # Rótulo 1 para PNEUMONIA

x_data = np.array(x_data)
y_data = np.array(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

print("Shape of x_train:", x_train.shape)
print("Shape of x_test:", x_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

        ''')
            if sss == 'VGG19':
                st.markdown('''## Código''')
                st.code('''import numpy as np
import os
from tensorflow.keras.applications import VGG19
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.utils import to_categorical
import numpy as np
num_classes = 2
#VGG19 pré-treinado
#VGG19 sem as camadas totalmente conectadas
base_model = VGG19(weights='imagenet', include_top=False)

#camadas totalmente conectadas para a nova tarefa de classificação
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

#Novo modelo combinando a base pré-treinada com as camadas personalizadas
model = Model(inputs=base_model.input, outputs=predictions)

# Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    
# Tranformando os rótulos em codificação one-hot
y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

# Treinando o modelo
model.fit(x_train, y_train_one_hot, epochs=10, batch_size=32, validation_data=(x_test, y_test_one_hot))

# Medindo o desempenho do modelo nos dados de teste
accuracy = model.evaluate(x_test, y_test_one_hot)[1]
print(f'Acurácia nos dados de teste: {accuracy * 100:.2f}%')
        ''')
        elif radio_vgg_cod == 'resultados':
            
            # Calcular métricas
            accuracy = 0.92
            precision = 0.93
            recall = 0.93
            f1 = 0.93
            conf_matrix = [[47 , 5],
            [ 5, 68]]
            #st.write(conf_matrix)

            # Exibir métricas
            st.subheader('Métricas de Avaliação:')
            st.write(f'Acurácia: {accuracy * 100:.2f}%')
            st.write(f'Precisão: {precision:.2f}')
            st.write(f'Recall: {recall:.2f}')
            st.write(f'F1 Score: {f1:.2f}')

            # Exibir a matriz de confusão como uma imagem
            st.subheader('Matriz de Confusão:')
            plt.figure(figsize=(3, 3))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
            plt.xlabel('Predito')
            plt.ylabel('Real')
            st.pyplot(plt)

            # Fechar a figura para liberar recursos
            plt.close()
    elif radio_vgg == 'app':
        model = load_model('modelos/treinados/modelo_1_VGG19_10_epocas_92cc.h5', compile=False)
        # Recriar o modelo com otimizador customizado sem 'weight_decay'
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        caminhos_completos = st.session_state.caminhos_completos
        caminho_da_imagem = st.selectbox('PNEUMONIA+NORMAL',caminhos_completos)
        caminho_da_imagem = caminho_da_imagem
        img = image.load_img(caminho_da_imagem, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        y_pred = model.predict(np.array([img_array[0]]))
        y_pred_class = np.argmax(y_pred)
        valores = ['É uma imagem que NORMAL','É uma imagem que contém Pneumonia']
        st.write(y_pred_class)
        if y_pred_class == 1:
            st.error(valores[1], icon="🚨")#img_array[0].shape)
        elif y_pred_class == 0:
            st.success(valores[0], icon="✅")#img_array[0].shape)




    st.write('---')