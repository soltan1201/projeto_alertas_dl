import os
import sys
import glob 
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import rasterio
from rasterio.transform import from_origin
# --- OTIMIZAÇÃO DE MEMÓRIA (Mixed Precision) ---
# Isso faz o modelo rodar em float16, economizando muita VRAM
tf.keras.mixed_precision.set_global_policy('mixed_float16')
from tensorflow import keras
from tensorflow.keras import layers, models, Input
register_serializable = keras.utils.register_keras_serializable( )

from tensorflow.keras import backend as K

# Detecta a versão do TF e usa o decorador correto
tf_version = tf.__version__
print(f"TensorFlow version: {tf_version}")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif v.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# --- CONFIGURAÇÕES ---
show_patchs = False
PATCH_SIZE = 256
SCALE = 10 # Para Landsat
BATCH_SIZE = 4          # Ajuste conforme VRAM da sua GPU (16 ou 32 para imagens 256x256)
BUFFER_SIZE = 1000       # Quantas imagens manter na RAM para embaralhar (Shuffle)
AUTOTUNE = tf.data.AUTOTUNE

# Definição das Bandas (Mesma ordem do script GEE)
BANDS_LIST = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'ndfia', 'ratio_brown', 'savi']
NUM_BANDS = len(BANDS_LIST)
IMG_SHAPE = (256, 256)
# Tamanhos
RAW_PATCH_SIZE = 257  # O tamanho real que está vindo do GEE (128*2 + 1) - Corrected from 257 to 256
TARGET_PATCH_SIZE = 256 # O tamanho que seu modelo quer

# Specify the size and shape of patches expected by the model.
KERNEL_SIZE  = 256
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
RESPONSE = 'label'

# Lista completa de chaves no TFRecord
# T0 + T1 + Label
FEATURES_KEYS = [f"{b}_t0" for b in BANDS_LIST] + \
                [f"{b}_t1" for b in BANDS_LIST] + \
                ['label']

parser = argparse.ArgumentParser()
parser.add_argument('PATH_BASE_INPUT_TFR', type=str,  default= True, help= "Especifica qual será o caminho base dos tfrecord" )
parser.add_argument('PATH_BASE_OUTPUT', type=str, default= False, help= 'Define qual será o caminho de destino do dado de predict')
parser.add_argument('inServer', type=str, default= False, help= 'Rodando desde o servidor ?')
args = parser.parse_args()

PATH_TFRECORDS = args.PATH_BASE_INPUT_TFR
PATH_OUTPUT = args.PATH_BASE_OUTPUT
isInServidor = str2bool(args.inServer)

# PATH_TFRECORDS = "/home/superusuario/db_images"
# PATH_TFRECORDS = "/home/superuser/db_images"
# PATH_OUTPUT = "/home/superuser/db_images/predicts"
LISTA_FOLDERS = [
    'PATCHS_S2_Setembro_CAAT', 'PATCHS_S2_Outubro_CAAT',  
    'PATCHS_S2_Novembro_CAAT', 'PATCHS_S2_Novembro_Caat',
    'PATCHS_S2_Dezembro_CAAT', 'PATCHS_S2_Dezembro_Caat'
]


# Dicionário de leitura (Deve incluir a geometria exportada pelo GEE)
# Quando geometries=True, o GEE exporta o ponto como uma string serializada ou float lat/lon
# Dicionário agora mapeia LAT e LON como floats diretos (FixedLenFeature)
FEATURES_DICT_PREDICT = {
    **{f"{b}_t0": tf.io.VarLenFeature(tf.float32) for b in BANDS_LIST},
    **{f"{b}_t1": tf.io.VarLenFeature(tf.float32) for b in BANDS_LIST},
    'label': tf.io.VarLenFeature(tf.float32),
    'latitude': tf.io.FixedLenFeature([], tf.float32),
    'longitude': tf.io.FixedLenFeature([], tf.float32)
}




def parse_tfrecord(example_proto, filename):
    """The parsing function.
    Read a serialized example into the structure defined by FEATURES_DICT_PREDICT.
    Args:
        example_proto: a serialized Example.
    Returns:
        A dictionary of tensors, keyed by feature name.
    """
    # Use FEATURES_DICT (which now has VarLenFeatures) for parsing
    parsed_sparse = tf.io.parse_single_example(example_proto, FEATURES_DICT_PREDICT)
    parsed_dense = {}
    
    # Tamanho esperado: 257 * 257 = 66049
    EXPECTED_FLAT_SIZE = RAW_PATCH_SIZE * RAW_PATCH_SIZE
    # 2. Converte de Sparse para Dense e Reshapa
    for key in FEATURES_KEYS:
        flat_tensor = tf.sparse.to_dense(parsed_sparse[key], default_value=0.0)

        # VERIFICAÇÃO DE SEGURANÇA:
        # tf.size retorna o número total de elementos
        data_size = tf.size(flat_tensor)

        # Usamos tf.cond para decidir se fazemos o reshape ou retornamos zeros
        reshaped = tf.cond(
            tf.equal(data_size, EXPECTED_FLAT_SIZE),
            lambda: tf.reshape(flat_tensor, [RAW_PATCH_SIZE, RAW_PATCH_SIZE]),
            lambda: tf.zeros([RAW_PATCH_SIZE, RAW_PATCH_SIZE], dtype=tf.float32)
        )

        # Crop para 256x256
        parsed_dense[key] = reshaped[:256, :256]

    # 2. Coordenadas (Agora vêm direto como número!)
    parsed_dense['latitude'] = parsed_sparse['latitude']
    parsed_dense['longitude'] = parsed_sparse['longitude']

    # Extrai apenas o nome do arquivo (sem o caminho completo e sem .gz)
    fname_clean = tf.strings.split(filename, os.sep)[-1]
    fname_clean = tf.strings.regex_replace(fname_clean, r'\.tfrecord\.gz', '')
    parsed_dense['filename'] = fname_clean

    return parsed_dense



def to_siamese_tuple(inputs):
    """
        Converte o dicionário para a estrutura Siamesa: ((T0, T1), Label)
    """
    # 1. Recupera as listas de tensores na ordem correta
    t0_list = [inputs.get(f"{b}_t0") for b in BANDS_LIST]
    t1_list = [inputs.get(f"{b}_t1") for b in BANDS_LIST]    

    # 2. Empilha as bandas para formar imagens (H, W, C)
    # Casting de segurança (Label para 0 ou 1 inteiro, Imagens normalizadas se necessário)
    # Assumindo que os dados já vieram normalizados do GEE ou são int16 brutos.
    # Se forem int16 brutos (0-10000), divida por 10000.0 aqui.
    img_t0 = tf.cast(tf.stack(t0_list, axis=-1), tf.float32) / 10000.0 # (256, 256, 9)
    img_t1 = tf.cast(tf.stack(t1_list, axis=-1), tf.float32) / 10000.0 # (256, 256, 9)

    # 3. Trata o Label
    # Label vem como (256, 256). Adicionamos dimensão de canal -> (256, 256, 1)
    label = tf.expand_dims(inputs.get("label"), axis=-1)  
    # Label deve ser int ou float 0.0/1.0. Vamos garantir float para sigmoid/dice
    label = tf.cast(label, tf.float32)

    # 3. Parte de Geometria (SOLUÇÃO DO ERRO)
    # Não usamos mais Regex! Lemos os campos que você definiu no FEATURES_DICT_PREDICT
    lat = inputs.get('latitude')
    lon = inputs.get('longitude')
    fname = inputs.get('filename') # Captura o nome

    # Retorno: ((T0, T1), (Label, Lat, Lon, NomeArquivo))
    return (img_t0, img_t1), (label, lat, lon, fname)

def get_dataset_from_folder(folder_path, is_training=True):
    """
    Cria o dataset usando interleave a partir de uma lista de arquivos já dividida.
    """
    # Padrão de busca na pasta específica
    pattern = os.path.join(folder_path, '*.tfrecord.gz')

    # 1. Cria um Dataset de NOMES DE ARQUIVOS
    # 1. Listar arquivos (High Performance)
    dataset = tf.data.Dataset.list_files(pattern, shuffle=is_training)

    # Função interna para ler o arquivo e injetar o nome
    def fetch_records_with_filename(filename):
        ds = tf.data.TFRecordDataset(filename, compression_type='GZIP')
        # Retorna um par: (registro_bruto, nome_do_arquivo)
        return ds.map(lambda record: (record, filename))

    # 3. Leitura Paralela (Interleave) - A SUA PREFERÊNCIA
    # Abre múltiplos arquivos simultaneamente e mistura seus registros
    dataset = dataset.interleave(
        # lambda filename: tf.data.TFRecordDataset(filename, compression_type='GZIP'),
        fetch_records_with_filename,
        cycle_length=AUTOTUNE, # Quantos arquivos abrir ao mesmo tempo
        num_parallel_calls=AUTOTUNE, # Paralelismo de leitura
        deterministic=not is_training # Não determinístico no treino (mais rápido)
    )

    # 4. Parsing e Mapeamento (Paralelo)
    # dataset = dataset.map(parse_tfrecord, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(lambda record, fname: parse_tfrecord(record, fname), num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(to_siamese_tuple, num_parallel_calls=AUTOTUNE)

    # 6. Batch e Prefetch
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset

def visualize_patchs(img_t0_batch, img_t1_batch, label_batch, irandom):
    # Visualização RGB (Bandas B4, B3, B2 são índices 2, 1, 0 na lista BANDS)
    # BANDS = ['B2', 'B3', 'B4', ...] -> Indices: 0, 1, 2
    rgb_indices = [2, 1, 0]

    plt.figure(figsize=(12, 4))

    # T0 RGB
    plt.subplot(1, 3, 1)
    # Multiplica por ganho de brilho (ex: 3x) para visualização, pois raw é escuro
    rgb_t0 = tf.gather(img_t0_batch[irandom], rgb_indices, axis=-1)
    plt.imshow(tf.clip_by_value(rgb_t0 * 3, 0, 1))
    plt.title("T0 (Antes) - RGB")
    plt.axis('off')

    # T1 RGB
    plt.subplot(1, 3, 2)
    rgb_t1 = tf.gather(img_t1_batch[irandom], rgb_indices, axis=-1)
    plt.imshow(tf.clip_by_value(rgb_t1 * 3, 0, 1))
    plt.title("T1 (Depois) - RGB")
    plt.axis('off')

    # Label (Mascara)
    plt.subplot(1, 3, 3)
    plt.imshow(label_batch[irandom, :, :, 0], cmap='gray')
    plt.title("Ground Truth (Alerta)")
    plt.axis('off')

    plt.show()

def mostrar_raster_builded(file_path):
    repository_ds = get_dataset_from_folder(file_path, is_training= False)
    print("========= carregou o dataset para a variavel repository_ds ====== ")

    for (img_t0_batch, img_t1_batch), (label_batch, lat, lon) in repository_ds.take(1):
        print(f"Shape T0   : {img_t0_batch.shape}") # (Batch, 256, 256, 7)
        print(f"Shape T1   : {img_t1_batch.shape}")
        print(f"Shape Label: {label_batch.shape}")

        for j in range(0, 6):
            numb_aleatorio = random.randint(0, BATCH_SIZE - 1)
            print(f"Número aleatório: {numb_aleatorio}")
            visualize_patchs(img_t0_batch, img_t1_batch, label_batch, numb_aleatorio)

        break


# ==========================================
# 1. BLOCOS CUSTOMIZADOS (Com serialização)
# ==========================================
@keras.utils.register_keras_serializable(package='Custom')
class SwinBlock(layers.Layer):
    """
    Bloco simplificado do Swin Transformer.
    """
    def __init__(self, embed_dim, num_heads, window_size=7, **kwargs):
        super(SwinBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = models.Sequential([
            layers.Dense(embed_dim * 4, activation='gelu'),
            layers.Dense(embed_dim)
        ])

    def build(self, input_shape):
        # build padrão para as normalizações e MLP
        self.ln1.build(input_shape)
        self.ln2.build(input_shape)
        self.mlp.build(input_shape)

        # build do MultiHeadAttention de forma segura:
        # Ele espera (query_shape, value_shape, key_shape)
        self.attn.build(input_shape, input_shape, input_shape)

        super(SwinBlock, self).build(input_shape)

    def call(self, x):
        # Atenção
        x = tf.cast(x, self.compute_dtype)
        res = x
        x = self.ln1(x)
        # Atenção: query, value, key
        x = self.attn(x, x, x)
        x = res + x

        # MLP
        res = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = res + x
        return x

    # def compute_output_shape(self, input_shape):
    #     return input_shape

    # --- MÉTODO CRUCIAL PARA SALVAR/CARREGAR O MODELO ---
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
        })
        return config

# --- 1. Dice Loss (Para combater desequilíbrio de classes) ---
@keras.utils.register_keras_serializable(package="Custom")
def dice_coef(y_true, y_pred, smooth=1e-7):
    """
    Cálculo do Coeficiente Dice (F1-Score diferenciável).
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

@keras.utils.register_keras_serializable(package="Custom")
def dice_loss(y_true, y_pred):
    """
    Loss baseada no Dice (1 - Dice).
    Excelente para segmentação binária desbalanceada.
    """
    return 1 - dice_coef(y_true, y_pred)

# --- 3. Hybrid Loss (Recomendado para Siam-Swin-Unet) ---
@keras.utils.register_keras_serializable(package="Custom")
def bce_dice_loss(y_true, y_pred, bce_weight=0.95, dice_weight=0.95):
    """
    Combina Binary Cross Entropy (estabilidade) + Dice Loss (precisão em bordas).
    """
    loss_bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    loss_dice = dice_loss(y_true, y_pred)
    return loss_bce + loss_dice


def save_patch_as_geotiff(prediction, lon_c, lat_c, scale, filename):
    """
    Salva a matriz de predição (256x256) como um GeoTIFF georreferenciado.
    """
    # Resolução em graus: 10m ~ 0.00008983 | 30m ~ 0.0002695
    res_deg = scale * 0.000008983 
    
    # O ponto lat/lon do GEE é o CENTRO do patch. 
    # Precisamos do canto superior esquerdo (Upper Left) para o rasterio.
    half_patch_deg = (TARGET_PATCH_SIZE / 2) * res_deg
    ul_lon = lon_c - half_patch_deg
    ul_lat = lat_c + half_patch_deg

    transform = from_origin(ul_lon, ul_lat, res_deg, res_deg)

    with rasterio.open(
        filename, 'w',
        driver='GTiff',
        height=TARGET_PATCH_SIZE,
        width=TARGET_PATCH_SIZE,
        count=1,
        dtype='float32',
        crs='EPSG:4326',
        transform=transform,
        compress='lzw' # Compactação LZW para economizar espaço no SSD do Arch
    ) as dst:
        dst.write(prediction.astype(np.float32), 1)



# --- 1. CARREGAMENTO DO MODELO ---
# Substitua pelo caminho real do seu arquivo .h5 ou SavedModel
if isInServidor:
    MODEL_PATH = "/home/superusuario/projetos/model_Siam_Swin_Unet/best_siam_swin_unet_ciclo2.keras"
else:
    MODEL_PATH = "/home/superuser/Dados/projAlertas/proj_alertas_DL/src/model/best_siam_swin_unet_ciclo2.keras"
print(f"🤖 Carregando modelo em: {MODEL_PATH}")
custom_objects = {
    'SwinBlock': SwinBlock,
    # Adicione outras camadas customizadas aqui
    'bce_dice_loss': bce_dice_loss,  # se tiver função de loss customizada
    'dice_coef': dice_coef,  # se tiver métrica customizada
}
# 5. Carregue o modelo com os objetos customizados
try:
    model_siam = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects=custom_objects
    )
    model_siam.summary()
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")

# --- 2. LOOP DE PREDIÇÃO E ESCRITA ---
for name_folder in LISTA_FOLDERS[:]:
    pattern = os.path.join(PATH_TFRECORDS, name_folder, "*.tfrecord.gz")
    print(f"🔍 Buscando em: {pattern}")

    # 1. Teste com glob do Python (mais fácil de ler)
    files = glob.glob(pattern)
    print(f"📂 Arquivos encontrados pelo Python: {len(files)}")
    if len(files) > 0:
        print(f"📄 Primeiro arquivo: {files[0]}")
    else:
        print("❌ ERRO: Nenhum arquivo encontrado. Verifique se a pasta existe e se o Samba/Rclone montou os dados corretamente.")
        # Lista o que existe na pasta pai para ajudar no debug
        parent = os.path.dirname(PATH_TFRECORDS)
        if os.path.exists(parent):
            print(f"📁 Pastas disponíveis em {parent}: {os.listdir(parent)}")
        sys.exit()
    print(f"============== READING FOLDER {name_folder} ON REPOSITORY =================== ")
    folder_dir = os.path.join(PATH_TFRECORDS, name_folder)
    
    if show_patchs:
        mostrar_raster_builded(folder_dir)
    # break
    predict_ds = get_dataset_from_folder(folder_dir, is_training= False)
    if not  os.path.exists(os.path.join(PATH_OUTPUT,name_folder)): 
        os.mkdir(os.path.join(PATH_OUTPUT,name_folder))
    count = 0
    for (img_t0_batch, img_t1_batch), (label_batch, lat_batch, lon_batch, name_batch) in predict_ds:
        # Fazer predição no BATCH (Usa o máximo da GPU RTX 2060)
        # O retorno costuma ser (Batch, 256, 256, 1)
        preds = model_siam.predict((img_t0_batch, img_t1_batch), verbose=0)
        
        # Iterar sobre os itens do batch para salvar individualmente
        for i in range(preds.shape[0]):
            lat_c = lat_batch[i].numpy()
            lon_c = lon_batch[i].numpy()

            # O nome vem como bytes do TensorFlow, decodificamos para string
            original_fname = name_batch[i].numpy().decode('utf-8')
            
            # Gerar nome de arquivo único usando coordenadas
            # tif_name = os.path.join(PATH_OUTPUT, name_folder ,f"pred_{count:05d}_{lat_c:.6f}_{lon_c:.6f}.tif")
            # Gerar o nome: [Nome_Original]_patch_[i].tif
            # Usamos o i para diferenciar patches que venham do mesmo shard
            tif_basename = f"{original_fname}_patch_{i}_{lat_c:.6f}_{lon_c:.6f}.tif"
            tif_full_path = os.path.join(PATH_OUTPUT, name_folder , tif_basename)
            # Remover a dimensão do canal (256, 256, 1) -> (256, 256)
            patch_pred = preds[i, :, :, 0]
            
            save_patch_as_geotiff(patch_pred, lon_c, lat_c, SCALE, tif_full_path)
            count += 1
            
        if count % 100 == 0:
            print(f"✅ {count} patches processados e salvos...")

print("\n✨ Processamento concluído com sucesso!")