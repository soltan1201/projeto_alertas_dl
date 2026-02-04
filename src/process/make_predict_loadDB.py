


import os
import sys
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras import backend as K

# --- OTIMIZAÇÃO DE MEMÓRIA (Mixed Precision) ---
# Isso faz o modelo rodar em float16, economizando muita VRAM
tf.keras.mixed_precision.set_global_policy('mixed_float16')

pathparent = str('/home/superuser/Dados/projAlertas/proj_alertas_DL/src/process')
sys.path.append(pathparent)
# --- IMPORTAÇÃO DO MÓDULO DO MODELO ---
# Certifique-se que o arquivo .py está na mesma pasta ou no PYTHONPATH
from building_model_Siam_Swin_Unet import build_siam_swin_unet

# ================= CONFIGURAÇÕES =================
# Caminho onde você baixou os arquivos do Drive
# Dica: No Colab, monte o drive e use o caminho '/content/drive/MyDrive/DATASET_CHANGE_DETECTION_S2/*.tfrecord'
DATA_PATHS = '/run/media/superuser/Almacen/imgDB/tfr_alerts/setembro'  # /*.tfrecord.gz
print(" we load tfrecords from : \n >>> ", DATA_PATHS)

BATCH_SIZE = 4          # Ajuste conforme VRAM da sua GPU (16 ou 32 para imagens 256x256)
BUFFER_SIZE = 1000       # Quantas imagens manter na RAM para embaralhar (Shuffle)
AUTOTUNE = tf.data.AUTOTUNE

# Definição das Bandas (Mesma ordem do script GEE)
BANDS_LIST = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'ndfia', 'ratio_brown', 'savi']
NUM_BANDS = len(BANDS_LIST)
IMG_SHAPE = (256, 256)
# Tamanhos
RAW_PATCH_SIZE = 257  # O tamanho real que está vindo do GEE (128*2 + 1)
PATCH_SIZE = 256
# Specify the size and shape of patches expected by the model.
KERNEL_SIZE  = 256
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
RESPONSE = 'label'
# Lista completa de chaves no TFRecord
# T0 + T1 + Label
FEATURES_KEYS = [f"{b}_t0" for b in BANDS_LIST] + \
                [f"{b}_t1" for b in BANDS_LIST] + \
                ['label']

# Definição do Dicionário de Leitura (CORREÇÃO DO ERRO DE TIPO)
# Lemos tudo como float32 para evitar o erro "expected int64", depois convertemos
FEATURES_DICT = {
    key: tf.io.VarLenFeature(tf.float32) 
    for key in FEATURES_KEYS
}

# ================= 1. CONSTRUÇÃO DO DICIONÁRIO DE FEATURES =================
# Precisamos dizer ao TF como ler os bytes brutos. 
# Geramos isso dinamicamente para não ter que escrever 15 linhas de código repetidas.

# It's good practice to have this helper function separate
def squeeze_mask(image, mask):
  """Ensures the mask has the shape [H, W] and not [H, W, 1]."""
  with tf.device('/gpu:0'):  
      mask = tf.squeeze(mask, axis=-1)
      mask = tf.cast(mask, tf.uint8)
      return image, mask

def augment_spatial(image, label):
    """Randomly translates/pads the image."""
    # This function expects a 2D label, so we don't need to squeeze inside here anymore.
    with tf.device('/gpu:0'):
    # We add a temporary channel dimension for padding, then remove it.
        label = label[..., tf.newaxis] # Temporarily add back the channel: (256, 256, 1)
    
        padded_image = tf.pad(image, [[64, 64], [64, 64], [0, 0]], mode='CONSTANT')
        padded_label = tf.pad(label, [[64, 64], [64, 64], [0, 0]], mode='CONSTANT')
    
        random_x = tf.random.uniform([], 0, 129, dtype=tf.int32)
        random_y = tf.random.uniform([], 0, 129, dtype=tf.int32)
    
        cropped_image = tf.slice(padded_image, [random_x, random_y, 0], [256, 256, 6])
        cropped_label = tf.slice(padded_label, [random_x, random_y, 0], [256, 256, 1])
        
        # Squeeze the label again after cropping to return a 2D tensor
        cropped_label = tf.squeeze(cropped_label, axis=-1)
        
        return cropped_image, cropped_label

def removeNan(image,label):
    image = tf.keras.ops.nan_to_num(image)
    label = tf.keras.ops.nan_to_num(label)
    return image,label

def parse_tfrecord(example_proto):
    """The parsing function.
    Read a serialized example into the structure defined by FEATURES_DICT.
    Args:
        example_proto: a serialized Example.
    Returns: 
        A dictionary of tensors, keyed by feature name.
    """
    # 1. Lê os dados brutos como SparseTensors (lista de valores variados)
    parsed_sparse = tf.io.parse_single_example(example_proto, FEATURES_DICT)
    
    parsed_dense = {}
    
    # 2. Converte de Sparse para Dense e Reshapa
    for key in FEATURES_KEYS:
        # Pega a lista plana de valores
        flat_tensor = tf.sparse.to_dense(parsed_sparse[key], default_value=0.0)
        
        # FORÇA O RESHAPE MANUALMENTE
        # Isso resolve o erro "slice not parsed". Nós garantimos a forma aqui.
        try:
            reshaped = tf.reshape(flat_tensor, [RAW_PATCH_SIZE, RAW_PATCH_SIZE])
        except:
            # Fallback de segurança: se o patch vier vazio ou corrompido, cria zeros
            reshaped = tf.zeros([RAW_PATCH_SIZE, RAW_PATCH_SIZE], dtype=tf.float32)

        # 3. CROP para 256x256
        # Cortamos 1 pixel da direita e de baixo para ficar 256x256
        cropped = reshaped[:PATCH_SIZE, :PATCH_SIZE]            
        parsed_dense[key] = cropped
        
    return parsed_dense


def to_siamese_tuple(inputs):
    """
    Converte o dicionário para a estrutura Siamesa: ((T0, T1), Label)
    """
    # 1. Recupera as listas de tensores na ordem correta
    t0_list = [inputs.get(f"{b}_t0") for b in BANDS_LIST]
    t1_list = [inputs.get(f"{b}_t1") for b in BANDS_LIST]
    label = inputs.get("label")

    # 2. Empilha as bandas para formar imagens (H, W, C)
    img_t0 = tf.stack(t0_list, axis=-1) # (256, 256, 9)
    img_t1 = tf.stack(t1_list, axis=-1) # (256, 256, 9)
    
    # 3. Trata o Label
    # Label vem como (256, 256). Adicionamos dimensão de canal -> (256, 256, 1)
    label = tf.expand_dims(label, axis=-1)
    
    # Casting de segurança (Label para 0 ou 1 inteiro, Imagens normalizadas se necessário)
    # Assumindo que os dados já vieram normalizados do GEE ou são int16 brutos.
    # Se forem int16 brutos (0-10000), divida por 10000.0 aqui.
    img_t0 = tf.cast(img_t0, tf.float32) / 10000.0 
    img_t1 = tf.cast(img_t1, tf.float32) / 10000.0
    
    # Label deve ser int ou float 0.0/1.0. Vamos garantir float para sigmoid/dice
    label = tf.cast(label, tf.float32) 

    return (img_t0, img_t1), label

# ================= 2. AUGMENTATION =================

def augment_data(inputs, label):
    """Augmentation Sincronizada (Flip/Rotate)"""
    (t0, t1) = inputs
    combined = tf.concat([t0, t1, label], axis=-1)
    
    combined = tf.image.random_flip_left_right(combined)
    combined = tf.image.random_flip_up_down(combined)
    
    num_bands = len(BANDS_LIST)
    t0_aug = combined[:, :, :num_bands]
    t1_aug = combined[:, :, num_bands:2*num_bands]
    label_aug = combined[:, :, 2*num_bands:]
    
    return (t0_aug, t1_aug), label_aug

# ================= FUNÇÕES DE PROCESSAMENTO =================

def removeNan(inputs, label):
    """Remove NaNs de T0, T1 e Label"""
    (t0, t1) = inputs
    t0 = tf.where(tf.math.is_nan(t0), tf.zeros_like(t0), t0)
    t1 = tf.where(tf.math.is_nan(t1), tf.zeros_like(t1), t1)
    label = tf.where(tf.math.is_nan(label), tf.zeros_like(label), label)
    return (t0, t1), label

def augment_siamese(inputs, label):
    """
    Aplica Augmentation espacial (Flip/Rotate/Crop) em T0, T1 e Label SIMULTANEAMENTE.
    Isso garante que as imagens continuem alinhadas.
    """
    (img_t0, img_t1) = inputs
    
    # 1. Concatena tudo na profundidade para aplicar a mesma transformação geométrica
    # Shape resultante: (256, 256, C_t0 + C_t1 + C_label)
    combined = tf.concat([img_t0, img_t1, label], axis=-1)
    
    # --- A. Random Flip ---
    combined = tf.image.random_flip_left_right(combined)
    combined = tf.image.random_flip_up_down(combined)
    
    # --- B. Random Crop / Pad (Sua lógica adaptada) ---
    # Adiciona padding para poder cortar depois
    padded = tf.image.resize_with_crop_or_pad(combined, 256 + 64, 256 + 64)
    
    # Random Crop de volta para 256x256
    cropped = tf.image.random_crop(padded, size=[256, 256, combined.shape[-1]])
    
    # 2. Separa de volta
    # img_t0 são as primeiras N bandas
    img_t0_aug = cropped[:, :, :NUM_BANDS]
    # img_t1 são as próximas N bandas
    img_t1_aug = cropped[:, :, NUM_BANDS:2*NUM_BANDS]
    # label é a última banda
    label_aug = cropped[:, :, 2*NUM_BANDS:]
    
    return (img_t0_aug, img_t1_aug), label_aug


def get_dataset(pattern):
    """Function to read, parse and format to tuple a set of input tfrecord files.
    Get all the files matching the pattern, parse and convert to tuple.
    Args:
        pattern: A file pattern to match in a Cloud Storage bucket.
    Returns: 
        A tf.data.Dataset
    """
    files = tf.io.gfile.glob(pattern)
    
    # Se não achar arquivos, avisa (importante para debug)
    if not files:
        raise ValueError(f"Nenhum arquivo encontrado em: {pattern}")
        
    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP', num_parallel_reads=tf.data.AUTOTUNE)
    
    # 1. # Parsing Robusto
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 2. Converte para estrutura ((T0, T1), Label)
    dataset = dataset.map(to_siamese_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset


def get_training_dataset(data_pattern):
    dataset = get_dataset(data_pattern)
    
    # Remove NaNs antes de qualquer coisa
    dataset = dataset.map(removeNan, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Pipeline de Augmentation
    # Cache pode ser usado se couber na RAM, senão remova
    # dataset = dataset.cache() 
    
    # Shuffle deve ser grande o suficiente
    dataset = dataset.shuffle(buffer_size=150) 
    
    # Aplica Augmentation
    # Nota: Não precisamos mais do 'squeeze_mask' separado pois tratamos isso no to_siamese_tuple
    augmented_dataset = dataset.map(augment_siamese, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch e Prefetch
    final_dataset = augmented_dataset.batch(BATCH_SIZE)
    final_dataset = final_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return final_dataset


def build_feature_description():
    features = {}
    
    # Features para T0 e T1
    for band in BANDS_LIST:
        
        features[f'{band}_t0'] = tf.io.FixedLenFeature(IMG_SHAPE, tf.int64)
        features[f'{band}_t1'] = tf.io.FixedLenFeature(IMG_SHAPE, tf.int64)
    
    # Feature do Label (Máscara de Mudança)
    # CORREÇÃO: Leia como float32
    features['label'] = tf.io.FixedLenFeature(IMG_SHAPE, tf.float32)
    
    return features

FEATURE_DESCRIPTION = build_feature_description()

# ================= 2. FUNÇÃO DE PARSING (ETL) =================
# Esta função roda em paralelo na CPU para preparar os dados para a GPU



# ================= 3. PIPELINE DE DADOS (DATASET API) =================

def get_dataset_from_folder(folder_path, is_training=True):
    """
    Cria o dataset usando interleave a partir de uma lista de arquivos já dividida.
    """
    # Padrão de busca na pasta específica
    pattern = os.path.join(folder_path, '*.tfrecord.gz')    
    
    # 1. Cria um Dataset de NOMES DE ARQUIVOS
    # 1. Listar arquivos (High Performance)
    dataset = tf.data.Dataset.list_files(pattern, shuffle=is_training)

    
    # 3. Leitura Paralela (Interleave) - A SUA PREFERÊNCIA
    # Abre múltiplos arquivos simultaneamente e mistura seus registros
    dataset = dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(filename, compression_type='GZIP'),
        cycle_length=AUTOTUNE, # Quantos arquivos abrir ao mesmo tempo
        num_parallel_calls=AUTOTUNE, # Paralelismo de leitura
        deterministic=not is_training # Não determinístico no treino (mais rápido)
    )
    
    # 4. Parsing e Mapeamento (Paralelo)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(to_siamese_tuple, num_parallel_calls=AUTOTUNE)
    
    # 5. Processamento Específico de Treino
    if is_training:
        dataset = dataset.shuffle(buffer_size=200) # Shuffle de amostras (buffer de memória)
        dataset = dataset.map(augment_data, num_parallel_calls=AUTOTUNE)
        dataset = dataset.repeat() # Repete infinito
    
    # 6. Batch e Prefetch
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset

def visualize_patchs(img_t0_batch, img_t1_batch, label_batch):
    # Visualização RGB (Bandas B4, B3, B2 são índices 2, 1, 0 na lista BANDS)
    # BANDS = ['B2', 'B3', 'B4', ...] -> Indices: 0, 1, 2
    rgb_indices = [2, 1, 0] 
    
    plt.figure(figsize=(12, 4))
    
    # T0 RGB
    plt.subplot(1, 3, 1)
    # Multiplica por ganho de brilho (ex: 3x) para visualização, pois raw é escuro
    rgb_t0 = tf.gather(img_t0_batch[0], rgb_indices, axis=-1)
    plt.imshow(tf.clip_by_value(rgb_t0 * 3, 0, 1)) 
    plt.title("T0 (Antes) - RGB")
    plt.axis('off')

    # T1 RGB
    plt.subplot(1, 3, 2)
    rgb_t1 = tf.gather(img_t1_batch[0], rgb_indices, axis=-1)
    plt.imshow(tf.clip_by_value(rgb_t1 * 3, 0, 1))
    plt.title("T1 (Depois) - RGB")
    plt.axis('off')

    # Label (Mascara)
    plt.subplot(1, 3, 3)
    plt.imshow(label_batch[0, :, :, 0], cmap='gray')
    plt.title("Ground Truth (Alerta)")
    plt.axis('off')
    
    plt.show()

# --- 1. Dice Loss (Para combater desequilíbrio de classes) ---
def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Cálculo do Coeficiente Dice (F1-Score diferenciável).
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """
    Loss baseada no Dice (1 - Dice).
    Excelente para segmentação binária desbalanceada.
    """
    return 1 - dice_coef(y_true, y_pred)

# --- 2. Contrastive Loss (Estilo DASNet) ---
def contrastive_loss(y_true, dist, margin=2.0):
    """
    Contrastive Loss aplicada pixel a pixel.
    Nota: Esta loss espera que 'dist' seja uma distância (ex: distância Euclidiana)
    e não uma probabilidade. Se usar na saída Sigmoid, adapte para:
    y_pred = probabilidade de mudança (distância normalizada).
    """
    # y_true: 0 = inalterado, 1 = mudado
    # dist: saída do modelo (0 a 1)
    
    # Parte inalterada (queremos dist -> 0)
    loss_unchanged = (1 - y_true) * K.square(dist)
    
    # Parte mudada (queremos dist > margin, ou dist -> 1 se normalizado)
    # Usando max(margin - dist, 0) é o padrão, mas para sigmoid (0-1), 
    # podemos simplificar incentivando dist a ser grande.
    loss_changed = y_true * K.square(K.maximum(margin - dist, 0))
    
    return K.mean(loss_unchanged + loss_changed)

# --- 3. Hybrid Loss (Recomendado para Siam-Swin-Unet) ---
def bce_dice_loss(y_true, y_pred):
    """
    Combina Binary Cross Entropy (estabilidade) + Dice Loss (precisão em bordas).
    """
    loss_bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    loss_dice = dice_loss(y_true, y_pred)
    return loss_bce + loss_dice

# ================= 4. TESTE E VISUALIZAÇÃO =================

# Verificar se existem arquivos
train_ds = get_dataset_from_folder(os.path.join(DATA_PATHS, 'train'), is_training=True)
val_ds   = get_dataset_from_folder(os.path.join(DATA_PATHS, 'val'),   is_training=False)
test_ds  = get_dataset_from_folder(os.path.join(DATA_PATHS, 'test'),  is_training=False)

show_patchs = False
    # sys.exit()
# Pega um batch para visualizar
if show_patchs: 
    for (img_t0_batch, img_t1_batch), label_batch in train_ds.take(1):
        print(f"Shape T0: {img_t0_batch.shape}") # (Batch, 256, 256, 7)
        print(f"Shape T1: {img_t1_batch.shape}")
        print(f"Shape Label: {label_batch.shape}")
        visualize_patchs(img_t0_batch, img_t1_batch, label_batch)
        break


# 1. Defina o shape baseado no seu print (256, 256, 9)
IMG_HEIGHT = 256
IMG_WIDTH = 256
NUM_CHANNELS = 9  # B2, B3, B4, B8, B11, B12, ndfia, ratio, savi
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)


# ==========================================
# INSTANCIANDO O MODELO
# ==========================================

# Chama a função importada
model = build_siam_swin_unet(input_shape=INPUT_SHAPE)

# Compila
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=bce_dice_loss, 
    metrics=['accuracy', dice_coef]
)

model.summary()

# Estimativa: Quantos patches existem em cada arquivo TFRecord?
# No passo de exportação, você definiu MAX_PATCHES_PER_FILE = 100.
# Vamos usar isso para estimar.
PATCHES_PER_SHARD = 100 

# Conte quantos arquivos existem nas pastas (Isso é rápido)
num_train_files = len(os.listdir(os.path.join(DATA_PATHS, 'train')))
num_val_files   = len(os.listdir(os.path.join(DATA_PATHS, 'val')))

print(f"Arquivos de Treino: {num_train_files}")
print(f"Arquivos de Validação: {num_val_files}")

# Calcula os steps
STEPS_PER_EPOCH = (num_train_files * PATCHES_PER_SHARD) // BATCH_SIZE
VALIDATION_STEPS = (num_val_files * PATCHES_PER_SHARD) // BATCH_SIZE

print(f"Treinando com {STEPS_PER_EPOCH} passos por época.")

# ==========================================
# CALLBACKS (Salvar o melhor modelo)
# ==========================================
callbacks_list = [
    # Salva o modelo apenas se o Dice Score na validação melhorar
    tf.keras.callbacks.ModelCheckpoint(
        filepath="best_siam_swin_unet.keras", # Ou .h5
        monitor="val_dice_coef",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    # Reduz o Learning Rate se o modelo parar de aprender
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-6
    ),
    # Para o treino se não houver melhoria por 10 épocas
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )
]

# ==========================================
# EXECUTANDO O TREINO
# ==========================================

EPOCHS= 50
with tf.device('/gpu:0'):
    history = model.fit(
        train_ds,
        epochs= EPOCHS, # Ajuste conforme necessário
        steps_per_epoch=STEPS_PER_EPOCH,
        
        validation_data=val_ds,
        validation_steps=VALIDATION_STEPS,
        
        callbacks=callbacks_list,
        verbose=1
    )

    print("Treinamento finalizado.")

    # --- 1. Definição dos Caminhos ---
    data_saved = '21_01_2026'
    path_base = "/content/drive/MyDrive/DL_alertas/src"
    path_model_folder = os.path.join(path_base, "model") # Pasta para os .keras
    path_saveModel = os.path.join(path_model_folder, f'model_{data_saved}.keras')

    # --- 2. Salvando o Modelo ---

    # (MELHORIA 1) Garante que a pasta de destino exista antes de salvar
    os.makedirs(path_model_folder, exist_ok=True)

    print(f"Salvando modelo em: {path_saveModel}...")
    history.save(
          path_saveModel,
          overwrite=True,
          include_optimizer=True
    )
    print("✅ Modelo salvo.")

    # --- 3. Salvando o Histórico ---
    name_csv_history = f'valores_histogram4L_{data_saved}.csv'
    path_saveHistory = os.path.join(path_base, name_csv_history) # Salva na pasta base

    # (MELHORIA 2.1) Cria o DataFrame diretamente do histórico.
    # Isso captura automaticamente TODAS as métricas (loss, acc, val_loss, etc.)
    dfVal = pd.DataFrame(history.history)

    # (Opcional, mas recomendado) Adiciona a coluna de época
    dfVal['epoch'] = dfVal.index + 1

    print(f"Salvando histórico em: {path_saveHistory}...")
    # (MELHORIA 2.2) Adiciona index=False para um CSV mais limpo
    dfVal.to_csv(path_saveHistory, index=False)
    print("✅ Histórico salvo.")