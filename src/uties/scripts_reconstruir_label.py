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
    # Use FEATURES_DICT (which now has VarLenFeatures) for parsing
    parsed_sparse = tf.io.parse_single_example(example_proto, FEATURES_DICT)
    parsed_dense = {}

    for key in FEATURES_KEYS:
        # Get the dtype from FEATURES_DICT for the current key
        feature_dtype = FEATURES_DICT[key].dtype

        # Set default_value based on the feature_dtype
        if feature_dtype == tf.float32:
            default_val = 0.0
        elif feature_dtype == tf.int64:
            default_val = 0
        else:
            # Fallback for other types, or raise an error
            default_val = 0 # Default to int zero, might need adjustment for other types

        # Convert SparseTensor to DenseTensor
        flat_tensor = tf.sparse.to_dense(parsed_sparse[key], default_value=default_val)

        try:
            # Reshape to PATCH_SIZE x PATCH_SIZE (256x256) as the rewritten TFRecords contain this size
            reshaped = tf.reshape(flat_tensor, [PATCH_SIZE, PATCH_SIZE])
        except:
            # Fallback for empty or corrupted patches, create a zero tensor of expected shape and dtype
            reshaped = tf.zeros([PATCH_SIZE, PATCH_SIZE], dtype=feature_dtype)

        # 3. CROP to 256x256
        # This step is redundant if reshaped is already PATCH_SIZE x PATCH_SIZE
        parsed_dense[key] = reshaped

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

# Verificar se existem arquivos
substituir_destino = True
if substituir_destino:
    DATA_PATHS = "/content/drive/MyDrive/DL_alertas/src/db_tfrecord"

train_ds = get_dataset_from_folder(os.path.join(DATA_PATHS, 'train'), is_training=True)
val_ds   = get_dataset_from_folder(os.path.join(DATA_PATHS, 'val'),   is_training=False)
# test_ds  = get_dataset_from_folder(os.path.join(DATA_PATHS, 'test'),  is_training=False)
show_patchs = True

# Pega um batch para visualizar
if show_patchs:
    for (img_t0_batch, img_t1_batch), label_batch in train_ds.take(1):
        print(f"Shape T0: {img_t0_batch.shape}") # (Batch, 256, 256, 7)
        print(f"Shape T1: {img_t1_batch.shape}")
        print(f"Shape Label: {label_batch.shape}")

        for j in range(0, 6):
          numb_aleatorio = random.randint(0, BATCH_SIZE - 1)
          print(f"Número aleatório: {numb_aleatorio}")
          visualize_patchs(img_t0_batch, img_t1_batch, label_batch, numb_aleatorio)

        break


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
    # Use FEATURES_DICT (which now has VarLenFeatures) for parsing
    parsed_sparse = tf.io.parse_single_example(example_proto, FEATURES_DICT)
    parsed_dense = {}

    for key in FEATURES_KEYS:
        # Get the dtype from FEATURES_DICT for the current key
        feature_dtype = FEATURES_DICT[key].dtype

        # Set default_value based on the feature_dtype
        if feature_dtype == tf.float32:
            default_val = 0.0
        elif feature_dtype == tf.int64:
            default_val = 0
        else:
            # Fallback for other types, or raise an error
            default_val = 0 # Default to int zero, might need adjustment for other types

        # Convert SparseTensor to DenseTensor
        flat_tensor = tf.sparse.to_dense(parsed_sparse[key], default_value=default_val)

        try:
            # Reshape to PATCH_SIZE x PATCH_SIZE (256x256) as the rewritten TFRecords contain this size
            reshaped = tf.reshape(flat_tensor, [PATCH_SIZE, PATCH_SIZE])
        except:
            # Fallback for empty or corrupted patches, create a zero tensor of expected shape and dtype
            reshaped = tf.zeros([PATCH_SIZE, PATCH_SIZE], dtype=feature_dtype)

        # 3. CROP to 256x256
        # This step is redundant if reshaped is already PATCH_SIZE x PATCH_SIZE
        parsed_dense[key] = reshaped

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



# Verificar se existem arquivos
substituir_destino = True
if substituir_destino:
    DATA_PATHS = "/content/drive/MyDrive/DL_alertas/src/db_tfrecord"

train_ds = get_dataset_from_folder(os.path.join(DATA_PATHS, 'train'), is_training=True)
val_ds   = get_dataset_from_folder(os.path.join(DATA_PATHS, 'val'),   is_training=False)
# test_ds  = get_dataset_from_folder(os.path.join(DATA_PATHS, 'test'),  is_training=False)
show_patchs = True



# Pega um batch para visualizar
if show_patchs:
    for (img_t0_batch, img_t1_batch), label_batch in train_ds.take(1):
        print(f"Shape T0: {img_t0_batch.shape}") # (Batch, 256, 256, 7)
        print(f"Shape T1: {img_t1_batch.shape}")
        print(f"Shape Label: {label_batch.shape}")

        for j in range(0, 6):
          numb_aleatorio = random.randint(0, BATCH_SIZE - 1)
          print(f"Número aleatório: {numb_aleatorio}")
          visualize_patchs(img_t0_batch, img_t1_batch, label_batch, numb_aleatorio)

        break


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


# Diretório onde estão os TFRecords misturados
# SOURCE_DIR = '/run/media/superuser/Almacen/imgDB/tfr_alerts/setembro/'
DATA_PATHS = '/content/drive/MyDrive/DATASET_CHANGE_DETECTION_S2_TFRECORDS'
print(" we load tfrecords from : \n >>> ", DATA_PATHS)


BATCH_SIZE = 12          # Ajuste conforme VRAM da sua GPU (16 ou 32 para imagens 256x256)
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
FEATURES_DICT = {}
for key in FEATURES_KEYS:
    if 'label' in key:
        FEATURES_DICT[key] = tf.io.VarLenFeature(tf.float32)
    else:
        FEATURES_DICT[key] = tf.io.VarLenFeature(tf.int64)



# ==========================================
# 2. FUNÇÕES AUXILIARES DE TOPOLOGIA
# ==========================================

def patch_embed(x, embed_dim, patch_size=4):
    x = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding='same')(x)
    x = layers.LayerNormalization()(x)
    return x

def patch_merging(x, filters):
    x = layers.Conv2D(filters, kernel_size=2, strides=2, padding='same')(x)
    return x

def patch_expanding(x, filters):
    x = layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding='same')(x)
    return x

def encoder_body(input_tensor, filter_list):
    skips = []
    # Stage 1
    x = patch_embed(input_tensor, filter_list[0])
    x = SwinBlock(filter_list[0], num_heads=3)(x)
    skips.append(x)
    # Stage 2
    x = patch_merging(x, filter_list[1])
    x = SwinBlock(filter_list[1], num_heads=6)(x)
    skips.append(x)
    # Stage 3
    x = patch_merging(x, filter_list[2])
    x = SwinBlock(filter_list[2], num_heads=12)(x)
    skips.append(x)
    # Stage 4 (Bottleneck)
    x = patch_merging(x, filter_list[3])
    x = SwinBlock(filter_list[3], num_heads=24)(x)
    return x, skips

def decoder_body(encoded_feature, skips_t1, skips_t2, filter_list):
    x = encoded_feature
    filters = filter_list[::-1]

    # Stage 1
    x = patch_expanding(x, filters[1])
    s1 = skips_t1[-1]
    s2 = skips_t2[-1]
    diff = layers.Subtract()([s1, s2])
    x = layers.Concatenate()([x, diff])
    x = layers.Conv2D(filters[1], 3, padding='same', activation='gelu')(x)
    x = SwinBlock(filters[1], num_heads=12)(x)

    # Stage 2
    x = patch_expanding(x, filters[2])
    s1 = skips_t1[-2]
    s2 = skips_t2[-2]
    diff = layers.Subtract()([s1, s2])
    x = layers.Concatenate()([x, diff])
    x = layers.Conv2D(filters[2], 3, padding='same', activation='gelu')(x)
    x = SwinBlock(filters[2], num_heads=6)(x)

    # Stage 3
    x = patch_expanding(x, filters[3])
    s1 = skips_t1[-3]
    s2 = skips_t2[-3]
    diff = layers.Subtract()([s1, s2])
    x = layers.Concatenate()([x, diff])
    x = layers.Conv2D(filters[3], 3, padding='same', activation='gelu')(x)
    x = SwinBlock(filters[3], num_heads=3)(x)

    return x

def build_feature_description():
    features = {}

    # Features para T0 e T1
    for band in BANDS_LIST:

        features[f'{band}_t0'] = tf.io.FixedLenFeature([RAW_PATCH_SIZE, RAW_PATCH_SIZE], tf.int64)
        features[f'{band}_t1'] = tf.io.FixedLenFeature([RAW_PATCH_SIZE, RAW_PATCH_SIZE], tf.int64)

    # Feature do Label (Máscara de Mudança)
    # CORREÇÃO: Leia como float32
    features['label'] = tf.io.FixedLenFeature([RAW_PATCH_SIZE, RAW_PATCH_SIZE], tf.float32)

    return features

FEATURE_DESCRIPTION = build_feature_description()
print("DEBUG: FEATURE_DESCRIPTION contents after creation:")
for key, val in FEATURE_DESCRIPTION.items():
    print(f"  {key}: {val.dtype}, {val.shape}")


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
    # Use FEATURE_DESCRIPTION which now correctly expects (RAW_PATCH_SIZE, RAW_PATCH_SIZE)
    parsed_features = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)
    parsed_dense = {}

    for key in FEATURES_KEYS:
        tensor = parsed_features[key]

        # DIAGNOSTIC: Print the expected dtype from FEATURE_DESCRIPTION
        if key == 'B11_t1':
            print(f"DEBUG: parse_tfrecord: Key {key}, Expected dtype: {FEATURE_DESCRIPTION[key].dtype}")

        # Since FixedLenFeature already provides dense tensors, no sparse_to_dense needed.
        # Also, the shape from FixedLenFeature is already correct (RAW_PATCH_SIZE, RAW_PATCH_SIZE).

        # 3. CROP to 256x256
        # Crop 1 pixel from the right and bottom to get 256x256
        cropped = tensor[:PATCH_SIZE, :PATCH_SIZE]
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

# Verificar se existem arquivos
substituir_destino = True
if substituir_destino:
    DATA_PATHS = "/content/drive/MyDrive/DL_alertas/src/db_tfrecord"

train_ds = get_dataset_from_folder(os.path.join(DATA_PATHS, 'train'), is_training=True)
val_ds   = get_dataset_from_folder(os.path.join(DATA_PATHS, 'val'),   is_training=False)
# test_ds  = get_dataset_from_folder(os.path.join(DATA_PATHS, 'test'),  is_training=False)
show_patchs = True



# Pega um batch para visualizar
if show_patchs:
    for (img_t0_batch, img_t1_batch), label_batch in train_ds.take(1):
        print(f"Shape T0: {img_t0_batch.shape}") # (Batch, 256, 256, 7)
        print(f"Shape T1: {img_t1_batch.shape}")
        print(f"Shape Label: {label_batch.shape}")

        for j in range(0, 6):
          numb_aleatorio = random.randint(0, BATCH_SIZE - 1)
          print(f"Número aleatório: {numb_aleatorio}")
          visualize_patchs(img_t0_batch, img_t1_batch, label_batch, numb_aleatorio)

        break


name_csv_history = f'valores_histogram4L_{data_saved}.csv'
path_saveHistory = os.path.join(path_base, name_csv_history) # Salva na pasta base

# (MELHORIA 2.1) Cria o DataFrame diretamente do histórico.
# Isso captura automaticamente TODAS as métricas (loss, acc, val_loss, etc.)
dfVal = pd.DataFrame(result.history)

# (Opcional, mas recomendado) Adiciona a coluna de época
dfVal['epoch'] = dfVal.index + 1

print(f"Salvando histórico em: {path_saveHistory}...")
# (MELHORIA 2.2) Adiciona index=False para um CSV mais limpo
dfVal.to_csv(path_saveHistory, index=False)
print("✅ Histórico salvo.")