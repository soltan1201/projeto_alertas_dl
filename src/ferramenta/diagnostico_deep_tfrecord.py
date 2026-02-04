import tensorflow as tf
import glob

# Ajuste o caminho para UM arquivo tfrecord seu
FILE_PATH = '/run/media/superuser/Almacen/imgDB/tfr_alerts/setembro/tfrecord_219_71_part018.tfrecord.gz' # (Exemplo, pegue um real)

# Pega o primeiro arquivo que encontrar se o caminho for wildcard
if '*' in FILE_PATH:
    FILE_PATH = glob.glob(FILE_PATH)[0]

print(f"Inspecionando: {FILE_PATH}")

raw_dataset = tf.data.TFRecordDataset(FILE_PATH, compression_type='GZIP')

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    
    # Lista as chaves e tipos
    keys = list(example.features.feature.keys())
    print("\n--- CHAVES ENCONTRADAS ---")
    for k in sorted(keys):
        feat = example.features.feature[k]
        kind = feat.WhichOneof('kind')
        
        # Verifica tamanho
        if kind == 'float_list':
            size = len(feat.float_list.value)
            sample = feat.float_list.value[:3]
        elif kind == 'int64_list':
            size = len(feat.int64_list.value)
            sample = feat.int64_list.value[:3]
        else:
            size = 0
            sample = 'Bytes'
            
        print(f"Key: {k:15} | Tipo: {kind:12} | Tamanho: {size} (Esperado 65536) | Sample: {sample}")