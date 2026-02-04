import os
import glob
import random
import shutil
import math

# ================= CONFIGURAÇÕES =================
# Diretório onde estão os TFRecords misturados
SOURCE_DIR = '/run/media/superuser/Almacen/imgDB/tfr_alerts/setembro/'

# Proporções (Devem somar 1.0)
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.2
TEST_RATIO  = 0.2

# Seed para garantir que o sorteio seja sempre o mesmo (Reprodutibilidade)
RANDOM_SEED = 42

# Extensão dos arquivos
FILE_EXTENSION = '*.tfrecord.gz'

def split_and_move_files():
    # 1. Verifica se o diretório existe
    if not os.path.exists(SOURCE_DIR):
        print(f"ERRO: Diretório não encontrado: {SOURCE_DIR}")
        return

    # 2. Lista todos os arquivos
    search_pattern = os.path.join(SOURCE_DIR, FILE_EXTENSION)
    all_files = glob.glob(search_pattern)
    total_files = len(all_files)

    if total_files == 0:
        print(f"Nenhum arquivo {FILE_EXTENSION} encontrado em {SOURCE_DIR}")
        return

    print(f"Total de arquivos encontrados: {total_files}")

    # 3. Embaralha a lista
    random.seed(RANDOM_SEED)
    random.shuffle(all_files)

    # 4. Calcula os índices de corte
    train_count = int(total_files * TRAIN_RATIO)
    val_count   = int(total_files * VAL_RATIO)
    # O restante vai para teste para evitar problemas de arredondamento
    test_count  = total_files - train_count - val_count

    print(f"Divisão planejada -> Train: {train_count}, Val: {val_count}, Test: {test_count}")

    # 5. Define os grupos
    train_files = all_files[:train_count]
    val_files   = all_files[train_count : train_count + val_count]
    test_files  = all_files[train_count + val_count :]

    # 6. Função auxiliar para mover
    def move_group(files, folder_name):
        target_dir = os.path.join(SOURCE_DIR, folder_name)
        
        # Cria a pasta se não existir
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            print(f"Pasta criada: {target_dir}")
        
        print(f"Movendo {len(files)} arquivos para '{folder_name}'...")
        
        for file_path in files:
            file_name = os.path.basename(file_path)
            new_path = os.path.join(target_dir, file_name)
            
            try:
                shutil.move(file_path, new_path)
            except Exception as e:
                print(f"Erro ao mover {file_name}: {e}")

    # 7. Executa a movimentação
    move_group(train_files, 'train')
    move_group(val_files, 'val')
    move_group(test_files, 'test')

    print("\n========================================")
    print("Processo concluído com sucesso!")
    print(f"Verifique as pastas em: {SOURCE_DIR}")
    print("========================================")

if __name__ == "__main__":
    # Confirmação de segurança
    print(f"ATENÇÃO: Este script vai MOVER arquivos em: {SOURCE_DIR}")
    resp = input("Deseja continuar? (S/N): ")
    if resp.lower() == 's':
        split_and_move_files()
    else:
        print("Cancelado.")