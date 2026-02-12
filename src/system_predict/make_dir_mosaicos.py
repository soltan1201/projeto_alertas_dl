import os

path_base = "/home/superusuario/db_images/rasters_alerts"

lst_folders =  [
    'PATCHS_S2_Setembro_CAAT', 'PATCHS_S2_Outubro_CAAT',  
    'PATCHS_S2_Novembro_CAAT', 'PATCHS_S2_Novembro_Caat',
    'PATCHS_S2_Dezembro_CAAT', 'PATCHS_S2_Dezembro_Caat'
]

for npath in lst_folders:
    print("  ---------------------------------------------------------------")
    path_end = os.path.join(path_base, npath)

    # Cria a pasta e todas as pastas pai (subdiretórios) se não existirem
    # Se já existir, o exist_ok=True impede que o Python lance um erro
    os.makedirs(path_end, exist_ok=True)