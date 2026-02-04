import ee
import os
import sys
import collections
collections.Callable = collections.abc.Callable
from pathlib import Path
pathparent = str(Path(os.getcwd()).parents[0])
sys.path.append(pathparent)
from configure_account_projects_ee import get_current_account, get_project_from_account
projAccount = get_current_account()
print(f"projetos selecionado >>> {projAccount} <<<")

try:
    ee.Initialize( project= projAccount)
    print('The Earth Engine package initialized successfully!')
except ee.EEException as e:
    print('The Earth Engine package failed to initialize!')
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise

def GetPolygonsfromFolder(assetFolder, sufixo):
  
    getlistPtos = ee.data.listAssets(assetFolder)
    asset_list = ee.Dictionary(getlistPtos).get('assets').getInfo()
    lstBacias = []
    for cc, idAsset in enumerate(asset_list): 
        path_ = idAsset.get('id') 
        name = path_.split("/")[-1]
        
        if sufixo in str(name): 
            print("eliminando {}:  {}".format(cc, name))
            print(path_)
            # ee.data.deleteAsset(path_) 
    
    print(lstBacias)

# asset = 'projects/mapbiomas-caatinga-cloud04/assets/Alertas/featCol_samples/setembro'
asset = 'projects/mapbiomas-caatinga-cloud04/assets/Alertas/featCol_samples/setembro_points'
GetPolygonsfromFolder(asset, '')  # 

