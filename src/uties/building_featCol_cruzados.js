var date_inic = '2025-09-01';
var name_month = 'setembro';
var name_folder = 'Setembro2025';

var asset_fcAlert = 'projects/mapbiomas-caatinga-cloud04/assets/Alertas/revisados_2025/alerts_pol_' + name_month + '_' + date_inic;
var asset_folderAlert = 'projects/mapbiomas-caatinga-cloud04/assets/Alertas/gerados_2025/' + name_folder;
var asset_output = 'projects/mapbiomas-caatinga-cloud04/assets/Alertas/featCol_samples/';
// 1. Carregar alertas aprovados
var featC_alerts = ee.FeatureCollection(asset_fcAlert)
                          .map(function(feat){return feat.set('aprovado', true)});

print("Feature Collection alerts ", featC_alerts.limit(4));
var size_alerts = featC_alerts.size();
print("Feature Collection alerts tamanho ", size_alerts);

var featPoints_alerts = ee.FeatureCollection(asset_fcAlert)
                            .map(function(feat){return feat.centroid()});
                            
featPoints_alerts = featPoints_alerts.geometry();
// 1. Listar todos os assets no folder
var assetList = ee.data.listAssets(asset_folderAlert, {'maxResults': 1000});
var number_grids = ee.List(assetList.assets).size();
print("numero de featCollection em folders  ", number_grids);

var num_coleta = ee.Number(size_alerts).multiply(100).divide(number_grids);
print("Numero de alertas a sere coletado por grids", num_coleta);

var assetIds = assetList.assets.map(function(asset) {return asset.id; });
print('Assets encontrados no folder:', assetIds);

// 2. Carregar todos os FeatureCollections do folder e combiná-los
var allFolderAlerts = ee.FeatureCollection([]);

// Combinar todos os FeatureCollections em um único
assetIds.forEach(function(assetId) {
    var fc = ee.FeatureCollection(assetId);
    print("tamanho original de featureCollection ", fc.size());
    // Filtra a pasta: mantém apenas o que NÃO intersecta a geometria dos aprovados
    var fc_remain = fc.filter(ee.Filter.bounds(featPoints_alerts).not());
    // print(fc_remain.size());
    fc_remain = fc_remain.map(function(feat){return feat.set('area', feat.area())});
    fc_remain = fc_remain.randomColumn({rowKeys: ['area']}); //distribution: 'normal', 
    fc_remain = fc_remain.filter(ee.Filter.lt('random', ee.Number(num_coleta).divide(ee.Number(fc.size()))));
    print(" rest feature Collection ", fc_remain.size());
    
    allFolderAlerts = allFolderAlerts.merge(fc_remain);
});

print("Total de alerts no folder (combinados):", allFolderAlerts.size());



// --- Saídas no Console ---
print("Total Aprovados:", featC_alerts.size());
// print("Total na Pasta Original:", mergedFolderFC.size());
// print("Total Restante (Para Revisão):", alertsRemaining.size());

// // --- Visualização ---
Map.addLayer(featC_alerts, {color: '6c1448'}, 'Aprovados (Verde)');
Map.addLayer(allFolderAlerts, {color: 'FF0000'}, 'Restantes (Vermelho)');
// Map.centerObject(featC_alerts, 10);


// Opcional: Exportar o resultado limpo
var name_export = 'alertas_' + name_folder;
Export.table.toAsset({
    collection: allFolderAlerts.merge(featC_alerts),
    description: name_export,
    assetId: asset_output + name_export
});


