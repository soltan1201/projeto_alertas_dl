#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
#SCRIPT DE CLASSIFICACAO POR BACIA
#Produzido por Geodatin - Dados e Geoinformacao
#DISTRIBUIDO COM GPLv2
'''
import ee
import os 
# import gee
import sys
import argparse
import collections
collections.Callable = collections.abc.Callable
from pathlib import Path
pathparent = str(Path(os.getcwd()).parents[0])
sys.path.append(pathparent)
from configure_account_projects_ee import get_current_account, get_project_from_account
projAccount = get_current_account()
print(f"projetos selecionado >>> {projAccount} <<<")
from gee_tools import *

try:
    ee.Initialize(
        project= projAccount
    )
    print('The Earth Engine package initialized successfully!')
except ee.EEException as e:
    print('The Earth Engine package failed to initialize!')
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise


param = {
    'unicaconta': False,
    'cancelar' : False,    
    'numeroTask': 16,
    'numeroLimit': 6,
    'conta' : {
        '1': 'caatinga01',
        '2': 'caatinga02',
        '3': 'caatinga03',
        '4': 'caatinga04',
        '5': 'caatinga05',        
        '6': 'superconta',
        '7': 'solkanCengine'
    }
}
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif v.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def gerenciador(cont):   
    
    #=====================================
    # gerenciador de contas para controlar 
    # processos task no gee   
    #=====================================    
    
    numberofChange = [kk for kk in param['conta'].keys()]
    print(cont)
    # print(numberofChange)
    
    if str(cont) in numberofChange:        
        switch_user(param['conta'][str(cont)])
        projAccount = get_project_from_account(param['conta'][str(cont)])
        try:
            ee.Initialize(project= projAccount) # project='ee-cartassol'
            print('The Earth Engine package initialized successfully!')
        except ee.EEException as e:
            print('The Earth Engine package failed to initialize!') 
        # relatorios.write("Conta de: " + param['conta'][str(cont)] + '\n')
        print("acessing conta de " + param['conta'][str(cont)] + '\n')

        tarefas = tasks(
            n= param['numeroTask'],
            return_list= True,
            print_tasks= False)
        
        for cc, lin in enumerate(tarefas):            
            # relatorios.write(str(lin) + '\n')
            print(cc, lin)
    
    elif cont > param['numeroLimit']:
        return 0
    
    cont += 1    
    return cont

parser = argparse.ArgumentParser()
parser.add_argument('unicaconta', type=str,  default= True, help= "Especifica se va ser unica conta com True" )
parser.add_argument('cancelar', type=str, default= False, help= 'Define se as task serão canceladas , deefault = False')
args = parser.parse_args()
unicaConta= args.unicaconta
cancelFile = args.cancelar

unicaConta = str2bool(unicaConta)
cancelFile = str2bool(cancelFile)
print("unica conta = " , unicaConta)
print("cancelar = " , cancelFile)

param['unicaconta'] = unicaConta
param['cancelar'] = cancelFile
# sys.exit()
if param['unicaconta'] == True:
    contaSel = input("Digite um número entre 1 e 7 para escoler a conta, sendo 6 = superconta, 7 =  solkanCengine :_ ")
    cont = int(contaSel)
    cont = gerenciador(cont)
    if param['cancelar']:
        cancel(opentasks= cancelFile)
else:
    cont = 0
    for cont in range(1,6):        
        cont = gerenciador(cont)
        if param['cancelar']:
            cancel(opentasks= cancelFile)
