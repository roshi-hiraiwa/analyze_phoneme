@echo off

PAUSE

rem 出力先ディレクトリの初期化など
rd /s /q target > nul 2>&1

rem MFCCの作成
cd ./src
matlab -wait -r make_mfcc_csv('../train_data.csv')
rem matlab -wait -r make_mfcc_csv('../test_data.csv')

PAUSE