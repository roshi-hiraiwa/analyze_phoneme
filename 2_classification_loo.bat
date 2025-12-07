@echo off

PAUSE

rem 仮想環境（anaconda）の有効化、keras_env_looは事前に用意した仮想環境
cd .\src
call C:\Users\hogehoge\anaconda3\Scripts\activate.bat keras_env_loo

python make_cnn_classification_loo.py
rem python svm_loo.py

PAUSE






