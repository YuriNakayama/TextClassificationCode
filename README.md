# TextClassificationCode

AgTitleのテキスト分類は以下の手続で、ファイルを順次実行する。\
Uploadと以下のセルはデータをローカル環境で管理する場合は実行しなくてよい。\
なお、実行ファイルと同じディレクトリにShellScriptがある場合は、それをbashで実行することもできる。ただし、config/All.pyで指定されたパラメータで実行される。



1. DataShaping(データの形式を整える)
DataShaping/AgNews/OutputData.ipynb
2. Preprocessing
Preprocessing/AgNewsTitle/RemoveStopword.ipynb
3. Vectorize
Vectorize/SentenceBERT/SentenceBERT.py
Vectorize/SentenceBERT/DimensionDepression.py
4. Clustering
Clustering/Preprocessing/CentralizeNormalize.py
Clustering/GMM/Gmm.py
5. Postprocessing(指標の計算)
Postprocessing/GMM/Calcstats.py
Postprocessing/GMM/Coherence.py

