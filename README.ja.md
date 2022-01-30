# PynkTrombone
これは人間の発声器官モデルの実装コードです。voc.pdfのC言語で書かれたものをPythonに移植しました。   
このリポジトリはdkadishさんのPinkTromboneのPython実装をより高速にするために作りました。  
各種関数やモジュールはno-python jit によっておよそ10倍に高速化されています。  
  
<img src=data/trombone.gif width="350px">

---
## 使い方
1. このリポジトリをクローンし、このフォルダ内で次のコードを実行しインストールしてください
    ```python
    pip install -e ./
    ```

2. 次のコードで発声器官(Voc)をimportします  
    ```python 
    from pynkTrombone import Voc
    ```

3. Vocをコンストラクトします。  
    ```python
    voc = Voc(Arguments)
    ```  
    main arguments:  
    - samplerate  
    生成する時の解像度です。22050Hzになると子音などの高周波成分が小さくなります。  

    - CHUNK  
    1 stepで生成する波形の長さです。

    - default_freq  
    声門の周波数の初期値です。

    - others  
    そのほかはvoc.py, tract.py, glottis.pyを参照してください。

4. vocを変更します。  
    ```voc.tongue_shape(args)```などを使って声道や声門を変更します。  
    そのほかの変更可能項目はdemos.pyを参照ください。
    

5. 波形を生成しましょう！  
    次のコードで登場するupdate_fnの中でvocのtongue_diameterやtensenessを変化させ、声を生成して行きましょう！demos.pyの中にいくつか例があります。
    ```python
    def play_update(update_fn, filename):
    sr: float = 44100
    voc = Voc(sr)
    x = 0
    voc = update_fn(voc, x)

    out = voc.play_chunk() # modify voc
    while out.shape[0] < sr * 5:
        out = np.concatenate([out, voc.play_chunk()])

        x += 1
        voc = update_fn(voc, x) # modify voc

    sf.write(filename, out, sr)
    ```
---
## 声道の構造
詳細はvoc.pdfをご覧ください。  
<img src=data/vocal_tract.png width="320px">