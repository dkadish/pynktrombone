# PynkTrombone
The implementation of human vocal organ model. Ported from voc.pdf written in C to Python.   
This repository was created to make dkadish's Python implementation of PinkTrombone faster.  
The functions and modules are about 10x faster with no-python jit.  

<img src=data/trombone.gif width="350px">

---
## Usage
1. Clone this repository and run the following code in this folder to install it.
    ```python
    pip install -e ./
    ```

2. We can import Vocal organ interface (Voc) by following code. 
    ```python 
    from pynkTrombone import Voc
    ```

3. Construct Voc 
    ```python
    voc = Voc(Arguments)
    ```  
    main arguments:  
    - samplerate  
    This is the resolution at which it is generated; at 22050 Hz, high-frequency components such as consonants are reduced.  

    - CHUNK  
    The length of the waveform to be generated in one step.

    - default_freq  
    The initial value of frequency of glottis.

    - others  
    please refer voc.py, tract.py and glottis.py  .

4. Modify voc  
    Modify voc using ```voc.tongue_shape(args)``` and etc.
    See demos.py for other items that can be changed.
    

5. Let's generate!  
    In the following code, we will change the tongue_diameter and tenseness of the voc in update_fn to generate the voice. there are some examples in demos.py
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
## Structure of vocal organ.
Please refer voc.pdf about details.   
<img src=data/vocal_tract.png width="320px">