# gains_ese3600
Classify pushup posture

Video + IMU
Posture + Type of Pushup

Google colab: https://colab.research.egoogle.com/drive/1_bb615_qASqIWBoNGF4A2Udvsltr4_Ne?usp=sharingI 
To flash:
```pio run -t upload```

To collect training data:
1. Flash the script in MW_DataCollection folder (must be done with home directory being MW_DataCollection in order for platformio.ini extension to be properly used)
2. Activate a venv
3. Run ```python pushup_data_collector.py```


To run inference model:
1. Run python notebook in google colab (use GPU)
2. Copy and paste the binary file into pushup_model_data.cpp
3. Flash the script in gains_ese3600 (must be done with home directory being gains_ese3600)


Steps  
1. Beeps  
2. Pause a bit and then do rep  
3. Once reach top, immediatly stop  
