# gains_ese3600
This github repository contains code for our ESE 3600 Tiny ML project, titled Gains, which is a wearable device to help the user classify their push up posture, in hopes to improve their performance.

We have uploaded our python notebook that trains and build the model onto Google Colab for convenience.  

Final Report: https://docs.google.com/document/d/1kXohgcP6mb6MmehOqjFfpL7uKsxaT0XsXnimC4CWX-E/edit?usp=sharing

Demo Video: https://www.youtube.com/watch?v=sGeO_O7ViBE

Google Colab: https://colab.research.egoogle.com/drive/1_bb615_qASqIWBoNGF4A2Udvsltr4_Ne?usp=sharingI 

Easy commands to remember:
- To flash: ```pio run -t upload```

To collect training data:
1. Flash the script in data_collection folder
2. Activate a venv
3. Run ```python pushup_data_collector.py```, which starts up a GUI
4. Collect data, the raw data will be placed in raw_dataset folder

To process the data(applying low/high pass buffers, normalization, data augmentation)
1. Follow the steps in the data_analysis.ipynb

To run inference model:
1. Run the model python notebook in google colab (use GPU model)
2. Copy and paste the binary file created from the notebook into pushup_model_data.cpp
3. Flash the script in gains_ese3600

To actually conduct inference:
1. Press the button on the hardware, or type r in the serial monitor, to start a recording session  
2. Pause a bit and then do 1 pushup rep
3. Once reach top (ie the end of the pushup), press the button or type r in the serial monitor to tend the recording session
4. Look at serial monitor for final classification  
