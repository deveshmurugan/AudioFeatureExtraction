Extracting features from audio files for creating a data-set

MATLAB code for extracting time, frequency and cepstral domain features from N number of .wav audio files and creating a feature data-set. The features are,

Time Domain
• Energy of the segment
• Zero crossing Rate
• Maximum, Minimum Amplitude
• Periodicity for voiced audio segments.

Frequency Domain
• Fundamental frequency.
• Power spectrum
• Frequency of max amplitude
• Frequency of min amplitude
• Frequency standard deviation

Cepstral Domain
• Pitch
• Formant Frequencies (first 3 frequencies)
• Mel Frequency Co-efficients

Keep your audio files in a single folder with a common name followed by a number from 1 to N
(eg) chunk1.wav, chunk2.wav, chunk1000.wav

Update the folder path in the code as
folder = 'C:\Users\JohnDoe\Desktop\MySamples\chunk';
where MySamples is the folder which contains the samples.

Set N and NumberofTestingSamples based on your need.

Run the code to extract the mentioned features.

The train_features contains the training data-set and test_features matrix contains the testing data-set.
