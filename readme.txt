DATA generation
I generate data in folder data_generation. If you don't want, you can go to folder deep_learning and download from dropbox(1gb or 15 gb).
If you want to generate, first we need to install Denise on your machine:

cd data_generation/DENISE-Black-Edition

0.cd data_generation/DENISE-Black-Edition


1. Compile the library cseife in /libcseife with

make 

2. In /src adapt the compiler options in the Makefile to your system and compile the DENISE code with

make denise

Generally for linux or Mac you need only to manually specify the location of your fftw3 library. E.g. in the makefile in src folder for my MacBook I use:
specifically I add this flag to IFLAGS 
-I/opt/homebrew/opt/fftw/include/

 
Then for data generation please use the notebook data_generation.ipynb. For plotting shots and their spectrums data_visualization.ipynb.


Machine learning

For deep learning I use the notebook ex2_training.ipynb . I use torch environment from yml file. You can use your own torch environment, if you experience problems with using of yml file.





P.S.
DENISE Manual available at:
https://danielkoehnsite.wordpress.com/software/
https://github.com/ovcharenkoo/DENISE-Black-Edition/blob/master/QUICKSTART_Marmousi.txt