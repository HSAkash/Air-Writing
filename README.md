# Air-Writing Using mediapipe

Estimate hand pose using mediapipe.<br>
This program recognizes the fingers.Art or write in the air according to the specified finger position.

This repository contains the following features:
* Hand recognition
* Finger lanmarks recognition
* Air-Writing
* Erase writing

# Requirements
* matplotlib==3.5.3
* mediapipe==0.8.10.1
* numpy==1.23.2
* opencv-python==4.6.0.66

# Demo
Here is how to run the air-writing program using the following command line.<br>
```bash
python airwriting.py
```

# Directories
<pre>
│  handTrackModule.py
│  airWriting.py
├─env
</pre>

### handTrackModule.py
This is the module that will be used to track the fingers in air.<br>
Functionalities:
* findHands (Find hands and track them)
* findPosition (Find finger lanmarks and box position)
* fingerUp (Find which  finger is currently up)
* thumsDown (Find is this thumsDown)
* findDistance (Find distance between two fingers)

### airWriting.py
This is the main module that will be used to write or draw in air.<br>
* draw in air.
* Erase it.

### Finger lanmarks
The key point coordinates are the ones that have undergone the following preprocessing up to ④.<br>
<img src="https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png" width="80%">


### How this program works.
* Index finger for writing.Writing mode works only when index finger up and other fingers are down.
* Index, Middle and Ring finger for erase mode.Erase mode works only when index, middle and ring finger are up and other are down.

