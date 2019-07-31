# Chord-Scale_Detection
Repository for "Chroma Based Automatic Chord-Scale Detection for Monophonic Jazz Solos" task.

##### Chord-Scale Dataset
###### Artists - instrument :
Jamey Aebersold - tenor saxophone  
Toprak Barut - tenor saxophone  
Hikmet Altunbaslier - trumpet  

#### List of chord-scales annotated & taxonomy :

Major (major)  
Dorian (dorian)  
Phyrgian (phyrgian)  
Lydian (lydian)  
Mixolydian (mixolydian)  
Natural Minor (minor)  
Locrian (locrian)  
Lydian b7 (lydianb7)  
Super Locrian /  Altered (slocrian)  
Half-whole Step Symmetrical Diminished (hwdiminished)  
Melodic Minor (mminor)  
Harmonic Minor (hminor)  
Whole-tone*
Chromatic*

* : The last two scales are only included in the estimation class set as scale templates.


Installation
  ---------
  In order to use the tools and notebooks, you need to install 'docker' . Docker provides a virtual environment with all the desired dependencies and libraries already installed. In this toolbox for 'Chroma Based Automatic Chord-Scale Detection for Monophonic Jazz Solos', we have used the tools in 'MIR-Toolbox' which contains a set of tools (including *numpy,matplotlib, scikit-learn,'Essentia'* ) installed and compiled for several Music Information Retrieval applications. For more information regarding toolbox, please refer to https://github.com/MTG/MIR-toolbox-docker  :
  
   1) Install docker-compose
   Follow [instructions](https://docs.docker.com/compose/install/).

   **Windows**
    https://docs.docker.com/docker-for-windows/install/

   **Mac**
    https://docs.docker.com/docker-for-mac/install/

   **Ubuntu**
    https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-docker-ce


   2) Clone the repository into target directory.
   
    git clone https://github.com/emirdemirel/Supervised_Mode_Recognition.git
    
   3) On MacOS or Windows, run:

    docker-compose up

On Linux, run the following (this command ensures that any files you create are owned by your own user):

    JUPYTER_USER_ID=$(id -u) docker-compose up

Then accesss http://localhost:8888 with your browser and when asked for a
password use ***mir***
     
   4) Open the Jupyter notebook  'Chord-scale_Detection.ipynb' for the implementation of the work presented in paper [1].


   
   Authors
   -------------
   Emir Demirel
   emir.demirel@upf.edu



   [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-ff69b4.svg)](http://creativecommons.org/licenses/by-nc-sa/4.0/)
