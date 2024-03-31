Machine learning-aided multiplexed fluorescence detection for point-of-care applications

## Overview: We developed a compact, lens-free, and cost-effective fluorescence sensing setup that incorporates machine learning for scalable multiplexed fluorescence detection. 
This method utilizes low-cost optical components and a pretrained machine learning (ML) model to enable multiplexed fluorescence sensing without optical adjustments.

## Purpose: This repository houses all the datasets and Python codes used to generate results presented in sections 3.2 and 3.4 of the manuscrtipt. 

## Prerequisites
Before you can work with this project, you need to have Git installed on your computer.

# Installing Git
->Through Command Prompt

    1. Download the Git installer: Visit the Git website (https://git-scm.com/downloadsand) and download the appropriate version for your operating system.

    2. Run the installer: Execute the downloaded file and follow the installation instructions. Accept the default options unless you have specific needs.

    3. Verify the installation: Open your command prompt or terminal and type the following command to check if Git was installed successfully using:
		`git --version` 
		
->Through an IDE

    IDEs like PyCharm, Visual Studio Code, and others often have Git integration built-in. If you prefer using an IDE, you can usually install and configure Git through the IDE's settings or preferences menu. 
	Look for a version control section in your IDE's settings, where you can specify the path to the Git executable or even install Git directly from there.

# Cloning the Repository
Once Git is installed, you can clone this project to your local machine using the following command in your command prompt or terminal:
	`git clone https://github.com/alk5897/ML-aided-optics-for-POC-multiplexed-fluorecense.git`
	
# Setting up Python virtual environment and installing dependencies
	1. Go to the project directory
		`cd ML-aided-optics-for-POC-multiplexed-fluorecense`
	2. Set Up a Virtual Environment: 
		`python -m venv venv`
	3. Activate the Virtual Environment:
		On Windows:
		`.\venv\Scripts\activate`
		On macOS/Linux:
		`source venv/bin/activate`
		
This project requires certain dependencies to run properly. These dependencies are listed in the requirements.txt file. 
To install them, navigate to the project directory in your command prompt or terminal and run:
	`pip install -r requirements.txt`

## Files and folders
The repository is distributed into multiple sub folders such as:
	1. datasets
		Contains excel fiels of the calibration data  
	2. models
		All the machine learning models will be saved here once the respective source codes are run.
	3. outputs
		All the outputs such as predictions and concentration-wise mean absolute errors and mean square errors will be saved here once the respective source codes are run.
	4. scr
		Source codes for training and evaluation of all machine learning modesl are availble here. The codes are distrinuted into two categories:
		A. Section 3.2 of the manuscript:
			MLR-train-and-eval.py
			SVR-train-and-eval.py
			NN-train-and-eval.py
		B. Section 3.4 of the manuscript:
			NN-scalability-two-fluorophore.py
			NN-scalability-three-fluorophore.py
			NN-scalability-four-fluorophore.py
	For steps to execute these codes please refere to Usage section.

## Usage
Below is a an example of running a code within this repository.
	With the virtual environment activated, and run the script.
		'python src/NN-train-and-eval.py`

All scripts have been written to load the appropriate data files contained in the 'datasets' folder. 
Similarly, the trained models and corresponding data scalers are saved to the 'models' and the evaluation outputs in the form of excel files and figures are saved to the 'outputs' folder.

## Contributing
This project currently not open to public contributions.

## License
Copyright © 2024, Aneesh Kshirsagar at GuanLab, The Pennsylvania State University. 
All rights reserved.
This work is part of a peer-reviewed research project. It is provided for review and personal use only. 
No part of this project may be reproduced, distributed, modified, or used for commercial purposes without the express written permission of the copyright owner.
		