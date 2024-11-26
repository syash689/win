# IPL-Match-Winner-Prediction

The IPL Match Winner Prediction is a machine learning project that aims to predict the outcome of IPL cricket matches. It utilizes historical match data, including team names, venues, toss decisions, player of the match, and other factors to estimate the win probability for each team.

# Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Interface](#interface)
- [Usage](#usage)
- [Features](#features)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

# Dataset
The dataset used for this project contains historical IPL match data with various features such as team names, venues, toss decisions, player of the match, and more. The dataset is preprocessed to handle missing values and converted into a suitable format for machine learning. You can find the dataset [here](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020).

# Installation
To use the IPL Match Winner Prediction, follow these steps: 

1. Clone this repository to your local machine: git clone https://github.com/sanidhyajadaun/IPL-Match-Winner-Prediction <br>
2. cd ipl-match-winner-prediction
3. Install the required Python packages: pip install -r requirements.txt

# Interface
<p align="center">
    <img src = "/Interface/Interface1.png" width = 500>
    <img src = "/Interface/Interface2.png" width = 500>
</p>

# Usage

1. Start the Flask web application: python app.py

2. Open your web browser and go to http://localhost:5000 to access the IPL Match Winner Predicton.

3. Fill in the required details for the upcoming match, such as city, venue, teams, toss winner, and toss decision.

4. Click on the "Predict" button to get the win probability for each team.

# Features
The IPL Match Winner Predictoion System offers the following features:

- User-friendly web interface for inputting match details.
- Prediction of win probability for each team based on historical match data.
- Support for various models, including Logistic Regression, Random Forest, SVM, and LightGBM.
- Visualizations for data exploration and feature importance.

# Models
The project uses multiple machine learning models to predict the match winner. Some of the models include:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- LightGBM Classifier

# Results
The performance of each model on the test dataset is evaluated using accuracy. The model with the highest accuracy is selected as the final predictor.

# Contributing
Contributions to the IPL Match Winner Prediction project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

# License
This project is licensed under the MIT License.
