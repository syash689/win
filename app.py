# Importing necessary modules from the Flask framework
from flask import Flask, render_template, request, jsonify


import numpy as np  # Library for numerical operations
import pandas as pd  # Library for data manipulation and analysis
from sklearn.model_selection import train_test_split  # Function to split data into training and testing sets
from sklearn.metrics import accuracy_score  # Function to calculate accuracy score
import lightgbm as lgb  # Importing LightGBM library, Light Gradient Boosting Machine library

# Create an instance of the Flask class and name it 'app'
app = Flask(__name__)



@app.route('/')
def Match_prediction_form():
    # The route decorator, '@app.route()', defines the URL path that this function will handle.
    # In this case, when a user accesses the root URL ('/'), this function will be executed.

    # The 'render_template' function is used to render an HTML template.
    # In this case, it will render the 'match_predict.html' template, which contains the form for match prediction.
    return render_template('match_predict.html')




@app.route('/predict', methods=['POST'])
 # The route decorator, '@app.route()', defines the URL path that this function will handle.
# In this case, when a POST request is made to the '/predict' URL, this function will be executed.
def predict():
     
    # Get data from user input
    City = request.form['city']
    Player_of_match = request.form['player_of_match']
    Venue = request.form['venue']
    Neutral_venue = int(request.form['neutral_venue'])
    Team1 = request.form['team1']
    Team2 = request.form['team2']
    Toss_winner = request.form['toss_winner']
    Toss_decision = request.form['toss_decision']
    Result_margin = float(request.form['result_margin'])
    Umpire1 = request.form['umpire1']
    Umpire2 = request.form['umpire2']

    # Reading the CSV file 'ipl.csv' and storing the data in a DataFrame called 'data'
    data = pd.read_csv(r'Final Project\ipl.csv')
    
    # Identifying information about composition and potential data quality
    # data.info()

    # Replacing 'Rising Pune Supergiants' with 'Rising Pune Supergiant' in the 'team1', 'team2', 'winner', and 'toss_winner' columns.
    data.team1.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant'},regex=True,inplace=True)
    data.team2.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant'},regex=True,inplace=True)
    data.winner.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant'},regex=True,inplace=True)
    data.toss_winner.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant'},regex=True,inplace=True)

    # Replacing 'Deccan Chargers' with 'Sunrisers Hyderabad' in the 'team1', 'team2', 'winner', and 'toss_winner' columns.
    data.team1.replace({'Deccan Chargers' : 'Sunrisers Hyderabad'},regex=True,inplace=True)
    data.team2.replace({'Deccan Chargers' : 'Sunrisers Hyderabad'},regex=True,inplace=True)
    data.winner.replace({'Deccan Chargers' : 'Sunrisers Hyderabad'},regex=True,inplace=True)
    data.toss_winner.replace({'Deccan Chargers' : 'Sunrisers Hyderabad'},regex=True,inplace=True)

    # Replacing 'Delhi Daredevils' with 'Delhi Capitals' in the 'team1', 'team2', 'winner', and 'toss_winner' columns.
    data.team1.replace({'Delhi Daredevils' : 'Delhi Capitals'},regex=True,inplace=True)
    data.team2.replace({'Delhi Daredevils' : 'Delhi Capitals'},regex=True,inplace=True)
    data.winner.replace({'Delhi Daredevils' : 'Delhi Capitals'},regex=True,inplace=True)
    data.toss_winner.replace({'Delhi Daredevils' : 'Delhi Capitals'},regex=True,inplace=True)

    # Replacing 'Pune Warriors' with 'Rising Pune Supergiant' in the 'team1', 'team2', 'winner', and 'toss_winner' columns.
    data.team1.replace({'Pune Warriors' : 'Rising Pune Supergiant'},regex=True,inplace=True)
    data.team2.replace({'Pune Warriors' : 'Rising Pune Supergiant'},regex=True,inplace=True)
    data.winner.replace({'Pune Warriors' : 'Rising Pune Supergiant'},regex=True,inplace=True)
    data.toss_winner.replace({'Pune Warriors' : 'Rising Pune Supergiant'},regex=True,inplace=True)

    # checking for the null values in the dataset
    # data.isnull().sum()
    # there exists null values

    # Fill missing values in 'city' column with 'Unknown'
    data['city'].fillna('Unknown', inplace=True)

    # Fill missing values in 'player_of_match', 'result', and 'eliminator' columns with 'Not Available'
    cols_to_fill = ['player_of_match', 'result', 'eliminator']
    data[cols_to_fill] = data[cols_to_fill].fillna('Not Available')

    # Calculate the mean of the 'result_margin' column
    mean_result_margin = data['result_margin'].mean()

    # Fill missing values in 'result_margin' column with the mean
    data['result_margin'].fillna(mean_result_margin, inplace=True)

    data.drop(['id','method'],axis=1,inplace=True)

   # Drop rows with missing values in the 'winner' column
    data.dropna(subset=['winner'], inplace=True)

    data['date'] = pd.to_datetime(data['date'])
    data['season'] = pd.DatetimeIndex(data['date']).year

    # Extracting day, month, and year from the 'date' column
    data['day'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year

    # Create a dictionary to map team names to unique numbers
    team_mapping = {
        'Kolkata Knight Riders': 1,
        'Chennai Super Kings': 2,
        'Delhi Capitals': 3,
        'Royal Challengers Bangalore': 4,
        'Rajasthan Royals': 5,
        'Kings XI Punjab': 6,
        'Sunrisers Hyderabad': 7,
        'Mumbai Indians': 8,
        'Rising Pune Supergiant': 9,
        'Kochi Tuskers Kerala': 10,
        'Gujarat Lions': 11
    }

    # Replace team names in 'team1' and 'team2' columns with unique numbers
    data['team1'] = data['team1'].map(team_mapping)
    data['team2'] = data['team2'].map(team_mapping)

    # Replace winner names in 'winner' column with unique numbers
    data['winner'] = data['winner'].map(team_mapping)
    data['toss_winner'] = data['toss_winner'].map(team_mapping)

    # Create a dictionary to map each unique venue name to a unique number
    venue_mapping = {venue: i for i, venue in enumerate(data['venue'].unique())}

    # Replace the venue names in the 'venue' column with the corresponding unique numbers
    data['venue'] = data['venue'].map(venue_mapping)

    # Create a dictionary to map 'toss_decision' values to numerical values
    temp = {'field': 0, 'bat': 1}

    # Use the map() function to replace 'toss_decision' values with numerical values
    data['toss_decision'] = data['toss_decision'].map(temp)

    # Create a set of unique umpires
    umpires_set = set(data['umpire1'].unique()).union(set(data['umpire2'].unique()))

    # Create a dictionary to map umpire names to unique numbers
    umpire_dict = {umpire: i for i, umpire in enumerate(umpires_set, 1)}

    # Apply the dictionary to create new encoded columns for 'umpire1' and 'umpire2'
    data['umpire1'] = data['umpire1'].map(umpire_dict)
    data['umpire2'] = data['umpire2'].map(umpire_dict)


    # Create a dictionary to map each unique venue name to a unique number
    player_of_match_mapping = {venue: i for i, venue in enumerate(data['player_of_match'].unique())}

    # Replace the venue names in the 'venue' column with the corresponding unique numbers
    data['player_of_match'] = data['player_of_match'].map(player_of_match_mapping)


    # Create a dictionary to map each unique venue name to a unique number
    city_mapping = {venue: i for i, venue in enumerate(data['city'].unique())}

    # Replace the venue names in the 'venue' column with the corresponding unique numbers
    data['city'] = data['city'].map(city_mapping)

    # List of unwanted columns
    unwanted_columns = ['date','result','eliminator','season','day','month','year']

    # Drop the unwanted columns from the DataFrame
    data.drop(columns=unwanted_columns, inplace=True)

    # Split the data into features (X) and the target variable (y)
    X = data.drop(['winner'], axis=1)
    y = data['winner']

    # Split the data into training and testing sets (70% training, 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Display the shapes of the training and testing sets
    # print("X_train shape:", X_train.shape)
    # print("y_train shape:", y_train.shape)
    # print("X_test shape:", X_test.shape)
    # print("y_test shape:", y_test.shape)

    # Create an instance of the LGBMClassifier model
    model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    #y_pred = model.predict(X_test)

    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy:", accuracy)

    # Map user input to numerical forms based on the mappings
    city_numeric = city_mapping.get(City, -1)
    player_of_match_numeric = player_of_match_mapping.get(Player_of_match, -1)
    venue_numeric = venue_mapping.get(Venue, -1)
    team1_numeric = team_mapping.get(Team1,-1)
    team2_numeric = team_mapping.get(Team2,-1)
    toss_winner_numeric = team_mapping.get(Toss_winner,-1)
    toss_decision_numeric = temp.get(Toss_decision,-1)
    umpire1_numeric = umpire_dict.get(Umpire1,-1)
    umpire2_numeric = umpire_dict.get(Umpire2,-1)

    user_data = pd.DataFrame({
        'city': [city_numeric],
        'player_of_match': [player_of_match_numeric],
        'venue': [venue_numeric],
        'neutral_venue': [Neutral_venue],
        'team1': [team1_numeric],
        'team2': [team2_numeric],
        'toss_winner': [toss_winner_numeric],
        'toss_decision': [toss_decision_numeric],
        'result_margin': [Result_margin],
        'umpire1': [umpire1_numeric],
        'umpire2': [umpire2_numeric]
    })

    # Make predictions on the user input data
    predictions = model.predict(user_data)

    # Get the probability of winning for the 1st team (team1)
    win_probability_team1 = predictions[0]

    # Convert probability to percentage
    win_probability_percentage_team1 = win_probability_team1 * 10
    # Convert percentage probability to string type
    win_probability_percentage_team1 = str(win_probability_percentage_team1) + "%"

    # Render the 'match_predict.html' template and pass the prediction result to display on the webpage
    return render_template('match_predict.html', prediction=win_probability_percentage_team1)




if __name__ == '__main__':
    # This block of code runs the Flask application when the script is executed directly.

    # The app.run() function starts the Flask development server.
    # - 'debug=True' enables the debug mode, which provides helpful error messages during development.
    # - 'host='0.0.0.0'' makes the app accessible from all network interfaces, allowing external access.
    app.run(debug=True, host='0.0.0.0')








