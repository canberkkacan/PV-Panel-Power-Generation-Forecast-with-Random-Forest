import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

def split_to_seasons(x):
    """
    months to seasons mapping, assuming the climate has 4 distinct seasons
    """
    if x in [12,1,2]:
        return 'winter'
    elif x in [3,4,5]:
        return 'spring'
    elif x in [6,7,8]:
        return 'summer'
    elif x in [9,10,11]:
        return 'fall'

def time_of_day(x):
    """
    hours into morning, highnoon, evening or night
    """
    if x in [7, 8, 9, 10]:
        return 'morning'
    elif x in [11, 12, 13, 14, 15, 16]:
        return 'highnoon'
    elif x in [17, 18, 19, 20]:
        return 'evening'
    elif x in [21, 22, 23, 0, 1, 2, 3, 4, 5, 6]:
        return 'night'


def forecast(path, forecast_date):
    pd.set_option('display.max_columns', 75)
    pd.set_option('display.min_rows', 10)
    pd.set_option('display.width', 1000)
    df = pd.read_excel(path)
    date_to_predict = forecast_date

    # Turning 24:00 to 00.00 and incrementing the day by 1
    twenty_fours = df['TIME'].str[-5:] == '24:00'
    df.loc[twenty_fours, 'TIME'] = df['TIME'].str[:-5] + '00:00'
    df['TIME'] = pd.to_datetime(df['TIME'], format='%H:%M')
    df.loc[twenty_fours, 'DATE'] += pd.DateOffset(1)
    df['TIME'] = df['TIME'].dt.strftime('%H:%M')
    df.loc[:,'DATE'] = pd.to_datetime(df.DATE.astype(str)+' '+df.TIME.astype(str))

    # Equating negative solar radiation values to zero
    df['POWER GENERATION (MW)'] = df['POWER GENERATION (MW)'].fillna(0)
    negatives = np.where(df['SOLAR RADIATION (W/m2)'] < 0.0)
    for i in range(len(negatives)):
        df.loc[negatives[i],'SOLAR RADIATION (W/m2)'] = 0.0

    # Setting the date as index
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.set_index(['DATE'])

    # Getting the hour, month, year, season and time of day data as columns
    df['hour'] = df.index.hour.astype('category')
    df['month'] = df.index.month.astype('category')
    df['year'] = df.index.year.astype('category')
    df['season'] = df['month'].apply(split_to_seasons).astype('category')
    df['time_of_day'] = df['hour'].apply(time_of_day).astype('category')


    # Exogenous variables for yearly, weekly and hourly for seasonality effect
    df['year_sin365'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['week_sin365'] = np.sin(2 * np.pi * df.index.dayofweek/7)
    df['hour_sin365'] = np.sin(2 * np.pi * df.index.hour/24)

    #min-max scaling
    df_T_min, df_T_max = df['TEMPERATURE (°C)'].min(), df['TEMPERATURE (°C)'].max()
    df_solar_min, df_solar_max = df['SOLAR RADIATION (W/m2)'].min(), df['SOLAR RADIATION (W/m2)'].max()
    df_cloud_min, df_cloud_max = df['CLOUDNESS (%)'].min(), df['CLOUDNESS (%)'].max()
    df['TEMPERATURE (°C)'] = (df['TEMPERATURE (°C)'] - df_T_min) / (df_T_max - df_T_min)
    df['SOLAR RADIATION (W/m2)'] = (df['SOLAR RADIATION (W/m2)'] - df_solar_min) / (df_solar_max - df_solar_min)
    df['CLOUDNESS (%)'] = (df['CLOUDNESS (%)'] - df_cloud_min) / (df_cloud_max - df_cloud_min)

    # Encoding months, seasons and time of days
    encoder_month=ce.OneHotEncoder(cols='month',handle_unknown='return_nan',return_df=True,use_cat_names=True)
    encoder_season=ce.OneHotEncoder(cols='season',handle_unknown='return_nan',return_df=True,use_cat_names=True)
    encoder_time=ce.OneHotEncoder(cols='time_of_day',handle_unknown='return_nan',return_df=True,use_cat_names=True)
    df = encoder_month.fit_transform(df)
    df = encoder_season.fit_transform(df)
    df = encoder_time.fit_transform(df)
    ''' 
    # Uncomment this if the actual PV gen data is available for the forecast day
    # Shift predicted environment values by 24hr
    for i in range(24):
        df['lag'+str(i+1)] = df['POWER GENERATION (MW)'].shift(i+1)
    df = df.dropna()
    '''

    # Create the model
    random_forest_df = df
    y = random_forest_df['POWER GENERATION (MW)']
    x = random_forest_df.drop(['POWER GENERATION (MW)'], axis=1).drop(['TIME'], axis=1).drop(['WIND SPEED (m/s)'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, shuffle=False)
    # Random forest model, note that parameters were found by RandomizedSearchCV (a tool for hyper tuning)
    rf = RandomForestRegressor(n_estimators = 100, min_samples_split= 3, min_samples_leaf= 4, max_features= None
                               , max_depth= None, random_state=42)
    rf.fit(X_train,y_train)

    # Finding the importances of the features
    features = df.drop(['POWER GENERATION (MW)'], axis=1).drop(['WIND SPEED (m/s)'], axis=1).drop(['TIME'], axis=1)
    feature_list = list(features.columns)
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 6)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances
    #[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

    # Prediction
    rf_prediction = rf.predict(X_test)
    predicted = y_test
    rf_prediction = pd.Series(rf_prediction, index=predicted.index)
    act_date_data = y_test[date_to_predict]
    predicted_date_data = rf_prediction[date_to_predict]


    # Uncomment this if you want to print error metrics
    mse = mean_squared_error(y_test, rf_prediction, squared=True)
    mae = mean_absolute_error(y_test, rf_prediction)
    rmse = mean_squared_error(y_test, rf_prediction, squared=False)
    r2 = r2_score(y_test, rf_prediction)
    print("r2:", r2)
    print("MSE:", mse)
    print("MAE:", mae)
    print("RMSE:", rmse)


    plt.title('Power Generation Forecast')
    plt.ylabel('Power Generation (MW)')
    plt.grid(True)
    plt.plot(act_date_data.index, act_date_data, label="Actual")
    plt.plot(predicted_date_data.index, predicted_date_data, label="Forecast")
    plt.legend()
    plt.show()
    return predicted_date_data

# Data file name should be specified in the path input and putting r before the path string may be needed to make the
# path viable. Forecast date should be in YEAR-MONTH-DAY format.
forecast_data = forecast(path, "date")
