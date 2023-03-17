import streamlit as st
import pickle
import pandas as pd
import xgboost
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor
import pandas as pd
import numpy as np
from operator import attrgetter
import datetime
from datetime import datetime
import pydeck as pdk

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; color: white;'>T H E B R A I N</h1>", unsafe_allow_html=True)

margins_css = """
    <style>
        .main > div {
            padding-left: 10rem;
            padding-right: 10rem;
        }
    </style>
"""

st.markdown(margins_css, unsafe_allow_html=True)

st.markdown(
    """
    <style>
@font-face {
font-family: 'Ogg Roman';
font-style: normal;
font-weight: normal;
src: local('Ogg-Roman'), url('Ogg-Roman.ttf') format('ttf');
}
html, body, [class*="css"] {
font-family: 'Ogg Roman', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

#################################

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "gif"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         

         </style>
         """,
         unsafe_allow_html=True
     )

main_bg_ext = "Sci Fi HUD_12.gif"
set_bg_hack(main_bg_ext)

with open('style.css') as f:
   st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

col1.metric("INSTAGRAM FOLLOWERS", '30.7 K', 'PLACEHOLDER') #PLACE HOLDER
col2.metric("PRODUCTS SOLD IN LAST 24H", '600', 'PLACEHOLDER') #PLACE HOLDER
col3.metric("QUARTERLY TARGETED COMPLETED", '95%', 'PLACEHOLDER') #PLACE HOLDER

df_city_spent = pd.read_csv('customer_location.csv')
X_future = pd.read_csv('X_future.csv')
X_complete = pd.read_csv('X_complete.csv')
pd.to_datetime(X_complete['Date'])
X_complete.set_index(['Date'], inplace=True)

ranking = pd.read_csv('ranking.csv')
ranking = ranking.iloc[:,1]

y_complete = pd.read_csv('y_complete.csv')
pd.to_datetime(y_complete['Date'])
y_complete.set_index(['Date'], inplace=True)
y_complete = pd.DataFrame(y_complete)
X_future = pd.DataFrame(X_future)
X_future = X_future.rename(columns={'Unnamed: 0': 'Date'})
X_future = X_future.set_index(pd.DatetimeIndex(X_future['Date']))
X_future.set_index(['Date'], inplace=True)

# # # # pre-processing
scaler = StandardScaler()
X_future_scaled = scaler.fit_transform(X_future)
option=()

c1, c2, c3= st.columns((1, 1.5, 4))

with c1:
        st.subheader('SALES FORECAST')
        dummy_data = ['XGBoost', 'Random Forest', 'Elastic Net']
        ml_model = st.selectbox('', options=['Select Model']+list(dummy_data))
        option = ml_model 
        option_2 = st.radio(
        "CROSS-VALIDATION:",
        key="fold_q",
        options=["Yes","No"],
        )
        st.markdown("---")

with c3:

    if option == 'Select Model':    

        st.line_chart(data=y_complete , use_container_width=True)
        st.write('---')

    elif option == 'XGBoost':

            if option_2 == 'Yes':
                loaded_model = pickle.load(open('finalized_XGBoostmodel.sav', 'rb'))
                result = loaded_model.predict(X_future_scaled)
                result= pd.DataFrame(result, index = X_future.index)
                result2 = result.loc['2023-03-12':]
                df = pd.concat([y_complete, result2], axis=1)
                df.columns = ['Gross sales', 'XGBoost']
                df2 = df.loc['2022-06-06':]
                st.line_chart(data=df2 , use_container_width=True)
            elif option_2 == 'No':
                loaded_model = pickle.load(open('finalized_XGBoostmodel_rtts.sav', 'rb'))
                result = loaded_model.predict(X_future_scaled)
                result= pd.DataFrame(result, index = X_future.index)
                result2 = result.loc['2023-02-03':]
                df = pd.concat([y_complete, result2], axis=1)
                df.columns = ['Gross sales', 'XGBoost']
                df2 = df.loc['2022-06-06':'2023-04-12']
                st.line_chart(data=df2 , use_container_width=True)

    elif option == 'Random Forest':
                        
                    if option_2 == 'Yes':    
                        loaded_model = pickle.load(open('finalized_RandomForestmodel.sav', 'rb'))
                        result = loaded_model.predict(X_future_scaled)
                        result= pd.DataFrame(result, index = X_future.index)
                        result2 = result.loc['2023-02-12':]
                        df = pd.concat([y_complete, result2], axis=1)
                        df.columns = ['Gross sales', 'Random Forest']
                        df2 = df.loc['2022-06-06':'2023-04-12']
                        st.line_chart(data=df2 , use_container_width=True)
                    elif option_2 == 'No':
                        loaded_model = pickle.load(open('finalized_RandomForestmodel_rtts.sav', 'rb'))
                        result = loaded_model.predict(X_future_scaled)
                        result= pd.DataFrame(result, index = X_future.index)
                        result2 = result.loc['2023-02-12':]
                        df = pd.concat([y_complete, result2], axis=1)
                        df.columns = ['Gross sales', 'Random Forest']
                        df2 = df.loc['2022-06-06':'2023-04-12']
                        st.line_chart(data=df2 , use_container_width=True)

    elif option == 'Elastic Net':
                    
                    if option_2 == 'Yes':
                        loaded_model = pickle.load(open('finalized_ElasticNet.sav', 'rb'))
                        result = loaded_model.predict(X_future_scaled)
                        result= pd.DataFrame(result, index = X_future.index)
                        result2 = result.loc['2023-02-12':]
                        df = pd.concat([y_complete, result2], axis=1)
                        df.columns = ['Gross sales', 'Elastic Net']
                        df2 = df.loc['2022-06-06':'2023-04-12']
                        st.line_chart(data=df2 , use_container_width=True)
                    elif option_2 == 'No':
                        loaded_model = pickle.load(open('finalized_ElasticNet_rtts.sav', 'rb'))
                        result = loaded_model.predict(X_future_scaled)
                        result= pd.DataFrame(result, index = X_future.index)
                        result2 = result.loc['2023-02-12':]
                        df = pd.concat([y_complete, result2], axis=1)
                        df.columns = ['Gross sales', 'Elastic Net']
                        df2 = df.loc['2022-06-06':'2023-04-12']
                        st.line_chart(data=df2 , use_container_width=True)

c1_2, c2_2, c3_2= st.columns((2, 0.5, 4));

with c1_2:

    st.subheader("TARGETED PREDICTION:")

    d = st.date_input("PREDICT SALES ON:")
    audience_growth = st.slider('NET AUDIENCE GROWTH',max_value = 500, step= 10)
    website_traffic = st.slider('WEBSITE TRAFFIC',max_value = 5000, step= 10)

    X_predict = {'Date': d,
                'Net Audience Growth': audience_growth, 
                'Website Traffic': website_traffic}

    X_predict = pd.DataFrame(X_predict, index=[0])

    X_predict = X_predict.set_index(pd.DatetimeIndex(X_predict['Date']))

    del X_predict['Date']

    lag_time=pd.DataFrame()

    X_predict['dayofweek'] = X_predict.index.dayofweek
    X_predict['quarter'] = X_predict.index.quarter
    X_predict['month'] = X_predict.index.month
    X_predict['dayofyear'] = X_predict.index.dayofyear
    X_predict['dayofmonth'] = X_predict.index.day
    X_predict['weekofyear'] = pd.Int64Index(X_predict.index.isocalendar().week)      #ofyear

    t1='30 days' #30 days X 3
    t2='60 days'
    t3='90 days'

    X_past = pd.read_csv('X_past.csv')

    X_past = X_past.set_index(pd.DatetimeIndex(X_past['Date']))

    del X_past['Date']

    X_predict['Net Audience Growth_lag1']= X_past['Net Audience Growth'].loc[X_past.index.max()- pd.Timedelta(t1)]
    X_predict['Net Audience Growth_lag2']= X_past['Net Audience Growth'].loc[X_past.index.max()- pd.Timedelta(t2)]
    X_predict['Net Audience Growth_lag3']= X_past['Net Audience Growth'].loc[X_past.index.max()- pd.Timedelta(t3)]

    X_predict['Website Traffic_lag1']= X_past['Website Traffic'].loc[X_past.index.max()- pd.Timedelta(t1)]
    X_predict['Website Traffic_lag2']= X_past['Website Traffic'].loc[X_past.index.max()- pd.Timedelta(t2)]
    X_predict['Website Traffic_lag3']= X_past['Website Traffic'].loc[X_past.index.max()- pd.Timedelta(t3)]

    X_predict['dayofweek_lag1']= X_past['dayofweek'].loc[X_past.index.max()- pd.Timedelta(t1)]
    X_predict['dayofweek_lag2']= X_past['dayofweek'].loc[X_past.index.max()- pd.Timedelta(t2)]
    X_predict['dayofweek_lag3']= X_past['dayofweek'].loc[X_past.index.max()- pd.Timedelta(t3)]

    X_predict['quarter_lag1']= X_past['quarter'].loc[X_past.index.max()- pd.Timedelta(t1)]
    X_predict['quarter_lag2']= X_past['quarter'].loc[X_past.index.max()- pd.Timedelta(t2)]
    X_predict['quarter_lag3']= X_past['quarter'].loc[X_past.index.max()- pd.Timedelta(t3)]

    X_predict['month_lag1']= X_past['month'].loc[X_past.index.max()- pd.Timedelta(t1)]
    X_predict['month_lag2']= X_past['month'].loc[X_past.index.max()- pd.Timedelta(t2)]
    X_predict['month_lag3']= X_past['month'].loc[X_past.index.max()- pd.Timedelta(t3)]

    X_predict['dayofmonth_lag1']= X_past['dayofmonth'].loc[X_past.index.max()- pd.Timedelta(t1)]
    X_predict['dayofmonth_lag2']= X_past['dayofmonth'].loc[X_past.index.max()- pd.Timedelta(t2)]
    X_predict['dayofmonth_lag3']= X_past['dayofmonth'].loc[X_past.index.max()- pd.Timedelta(t3)]

    X_predict['weekofyear_lag1']= X_past['weekofyear'].loc[X_past.index.max()- pd.Timedelta(t1)]
    X_predict['weekofyear_lag2']= X_past['weekofyear'].loc[X_past.index.max()- pd.Timedelta(t2)]
    X_predict['weekofyear_lag3']= X_past['weekofyear'].loc[X_past.index.max()- pd.Timedelta(t3)]

    X_predict = X_predict[['Net Audience Growth', 'Net Audience Growth_lag1',
                        'Net Audience Growth_lag2', 
                        'Net Audience Growth_lag3', 
                        'Website Traffic', 
                        'Website Traffic_lag1', 
                        'Website Traffic_lag2',
                        'Website Traffic_lag3', 
                        'dayofweek', 
                        'dayofweek_lag1', 
                        'dayofweek_lag2',
                        'dayofweek_lag3', 
                        'quarter', 
                        'quarter_lag1', 
                        'quarter_lag2',
                        'quarter_lag3', 
                        'month', 
                        'month_lag1', 
                        'month_lag2', 
                        'month_lag3',
                        'dayofmonth', 
                        'dayofmonth_lag1', 
                        'dayofmonth_lag2', 
                        'dayofmonth_lag3',
                        'weekofyear', 
                        'weekofyear_lag1', 
                        'weekofyear_lag2', 
                        'weekofyear_lag3']]
    seer = st.button('S E E R')

#with c2_2:
    if seer:
        c1_3, c2_3, c3_3 = st.columns(3)

        X_predict_scaled = scaler.transform(X_predict)

        with c1_3:
            loaded_model_XGBoost = pickle.load(open('finalized_XGBoostmodel.sav', 'rb'))
            result_model_XGBoost = loaded_model_XGBoost.predict(X_predict_scaled).round(2)
            st.metric(label="XGBoost", value=int(result_model_XGBoost))
        with c2_3:
            loaded_model_rfr = pickle.load(open('finalized_RandomForestmodel.sav', 'rb'))
            result_model_rfr  = loaded_model_rfr.predict(X_predict_scaled).round(2)
            st.metric(label="Random Forest", value=int(result_model_rfr ))
        with c3_3:
            loaded_model_elasticnet = pickle.load(open('finalized_ElasticNet.sav', 'rb'))
            result_model_elasticnet = loaded_model_elasticnet.predict(X_predict_scaled).round(2)
            st.metric(label="ElasticNet", value=int(result_model_elasticnet))
    # if seer:

        models = [('XGBoost', loaded_model_XGBoost), 
            ('Random Forest',loaded_model_rfr),
            ('ElasticNet', loaded_model_elasticnet)
            ]

        X_complete_scaled = scaler.transform(X_complete)

        ensemble = VotingRegressor(estimators=models, weights = ranking)
        train_complete_ensemble = ensemble.fit(X_complete_scaled, y_complete)
        pred_complete_ensemble = ensemble.predict(X_predict_scaled)
        # models_mean = np.mean([result_model_XGBoost,result_model_rfr,result_model_elasticnet ])
        st.metric(label="Ensemble:", value=int(pred_complete_ensemble))

with c3_2:  
        st.subheader("CUSTOMERVERSE")
        #plot customers map
        df_city_spent_plot = pd.DataFrame()
        df_city_spent_plot['lat'] = df_city_spent['latitude']
        df_city_spent_plot['lon'] = df_city_spent['longitude']
        height = 300
        width = 700


        layer=pdk.Layer(
                'HexagonLayer',
                data=df_city_spent_plot,
                get_position='[lon, lat]',
                auto_highlight=True,
                elevation_scale=1000,
                pickable=True,
                elevation_range=[0, 3000],
                extruded=True,
                coverage=15)


# Set the viewport location
        view_state = pdk.ViewState(
            longitude=-1.415,
            latitude=53.4,
            zoom=3,
            min_zoom=5,
            max_zoom=1000,
            pitch=40.5,
            )

        r = st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            'html': '<b>Elevation Value:</b> {elevationValue}',
            'style': {
                'color': 'white'
                    }
                }
            ))