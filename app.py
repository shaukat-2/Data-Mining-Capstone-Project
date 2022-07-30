from click import option
import streamlit as st
import pandas as pd
import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from joblib import dump, load

def findPolarity(sentences):
    toAnalyze = sid_obj.polarity_scores(sentences)
    sent = toAnalyze.get('compound')
    return sent

def findCounts(sentences,word):
    counts = len(re.findall(rf'\b{word}\b', sentences, re.IGNORECASE))
    return counts

sentiment = ''
sentiment_image = ''
dish_mentioned = []
cuisine_type = 'Indian'
sid_obj= SentimentIntensityAnalyzer()
dish_sentiment_df = pd.read_csv("dish_sentiment_dfTop.csv")
dish_list = dish_sentiment_df.Dish_Name.to_list()

restaurant_top = pd.read_csv("rest_rating_dishwise.csv")
zip_code_list = restaurant_top.zip_code.unique()
city_state_list = restaurant_top.city_state.unique()

# load serialized model
serialize_path = './pipe.joblib'
pipeline_serialized = load(serialize_path)



st.set_page_config(layout="wide")
reduce_header_height_style = """
    <style>
        div.block-container {padding-top:0rem;padding-left:1rem;}
    </style>
"""

# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
            
            
st.image("header.jpg", width = 900)
#st.markdown(reduce_header_height_style, unsafe_allow_html=True)
#st.markdown("<h4 style='text-align: top-right;padding: 10px;'>Review Analysis</h4>", unsafe_allow_html=True)

# Review Input
st.sidebar.markdown(reduce_header_height_style, unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='text-align: top-right;'>Indian Cuisine Information</h4>", unsafe_allow_html=True)
reviewTxt = st.sidebar.text_area('Enter Restaurant Reviews', '''this review is for the lunch buffet.this is a nice-looking restaurant, big buffet- there were lots of choices, and definitely more vegetarian-friendly options than nearby mint. i think they are a little better than mint in terms of sheer flavor and spice-level- while still not where i\'d like it to be, it\'s a little more flavorful than typical indian buffet food.the malai kofta was good and so was the aloo gobi- the latter had a nice fresh gingery flavor.naan was served piping hot but our second basket didn\'t make it to our table until we were almost done eating. the service was a bit tricky in general- several different people waited on us, and there was no coordination between them, and they could be a little slow. at one point we waited like 15 minutes for someone to get us drink refills, and then right after we asked one guy to get them, another was asking if we wanted refills. it was confusing and inefficient.it\'s somewhere between 3 and 4 stars as far as vegas indian buffets but i am erring on the side of less stars because of one glaring buffet omission: paneer. where was the paneer?''')
if(len(reviewTxt)==0):
    st.sidebar.error("Please input review text.")
else:
    pol = findPolarity(reviewTxt)
    if (pol <=0.2):
        sentiment = 'Very Poor'
        sentiment_image = "Negative.png"
    if (pol > 0.2 and pol <= 0.4):
        sentiment = 'Poor'
        sentiment_image = "Negative.png"
    elif (pol > 0.4 and pol <= 0.6):
        sentiment = 'Neutral'
        sentiment_image = "Neutral.PNG"
    elif (pol > 0.6 and pol <= 0.8):
        sentiment = 'Good'
        sentiment_image = "positive.png"
    elif (pol > 0.8 and pol <= 1):
        sentiment = 'Excellent'
        sentiment_image = "positive.png"
    

# Dish List
options_dishes = st.sidebar.multiselect(
     'Select Indian Dishes served by restaurant',
     dish_list,
     ['naan', 'rice'])



if(len(options_dishes)<=0):
    st.sidebar.error("Please select at least one dish.")
if(len(options_dishes)> 3):
    st.sidebar.error("Please select at most three dishes.")
else:
    dish_list = dish_list + options_dishes
    for dish in dish_list:
        if(findCounts(reviewTxt.replace("\t", " ").replace("\n", "").replace("\r", "").lower().strip(),dish)>0 ):
            dish_mentioned.append(dish)
    dish_mentioned = list(set(dish_mentioned))
dish_mentioned_df = pd.DataFrame({'Indian Dishes Mined & Selected':dish_mentioned}) 
# dish_selected_df = pd.DataFrame({'Dish Selected':options_dishes})   


st.sidebar.write("Cuisine Selection: " + cuisine_type)

# Zip Code Input
option_zip = st.sidebar.selectbox(
     'Please select relevant area:',
     zip_code_list)
if(len(option_zip) < 0):
    st.error("Please select area.")

selected_area = restaurant_top[restaurant_top.zip_code == option_zip]
option_area = selected_area.city_state.unique()
option_area = ''.join(option_area)

# Area Selected
st.sidebar.write("Area Selected: " + option_area)



# Average Rating Input
average_rating = st.sidebar.radio(
     "Select Average Rating for the Restaurant",
     ('1', '2', '3','4','5'), horizontal=True)

# Map
#st.map(latlongphx,use_container_width=True)

tdf = restaurant_top[restaurant_top.Dish_Name.isin(dish_mentioned)].sort_values(by=["Dish_Name","Sentiment"])
tdf = tdf[tdf.city_state == option_area]
tdf = tdf.groupby("Dish_Name").head(2).reset_index(drop=True).sort_values(by=['Dish_Name','Sentiment'])
recommended_restaurants = tdf.name.unique()
recommended_restaurants = pd.DataFrame({'Recommended Restaurants based on Mined Dishes':recommended_restaurants}) 

latlong_info = tdf[["city_state","latitude",'longitude']]
latlong_infoMap = latlong_info.copy()
latlong_infoMap.drop_duplicates(keep='first', inplace = True)


submit = st.sidebar.button('Submit')
if submit:
    
    cuisines_offered = str(cuisine_type) # need to convert list to string
    data = [[reviewTxt, cuisine_type, option_zip, average_rating]]
    input_df = pd.DataFrame(data, columns = ['preprocessed_texts', 'cuisines_offered', 'zipcode', 'avg_rating'])
    pred = float(pipeline_serialized.predict_proba(input_df)[:,1])
    output_string = 'This restaurant is: {:.2%} likely to pass a hygiene inspection'.format(pred)
    
    
    c1, c2 = st.columns([2,10], gap='small')
    
    with st.container():
        c1.write(output_string)
        c1.markdown(hide_table_row_index, unsafe_allow_html=True)
        c1.table(recommended_restaurants)
        c2.map(latlong_infoMap)
        c1.markdown(hide_table_row_index, unsafe_allow_html=True)
        c1.table(dish_mentioned_df)
        c1.image(sentiment_image, width = 100, caption='Sentiment')



    





