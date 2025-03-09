import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('Machine Learning App')

st.info('This app builds a machine learning model')

with st.expander('Data'):
  st.write('**Raw data**')
  df= pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")
  df

  st.write('**X**')
  X_raw=df.drop('species',axis=1)
  X_raw
  
  st.write('**y**')
  y_raw=df.species
  y_raw

with st.expander('Data Visualization'):
  st.scatter_chart(data=df,x='bill_length_mm',y='body_mass_g',color='species')

# Input Featues
with st.sidebar:
  st.header('Input features')
  island=st.selectbox('Island',('Biscoe','Dream','Torgersen'))
  bill_length_mm=st.slider("Bill Length(mm)",32.1,59.6,43.9)
  bill_depth_mm=st.slider("Bill Depth(mm)",13.1,21.5,17.2)
  flipper_length_mm=st.slider('Flipper Length(mm)',172.0,231.0,201.0)
  body_mass_g=st.slider("Body Mass(gm)",2700.0,4207.0,6300.0)
  gender=st.selectbox('Gender',('Male','Female'))

  #create a df for input features
  data = {'island':island,
          'bill_length_mm':bill_length_mm,
          'bill_depth_mm':bill_depth_mm,
          'flipper_length_mm':flipper_length_mm,
          'body_mass_g':body_mass_g,
          'sex':gender}
  input_df= pd.DataFrame(data,index=[0])
  input_penguins = pd.concat([input_df,X_raw],axis=0)  

with st.expander('Input features'):
  st.write('**Input penguin features**')
  input_df
  st.write('**Combined penguins feature**')
  input_penguins

#data prep
#Encode X
encode = ['island','sex']
df_penguins= pd.get_dummies(input_penguins,prefix=encode)
X= df_penguins[1:]
input_row = df_penguins[:1]
#Encode y
target_mapper = {
  'Adelie':0,
  'Chinstrap':1,
  'Gentoo':2}
def target_encode(val):
  return target_mapper[val]

y=y_raw.apply(target_encode)
y
y_raw

with st.expander('Data preparation'):
  st.write('**Encoded input penguin(X)**')
  input_row
  st.write("**Encoded y**")
  y

#Model Training
#train the model
clf = RandomForestClassifier()
clf.fit(X,y)

#apply the model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba= pd.DataFrame(prediction_proba)
df_prediction_proba.column=['Adelie','Chinstrap','Gentoo']
df_predicton_proba.rename(columns={0:'Adelie',
                                    1:'Chinstrap',
                                    2:'Gentoo'})

                                  
#display predicted species
st.subheader('Predicted Species')
st.dataframe(df_prediction_proba,
             column_config={
               'Adelie':st.column_config.ProgressColumn(
                  'Adelie',                                             
                  format='%f',                                                                     
                  width='medium',   
                  min_value=0,
                  max_value=1),
              'Chinstrap':st.column_config.ProgressColumn(
                  'Chinstrap',                                             
                  format='%f',                                                                     
                  width='medium',   
                  min_value=0,
                  max_value=1),
               'Gentoo':st.column_config.ProgressColumn(
                  'Gentoo',                                             
                  format='%f',                                                                     
                  width='medium',   
                  min_value=0,
                  max_value=1
               ),
             },hide_index=True)
                                                                                         


df_prediciton_proba

penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.success(str(penguins_species[prediction][0]))
