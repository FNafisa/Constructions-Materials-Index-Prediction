import pandas as pd
import matplotlib.pyplot as plt
from autots import AutoTS

def ts_pipeline(construction_material_name):
    # 1 store the path of input file
    name_w_extention= construction_material_name.strip() +'.xlsx'
    path_input= r'C:\Users\fnafisa\Construction Material Price Index\data\training data\combined'+f'\{name_w_extention}'

    # 2 load the data
    dataset_train= pd.read_excel(path_input)
    dataset_train
    
    # 3 prepare the data
    dataset_train= dataset_train.sort_values(by=['Date'], ascending=True)

    # 4 display correlation
    correlation= dataset_train[["Price","brent_oil_price",'gold_price', 'tasi_price', 'fed_rate']].corr()
    print(construction_material_name,correlation['Price'].sort_values(ascending=False))

    # 5 train the model
    model= AutoTS(forecast_length= 3, frequency= 'infer', ensemble= 'simple', n_jobs=-1)
    model= model.fit(dataset_train, date_col= 'Date', value_col= 'Price', id_col= None)

    # 6 forecast
    prediction= model.predict()
    forecast= prediction.forecast

    # 7 setup datasets
    dataset_train['data_point']= 'actual'


    forecast['Date']= forecast.index
    forecast['data_point']= 'prediction'

    # Combine dataset_train and forecast dataframes
    df_combined = pd.concat([dataset_train, forecast[['Date', 'Price', 'data_point']]], ignore_index=True)
    # Fill NaN values in specified columns with the most frequent value
    columns_to_fill = ['Category_En', 'Category', 'Material_AR', 'Material_EN', 'Unit_ar', 'Unit']
    for col in columns_to_fill:
        if col in df_combined.columns:
            mode_val = df_combined[col].mode()
            if not mode_val.empty:
                df_combined[col] = df_combined[col].fillna(mode_val[0])

    # 10 export result
    file_location=  r'C:\Users\fnafisa\Construction Material Price Index\data\predictions data\{}'
    file_location= file_location.format(name_w_extention)
    df_combined.to_excel(file_location, index=False)
    
# loop through all the elements to predict each one of them
# in every iteration 
type_names= pd.read_excel(r'C:\Users\fnafisa\Construction Material Price Index\data\training data\target\type names.xlsx')
type_names.columns= ['names']

type_names= type_names.names.values[:]


for name in type_names:
    ts_pipeline(name)