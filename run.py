import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from lime import explanation
from lime import lime_base
from lime_timeseries import LimeTimeSeriesExplainer

st.sidebar.header('Parameter Setting')
with st.sidebar.form(key ='form1'):
    courseid = st.text_input('*Course ID (e.g., CL10120192)')
    startdate = st.text_input('*Course Start Date (e.g., 2020-02-23)')
    enddate = st.text_input('*Course End Date (e.g., 2020-06-29)')
    studentid = st.text_input('*Student ID (e.g., 3180300116)')
    datatype = st.selectbox('Time Series Data Type Preferred', ('time spent per day (minutes)', 
    'log time spent per day (seconds)', 'visits per day'))
    num_features = st.selectbox('Number of Features Preferred', ('1', '2', '3', '4', '5', 
    '6', '7', '8', '9', '10'))
    cluster_n = st.selectbox('Number of Clusters Preferred', ('1', '2', '3', '4', '5', 
    '6', '7', '8', '9', '10'))
    
    st.write('*: required parameters')
    submitted1 = st.form_submit_button(label = 'Submit Parameters')

@st.cache
def x1_x2_x3_table(): 
    url = "https://raw.githubusercontent.com/kevinTHlin/Character_Persona/master/app/{0}.csv".format(courseid)
    df_uploaded = pd.read_csv(url, sep=",")
    class_start_date =  datetime.strptime(startdate , '%Y-%m-%d').date()
    class_end_date = datetime.strptime(enddate , '%Y-%m-%d').date()
    after_end = 196 - 28 - len(pd.date_range(class_start_date, class_end_date))
    new_index = pd.date_range(class_start_date - timedelta(days=28), class_end_date + timedelta(days=after_end))
    user = df_uploaded['get_user_ID'].unique()

    Table = {} 
    df_time_sum_minutes = pd.DataFrame(columns = new_index)
    df_time_sum_log = pd.DataFrame(columns = new_index)
    df_visits = pd.DataFrame(columns = new_index)
    
    for i in user:
        Table[i] = df_uploaded.loc[df_uploaded['get_user_ID'] == i]
        Table[i]['get_user_ID'] = Table[i]['get_user_ID'].astype(str)
        Table[i] = Table[i].replace('不足1s', '00:00:01')
        Table[i]['time_spent'] = pd.to_timedelta(Table[i].time_spent)
        Table[i]['visit_time'] = pd.to_datetime(Table[i].visit_time, format='%Y.%m.%d %H:%M')
        Table[i]['visit_date'] = Table[i]['visit_time'].dt.date
        Table[i] = Table[i].sort_values('visit_time')
        Table[i]['time_spent'] = Table[i]['time_spent'].dt.total_seconds()
        Table[i] = Table[i].groupby(['visit_date']).agg(time_sum = 
                                                         ('time_spent', 'sum'),
                                                         visits_count = 
                                                         ('visit_time', 'count')).reset_index()
        Table[i]['time_sum_log'] = np.log(Table[i]['time_sum']+1) 
        Table[i]['time_sum'] = Table[i]['time_sum']/60
        Table[i].set_index('visit_date', inplace=True)
        Table[i].index = pd.DatetimeIndex(Table[i].index)
        Table[i] = Table[i].reindex(new_index, fill_value=0)
        Table[i] = Table[i].transpose()
        
        df_time_sum_minutes.loc[str(i)] = Table[i].loc['time_sum']
        df_time_sum_log.loc[str(i)] = Table[i].loc['time_sum_log']
        df_visits.loc[str(i)] = Table[i].loc['visits_count']
    print(df_time_sum_minutes.isnull().sum().sum())
    print(df_time_sum_minutes.isin([np.inf, -np.inf]).sum().sum())  
    print(df_time_sum_log.isnull().sum().sum())
    print(df_time_sum_log.isin([np.inf, -np.inf]).sum().sum())  
    print(df_visits.isnull().sum().sum())
    print(df_visits.isin([np.inf, -np.inf]).sum().sum()) 
    
    scaler = MinMaxScaler()  
    df_time_sum_minutes_scaled = df_time_sum_minutes[:]
    df_time_sum_minutes_scaled = scaler.fit_transform(df_time_sum_minutes_scaled)    
    df_time_sum_minutes_scaled = pd.DataFrame(df_time_sum_minutes_scaled, index = df_time_sum_minutes.index, 
                                              columns = df_time_sum_minutes.columns)
    df_time_sum_log_scaled = df_time_sum_log[:]
    df_time_sum_log_scaled = scaler.fit_transform(df_time_sum_log_scaled)    
    df_time_sum_log_scaled = pd.DataFrame(df_time_sum_log_scaled, index = df_time_sum_log.index, 
                                              columns = df_time_sum_log.columns)
    df_visits_scaled = df_visits[:]
    df_visits_scaled = scaler.fit_transform(df_visits_scaled)    
    df_visits_scaled = pd.DataFrame(df_visits_scaled, index = df_visits.index, 
                                              columns = df_visits.columns)
    
    return df_time_sum_minutes_scaled, df_time_sum_log_scaled, df_visits_scaled


@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
def y_table_fig(cluster_n, cluster_coldict_n = {0:'cornflowerblue', 
                        1:'mediumaquamarine', 2:'khaki', 3:'tomato',
                        4:'lightpink', 5:'cyan', 6:'blue',
                        7:'teal', 8:'yellowgreen', 9:'olive'}, PC_n=2):
    url = "https://raw.githubusercontent.com/kevinTHlin/Character_Persona/master/app/{0}.csv".format(courseid)
    df_uploaded = pd.read_csv(url, sep=",")
    class_start_date =  datetime.strptime(startdate , '%Y-%m-%d').date()
    class_end_date = datetime.strptime(enddate , '%Y-%m-%d').date()
    user = df_uploaded['get_user_ID'].unique()

    Table = {}     
    Learners = {}
    
    for i in user:
        Table[i] = df_uploaded.loc[df_uploaded['get_user_ID'] == i]
        Table[i]['get_user_ID'] = Table[i]['get_user_ID'].astype(str)
        Table[i] = Table[i].replace('不足1s', '00:00:01')
        Table[i]['time_spent'] = pd.to_timedelta(Table[i].time_spent)
        Table[i]['visit_time'] = pd.to_datetime(Table[i].visit_time, format='%Y.%m.%d %H:%M')
        Table[i]['visit_date'] = Table[i]['visit_time'].dt.date
        Table[i] = Table[i].sort_values('visit_time')
        Table[i]['time_spent'] = Table[i]['time_spent'].dt.total_seconds()
        Table[i] = Table[i].groupby(['visit_date']).agg(time_sum = 
                                                         ('time_spent', 'sum'),
                                                         visits_count = 
                                                         ('visit_time', 'count')).reset_index()
        Table[i]['day_difference'] = Table[i].diff(periods=1, axis=0)['visit_date'].fillna(pd.Timedelta(days=0))  
        Table[i]['day_difference'] = Table[i]['day_difference'].apply(lambda x: x.days)
        Table[i]['time_spent_pv'] = Table[i]['time_sum']/Table[i]['visits_count']        
        Learners["LEARNER{0}".format(i)] = []
     
        sum_Q1 = Table[i]['time_sum'].quantile(0.25)
        sum_Q3 = Table[i]['time_sum'].quantile(0.75)
        sum_IQR = sum_Q3 - sum_Q1
        
        before_class = Table[i][(Table[i]['visit_date'] < class_start_date)]['time_sum']
        in_semester = Table[i].loc[(Table[i]['visit_date'] >= class_start_date)
                                   & (Table[i]['visit_date'] <= class_end_date)]['time_sum']
        after_class = Table[i][(Table[i]['visit_date'] > class_end_date)]['time_sum']
        
        if before_class.sum(axis=0) > 0:
            meta = 1
            in_moti_before = before_class.median()
        else: 
            meta = 0
            in_moti_before = 0
        
        if after_class.sum(axis=0) > 0:
            in_moti_after = after_class.median()
        else:
            in_moti_after = 0
                     
        half = round(Table[i].shape[0]/2)
        first_half = Table[i].loc[:half, 'time_sum'] 
        second_half = Table[i].loc[half:, 'time_sum']  
        
        #Six Character Features
        Grit = Table[i]['time_sum'].median() / sum_IQR
        Self_control = Table[i]['time_spent_pv'].median()
        Meta_cog_Self_reg = (meta + (Table[i]['day_difference'].isin(Table[i]['day_difference'].mode()).count()/(Table[i]['day_difference'].nunique()))) / sum_IQR
        Motivation = in_moti_before + in_semester.median() + in_moti_after
        Engagement = Table[i]['time_sum'].median()
        Self_perception = -math.log(1 + abs(first_half.median() - second_half.median()))
                
        Learners["LEARNER{0}".format(i)].extend([str(i), Grit, Self_control, Meta_cog_Self_reg, Motivation, Engagement, Self_perception])    
        
    Learners_col = ['user_ID', 'Grit', 'Self_control', 'Meta_cog_Self_reg', 'Motivation', 'Engagement', 'Self_perception']
    df_Learners = pd.DataFrame(columns = Learners_col)
    for i in user:
        a_series = pd.Series(Learners['LEARNER' + str(i)], index = df_Learners.columns)
        df_Learners = df_Learners.append(a_series, ignore_index=True)
    df_Learners.set_index('user_ID', inplace=True)
    df_Learners = df_Learners.astype(np.float64)
        
    print(df_Learners.isnull().sum().sum())
    print(df_Learners.isin([np.inf, -np.inf]).sum().sum())        
    scaler = MinMaxScaler()    
    df_Learners_scaled = df_Learners[:]
    df_Learners_scaled = scaler.fit_transform(df_Learners_scaled)    
    table = pd.DataFrame(df_Learners_scaled, index = df_Learners.index, 
                                          columns = df_Learners.columns)
    pca = PCA(n_components=PC_n, svd_solver = "full")
    pca_result = pca.fit_transform(table)
    df_Learners_scaled_PCA = table[:]
    for i in range(PC_n):
        df_Learners_scaled_PCA['PC' + str(i + 1)] = pca_result[:, i]
    df_PC = df_Learners_scaled_PCA.iloc[:, -PC_n:]
    kmeans = KMeans(n_clusters=cluster_n)
    kmeans.fit(df_PC)
    y_kmeans = kmeans.predict(df_PC)
    df_Learners_scaled_PCA['cluster'] = y_kmeans.astype(int)
    series_cluster = df_Learners_scaled_PCA.loc[: ,'cluster']
    
    fig, ax = plt.subplots(figsize=(22, 21))  #create a fig with 1 ax
    colors =cluster_coldict_n 
    PSGR=df_Learners_scaled_PCA['cluster'].apply(lambda x: colors[x])   
    x = df_Learners_scaled_PCA['PC1'].astype('float32')
    y = df_Learners_scaled_PCA['PC2'].astype('float32')
    
    for i in range(0, len(table.columns)): 
        ax.arrow(0,
                 0,  
                 pca.components_[0, i],  
                 pca.components_[1, i],  
                 head_width=0.01,
                 head_length=0.01)
        plt.text(pca.components_[0, i] + 0.01,
                 pca.components_[1, i] + 0.01,
                 table.columns[i],
                 fontstyle = 'italic', 
                 fontsize = 'large', 
                 color = 'slategray') 
    ax.spines['left'].set_position('zero')  
    ax.spines['bottom'].set_position('zero')  
    plt.axvline(0)  
    plt.axhline(0)          
    plt.scatter(x, y, c=PSGR, s=150, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='dimgray', s=600, alpha=0.5)
    cluster_list = ["cluster {0}".format(i+1) for i in range(cluster_n)]
    for i, txt in enumerate(cluster_list):
        plt.annotate(txt, (centers[i, 0], centers[i, 1]), color='black', fontsize=20)
    plt.axis('equal')
    plt.close()
        
    return series_cluster, fig


def revised_model(x_table, y_table):
    img_input = layers.Input(shape=(14, 14, 1))
    x = layers.Conv2D(8, 2, activation='relu')(img_input)
    x = layers.MaxPooling2D((2,2), padding='same')(x)

    x = layers.Conv2D(16, 2, activation='relu')(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)

    x = layers.Conv2D(32, 2, activation='relu')(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(4, activation='sigmoid')(x)

    model = Model(img_input, output)
    model.compile(loss='sparse_categorical_crossentropy',    
                  optimizer=RMSprop(lr=0.001), 
                  metrics=['acc'])    
      
    tableX_reshaped = x_table.values.reshape(97, 14, 14, -1)
    tableY_array = y_table.astype(int)
    
    model.fit(tableX_reshaped, tableY_array, batch_size=10, epochs=150)
    return model



if submitted1:
    df_time_sum_minutes_scaled, df_time_sum_log_scaled, df_visits_scaled = x1_x2_x3_table()
    series_cluster, fig1 = y_table_fig(cluster_n = int(cluster_n))

    st.title('Clustering Plot: {0} Clusters'.format(cluster_n))
    st.write(fig1)

    if datatype == 'time spent per day (minutes)':
        x_table = df_time_sum_minutes_scaled
    elif datatype == 'log time spent per day (seconds)':
        x_table = df_time_sum_log_scaled
    elif datatype == 'visits per day':
        x_table = df_visits_scaled

    revised_model_instance = revised_model(x_table = x_table, 
                                       y_table = series_cluster)
    

    def revised_model_predict(series, model = revised_model_instance):
        series = np.array(series)
        prob_list = model.predict(np.expand_dims(series.reshape(14, 14, -1), axis=0))
        return np.squeeze(prob_list)

   
    def Character_Feature(studentid, num_features, x_table, y_table, num_slices = 28): 
        table = x_table[:]
        table['cluster'] = y_table.astype(int)
        explainer = LimeTimeSeriesExplainer(class_names=['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4'])
        exp = explainer.explain_instance(table.loc[studentid].iloc[:-1], revised_model_predict, 
                                        num_features=num_features, num_samples=5000, num_slices=num_slices,
                                    replacement_method='total_mean')    #labels=None, top_labels=4
                                                                        #advice: set top_labels = cluster_n
        cluster = table.loc[studentid].iloc[-1:]
        exp_figure = exp.as_pyplot_figure(label = int(cluster))    #arg: label = desired label
        plt.close()
        print(np.array(exp.available_labels()).shape)
        print(exp.available_labels()[:5])
    
        series = pd.DataFrame(table.loc[studentid].iloc[:-1])
        pd.to_datetime(series.index)
        others = pd.DataFrame(table[(table['cluster'] != int(cluster))].iloc[:, :-1].mean())
        pd.to_datetime(others.index)
        values_per_slice = math.ceil(len(series) / num_slices)
        
        fig, ax = plt.subplots(figsize=(20, 28))
        plt.plot(series, color='cornflowerblue', label='Explained instance')
        plt.plot(others, color='silver', label='Mean of other class')
        plt.legend(loc='lower left')
        for i in range(num_features):
            feature, weight = exp.as_list(label = int(cluster))[i]    #arg: label = desired label
            print([feature, weight])
            start = feature * values_per_slice
            end = start + values_per_slice
            color = 'tomato' if weight < 0 else 'mediumaquamarine' 
            plt.axvspan(others.index[0] + timedelta(days = int(start)) , 
                        others.index[0] + timedelta(days = int(end)), 
                        color=color, alpha=abs(weight))
            ax.annotate(str(feature), (others.index[0] + timedelta(days = int(start)+2), 0.9),
                    color = 'grey', fontsize = 20)
        plt.close()
        return exp_figure, fig


    exp_figure, fig2 = Character_Feature(studentid = studentid, num_features = int(num_features), 
    x_table = x_table, y_table = series_cluster, num_slices = 28)

    st.title('Top {0} Features'.format(num_features))
    st.write(exp_figure)

    st.title('Top {0} Features in Time Series'.format(num_features))
    st.write(fig2)


