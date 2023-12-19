import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from streamlit_option_menu import option_menu
from babel.numbers import format_currency

# Set daily_orders function to return daily_orders_df
def daily_orders(df):
    daily_orders_df = df.resample(rule='D', on='order_date').agg(
        count_order = ('order_id','nunique'), 
        sum_order_value = ('total_order_value','sum')
        ).reset_index()
    
    return daily_orders_df

# Set order_product_category function to return order_by_product_category_df
def order_product_category(df):
    order_by_product_category_df = df.groupby(by="product_category").agg(
        num_of_order = ('order_id','count'), 
        sum_order_value = ('total_order_value', 'sum')
        ).reset_index()
    
    return order_by_product_category_df

# Set count_order_by_time function to return count_time_order and count_date_order
def count_order_by_time(df):
    count_time_order = df.groupby(['day_order','daytime_order']).agg(
        count_order = ('order_id','nunique')
        ).reset_index()
    count_time_order.insert(2, 'time_order', 
                            count_time_order[['day_order', 'daytime_order']].agg(', '.join, axis=1)
                            )

    count_date_order = df.groupby('date_order').agg(
        count_order = ('order_id','nunique')
        ).reset_index()
    
    return count_time_order, count_date_order

# Set count_customers function to return customers_in_cities and customers_in_states
def count_customers(df):
    customers_in_cities = df.groupby(by="customer_city").agg(
        count_order = ('customer_unique_id','nunique')
        ).reset_index()
    
    customers_in_states = df.groupby(by="customer_state").agg(
        count_order = ('customer_unique_id','nunique')
        ).reset_index()
    
    return customers_in_cities, customers_in_states

# Set customers_order function to return count_sum_order
def customers_order(df):
    cust_count_sum_order = df.groupby(by="customer_unique_id").agg(
        count_order = ('order_id','nunique'), 
        sum_order_value = ('total_order_value', 'sum')
        ).reset_index()
    
    return cust_count_sum_order

#Set create_rfm_df function to return rfm_df
def create_rfm_df(df):
    rfm_df = df.groupby(by="customer_unique_id", as_index=False).agg(
    max_order_date = ("order_date", "max"), # get last order date
    frequency = ("order_id", "nunique"), # count total order
    monetary = ("total_order_value", "sum") # count total money for order
    )
    rfm_df['max_order_date'] = rfm_df['max_order_date'].dt.date #change to date format
    recent_order_date = df['order_date'].dt.date.max() #choose last date from order_date column
    rfm_df.insert(1,'recency', rfm_df['max_order_date'].apply(lambda x: (recent_order_date - x).days)) #calculate different days from last order date
    rfm_df.drop('max_order_date', axis=1, inplace=True) #drop unnecessary column

    rfm_df['R_rank'] = rfm_df['recency'].rank(ascending=False) #less recency, better rank
    rfm_df['F_rank'] = rfm_df['frequency'].rank(ascending=True) #more frequency, better rank
    rfm_df['M_rank'] = rfm_df['monetary'].rank(ascending=True) #more monetary, better rank

    #Normalize ranking of customers
    rfm_df['R_rank_norm'] = (rfm_df['R_rank']/rfm_df['R_rank'].max())*100
    rfm_df['F_rank_norm'] = (rfm_df['F_rank']/rfm_df['F_rank'].max())*100
    rfm_df['M_rank_norm'] = (rfm_df['F_rank']/rfm_df['M_rank'].max())*100

    rfm_df.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)

    #Create RFM Score with weighted value
    rfm_df['RFM_Score'] = 0.15*rfm_df['R_rank_norm'] + 0.28 *rfm_df['F_rank_norm'] + 0.57*rfm_df['M_rank_norm'] #Weighting to each parameter
    rfm_df['RFM_Score'] = (0.05*rfm_df['RFM_Score']).round(2) #Change RFM Score to value with max is 5 and round it until 2 desimal

    rfm_df = rfm_df[['customer_unique_id', 'recency', 'frequency', 'monetary', 'RFM_Score']]

    #Give rating to customer based on RFM Score
    '''
    RFM Score > 4.5 : Top Customer
    4.5 > RFM Score > 4 : High Value Customer
    4> RFM Score > 3 : Medium Value Customer
    3> RFM Score > 1.6 : Low Value Customer
    RFM Score <1.6 : Lost Customer
    '''
    rfm_df["customer_segment"] = np.where(
        rfm_df['RFM_Score'] > 4.5, "Top Customer", (np.where(
            rfm_df['RFM_Score'] > 4, "High Value Customer",(np.where(
                rfm_df['RFM_Score'] > 3, "Medium Value Customer", np.where(
                    rfm_df['RFM_Score'] > 1.6, 'Low Value Customer', 'Lost Customer')))))
    )

    #Define categorical order
    segment_order = ["Lost Customer", "Low Value Customer", "Medium Value Customer", "High Value Customer", "Top Customer"]

    #Change customer_segment column to categorical data type which have ordered
    rfm_df['customer_segment'] = pd.Categorical(rfm_df['customer_segment'], categories=segment_order, ordered=True)
    
    return rfm_df

# Set count_sellers function to return sellers_in_cities and sellers_in_states
def count_sellers(df):
    sellers_in_cities = df.groupby(by="seller_city").agg(
        count_order = ('seller_id','nunique')
        ).reset_index()
    
    sellers_in_states = df.groupby(by="seller_state").agg(
        count_order = ('seller_id','nunique')
        ).reset_index()
    
    return sellers_in_cities, sellers_in_states

# Set customers_order function to return count_sum_order
def sellers_order(df):
    seller_count_sum_order = df.groupby(by="seller_id").agg(
        count_order = ('order_id','nunique'), 
        sum_order_value = ('total_order_value', 'sum')
        ).reset_index()
    
    return seller_count_sum_order

#Set palette colors and one palette for max values
colors=["#3187d4",'#b3bcc4','#b3bcc4','#b3bcc4','#b3bcc4','#b3bcc4','#b3bcc4','#b3bcc4','#b3bcc4','#b3bcc4']
def set_pal_max_value(series, max_color = '#3187d4', other_color = '#b3bcc4'):
    max_value = series.max()
    palette = []

    for item in series:
        if item == max_value:
            palette.append(max_color)
        else:
            palette.append(other_color)
    return palette

################################### ORDERS ###################################
def orders_analysis():
    daily_orders_df = daily_orders(main_df)
    order_by_product_category_df = order_product_category(main_df)
    count_time_order, count_date_order = count_order_by_time(main_df)

    #Count Orders and Total Order Value per Day
    st.subheader("Daily Orders")

    col1, col2 = st.columns(2)
 
    with col1:
        total_orders = daily_orders_df.count_order.sum()
        st.metric("Total Orders", value=total_orders)
 
    with col2:
        total_order_value = format_currency(daily_orders_df.sum_order_value.sum(), "R$", locale='pt_BR') 
        st.metric("Total Order Value", value=total_order_value)

    fig, ax = plt.subplots(figsize=(25, 10))
    ax.plot(daily_orders_df["order_date"],
            daily_orders_df["count_order"],
            marker='o', 
            linewidth=3,
            color= "#3187d4"
            )
    ax.set_title("Count Order per Day", loc="center", fontsize=30, pad=20)
    ax.tick_params(axis='y', labelsize=25)
    ax.tick_params(axis='x', labelsize=20)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(25, 10))
    ax.plot(daily_orders_df["order_date"],
            daily_orders_df["sum_order_value"],
            marker='o', 
            linewidth=3,
            color= "#3187d4"
            )
    ax.set_title("Total Order Value per Day", loc="center", fontsize=30, pad=20)
    ax.set_ylabel("R$", fontsize=20)
    ax.tick_params(axis='y', labelsize=25)
    ax.tick_params(axis='x', labelsize=20)
    st.pyplot(fig)

    #Best and Worst Performing Product Category
    st.subheader("Best and Worst Performing Product Category")
 
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25,10))

    sns.barplot(
        x="num_of_order",
        y="product_category",
        data= order_by_product_category_df.sort_values('num_of_order', ascending=False).head(10),
        palette= colors,
        ax=ax[0]
        )
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].set_title("Highest Number Ordered", loc="center", fontsize=20)
    ax[0].tick_params(axis ='y', labelsize=18)
    ax[0].tick_params(axis ='x', labelsize=18)

    sns.barplot(
        x="num_of_order",
        y="product_category",
        data= order_by_product_category_df.sort_values(by=['num_of_order','sum_order_value'], ascending=True).head(10),
        palette=colors,
        ax=ax[1]
        )
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].invert_xaxis()
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].set_title("Lowest Number Ordered", loc="center", fontsize=20)
    ax[1].tick_params(axis='y', labelsize=18)
    ax[1].tick_params(axis='x', labelsize=18)
    plt.suptitle("Best and Worst Performing Product Category by Number Ordered", fontsize=25)

    st.pyplot(fig)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25,10))

    sns.barplot(
        x="sum_order_value",
        y="product_category",
        data= order_by_product_category_df.sort_values('sum_order_value', ascending=False).head(10),
        palette= colors,
        ax=ax[0]
        )
    ax[0].set_ylabel(None)
    ax[0].set_xlabel('Total Order Value (R$)', fontsize=15)
    ax[0].set_title("Highest Total Order Value", loc="center", fontsize=20)
    ax[0].tick_params(axis ='y', labelsize=18)
    ax[0].tick_params(axis ='x', labelsize=18)

    sns.barplot(
        x="sum_order_value",
        y="product_category",
        data= order_by_product_category_df.sort_values('sum_order_value', ascending=True).head(10),
        palette= colors,
        ax=ax[1]
        )
    ax[1].set_ylabel(None)
    ax[1].set_xlabel('Total Order Value (R$)', fontsize=15)
    ax[1].invert_xaxis()
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].set_title("Lowest Total Order Value", loc="center", fontsize=20)
    ax[1].tick_params(axis='y', labelsize=18)
    ax[1].tick_params(axis='y', labelsize=18)
    plt.suptitle("Best and Worst Performing Product Category by Total Order Value", fontsize=25)

    st.pyplot(fig)

    #Number Ordered by Time
    st.subheader("Number Ordered by Time")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 10))

    sns.barplot(x="time_order",
                y="count_order",
                data= count_time_order,
                palette= set_pal_max_value(count_time_order.count_order),
                ax=ax[0])
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].set_title("Based on the Day and Time of Order", loc="center", fontsize=20)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=15)
    ax[0].tick_params(axis ='y', labelsize=18)
    ax[0].tick_params(axis ='x', labelsize=18)

    sns.barplot(x="date_order",
                y="count_order",
                data= count_date_order,
                palette= set_pal_max_value(count_date_order.count_order),
                ax=ax[1])
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].set_title("Based on Date of Order", loc="center", fontsize=20)
    ax[1].tick_params(axis ='y', labelsize=18)
    ax[1].tick_params(axis ='x', labelsize=18)

    plt.suptitle("Number of Customer Orders Based on Time and Date of Order", fontsize=25)

    st.pyplot(fig)

################################### CUSTOMERS ###################################
def customers_analysis():
    customers_in_cities, customers_in_states = count_customers(main_df)
    cust_count_sum_order = customers_order(main_df)
    rfm_df = create_rfm_df(main_df)

    #Distribution of Customers by City and State
    st.subheader("Distribution of Customers by City and State")
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))

    sns.barplot(x="customer_city", 
                y="count_order", 
                data= customers_in_cities.sort_values('count_order', ascending=False).head(10), 
                palette= colors, 
                ax=ax[0])
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].tick_params(axis='x', labelrotation=45)
    ax[0].set_title("Based on City", loc="center", fontsize=18)
    ax[0].tick_params(axis ='y', labelsize=15)

    sns.barplot(x="customer_state", 
                y="count_order", 
                data= customers_in_states.sort_values('count_order', ascending=False).head(10),
                palette= colors, 
                ax=ax[1])
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].tick_params(axis='x', labelrotation=45)
    ax[1].set_title("Based on State", loc="center", fontsize=18)
    ax[1].tick_params(axis='y', labelsize=15)

    plt.suptitle("Distribution of Number of Customers by City and State", fontsize=20)
    st.pyplot(fig)

    #Customer with Largest Order
    st.subheader("Customer with Largest Number of Order and Total Order Value")
    tab1, tab2 = st.tabs(['Count Order','Total Order Value'])
 
    with tab1:
        fig, ax = plt.subplots(figsize=(25, 10))
        sns.barplot(x="count_order", 
                y="customer_unique_id", 
                data= cust_count_sum_order.sort_values('count_order',ascending=False), 
                palette= colors)
        ax.set_ylabel('Customer Unique ID', fontsize=18)
        ax.set_xlabel('Number of Order', fontsize=18)
        ax.set_title("Customer with Largest Number of Order", loc="center", fontsize=20)
        ax.bar_label(ax.containers[0], label_type='center')
        ax.tick_params(axis ='y', labelsize=15)
        st.pyplot(fig)
 
    with tab2:
        fig, ax = plt.subplots(figsize=(25, 10))

        sns.barplot(x="sum_order_value", 
                    y="customer_unique_id", 
                    data= cust_count_sum_order.sort_values('sum_order_value',ascending=False), 
                    palette= colors)
        ax.set_ylabel('Customer Unique ID', fontsize=18)
        ax.set_xlabel('Total Order Value (R$)', fontsize=18)
        ax.set_title("Customer with Largest Total Order Value", loc="center", fontsize=20)
        ax.bar_label(ax.containers[0], label_type='center')
        ax.tick_params(axis ='y', labelsize=15)
        st.pyplot(fig)

    #RFM Analysis
    st.subheader("RFM Analysis")

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(32, 8))

    sns.barplot(x="customer_unique_id", 
                y="recency", 
                data= rfm_df.sort_values(by='recency', ascending=True).head(10), 
                palette=colors, 
                ax=ax[0])
    ax[0].set_ylabel('Hari', fontsize=12)
    ax[0].set_xlabel(None)
    ax[0].set_title("Based on Recency", loc="center", fontsize=16)
    ax[0].tick_params(axis ='y', labelsize=15)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=14)

    sns.barplot(x="customer_unique_id", 
                y="frequency", 
                data= rfm_df.sort_values(by='frequency', ascending=False).head(10), 
                palette=colors, 
                ax=ax[1])
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].set_title("Based on Frequency", loc="center", fontsize=16)
    ax[1].tick_params(axis='y', labelsize=15)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=14)

    sns.barplot(x="customer_unique_id", 
                y="monetary", 
                data= rfm_df.sort_values(by='monetary', ascending=False).head(10), 
                palette=colors, 
                ax=ax[2])
    ax[2].set_ylabel('R$', fontsize=12)
    ax[2].set_xlabel(None)
    ax[2].set_title("Based on Monetary", loc="center", fontsize=16)
    ax[2].tick_params(axis ='y', labelsize=15)
    ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=14)

    plt.suptitle("Best Customer Based on Each Parameter of RFM", fontsize=20)
    st.pyplot(fig)

    #Customer Segmentation Based on RFM Score
    plt.figure(figsize=(6,8))
    plt.pie(
        rfm_df.customer_segment.value_counts(),
        labels= rfm_df.customer_segment.value_counts().index,
        autopct= '%1.2f%%',
        explode = [0.3, 0.5, 0],
        colors= sns.color_palette('Set2')
        )
    plt.title("Customer Segmentation Based on RFM Score", loc='center', fontsize=16)
    st.pyplot(fig)

    #Detailed Data RFM Analysis
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_sortby= st.selectbox('Sort by:', ['frequency', 'recency', 'monetary', 'customer_segment'])
        #Make radio button to select sort order
        sort_order = st.radio("Sort Order:", ["Ascending", "Descending"])
               
        #Sorting DataFrame based on selected category and sorting order
        ascending_order = (sort_order == "Ascending")
        sorted_rfm_df = rfm_df.sort_values(by=selected_sortby, ascending=ascending_order)
        
        #Make a slider for filter dataframe based on RFM Score value
        min_score_value = rfm_df.RFM_Score.min()
        max_score_value = rfm_df.RFM_Score.max()
        
        min_score, max_score = st.slider(
            'RFM Score Range', 
            min_value=0.00,
            max_value=5.00,
            value= [min_score_value, max_score_value]
            )
        
        filtered_rfm_df = sorted_rfm_df.loc[(sorted_rfm_df.RFM_Score >= min_score) & 
                                            (sorted_rfm_df.RFM_Score <= max_score)]
        st.markdown(
            """
            RFM > 4.5  : Top Customer
            RFM > 4    : High Value Customer
            RFM > 3    : Medium Value Customer
            RFM > 1.6  : Low Value Customer
            RFM <= 1.6 : Lost Customer
            """
            )
        with col2:
            st.write("Result: ")
            st.dataframe(filtered_rfm_df, use_container_width=True)

################################### SELLERS ###################################
def sellers_analysis():
    sellers_in_cities, sellers_in_states = count_sellers(main_df)
    seller_count_sum_order = sellers_order(main_df)

    #Distribution of Sellers by City and State
    st.subheader("Distribution of Sellers by City and State")
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))

    sns.barplot(x="seller_city", 
                y="count_order", 
                data= sellers_in_cities.sort_values('count_order', ascending=False).head(10), 
                palette= colors, 
                ax=ax[0])
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].tick_params(axis='x', labelrotation=45)
    ax[0].set_title("Based on City", loc="center", fontsize=18)
    ax[0].tick_params(axis ='y', labelsize=15)

    sns.barplot(x="seller_state", 
                y="count_order", 
                data= sellers_in_states.sort_values('count_order', ascending=False).head(10),
                palette= colors, 
                ax=ax[1])
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].tick_params(axis='x', labelrotation=45)
    ax[1].set_title("Based on State", loc="center", fontsize=18)
    ax[1].tick_params(axis='y', labelsize=15)

    plt.suptitle("Distribution of Number of Sellers by City and State", fontsize=20)
    st.pyplot(fig)

    #Seller with Largest Order
    st.subheader("Seller with Largest Number of Order and Total Order Value")
    tab1, tab2 = st.tabs(['Count Order','Total Order Value'])
 
    with tab1:
        fig, ax = plt.subplots(figsize=(25, 10))
        sns.barplot(x="count_order", 
                    y="seller_id", 
                    data= seller_count_sum_order.sort_values('count_order',ascending=False), 
                    palette= colors)
        ax.set_ylabel('Seller ID', fontsize=18)
        ax.set_xlabel('Number of Order', fontsize=18)
        ax.set_title("Seller with Largest Number of Order", loc="center", fontsize=20)
        ax.bar_label(ax.containers[0], label_type='center')
        ax.tick_params(axis ='y', labelsize=15)
        st.pyplot(fig)
 
    with tab2:
        fig, ax = plt.subplots(figsize=(25, 10))

        sns.barplot(x="sum_order_value", 
                    y="seller_id", 
                    data= seller_count_sum_order.sort_values('sum_order_value',ascending=False), 
                    palette= colors)
        ax.set_ylabel('Seller ID', fontsize=18)
        ax.set_xlabel('Total Order Value (R$)', fontsize=18)
        ax.set_title("Seller with Largest Total Order Value", loc="center", fontsize=20)
        ax.bar_label(ax.containers[0], label_type='center')
        ax.tick_params(axis ='y', labelsize=15)
        st.pyplot(fig)

#Load data from cleaned dataframe
main_df = pd.read_csv('E:\Dicoding Data Scientist\main_data_for_dashboard.csv')

dt_columns = ['order_date', 'approved_date', 'shipped_date', 'delivery_date']
main_df.sort_values(by="order_date", inplace=True)
main_df.reset_index(inplace=True)

for column in dt_columns:
    main_df[column] = pd.to_datetime(main_df[column])

#Set min_date and max_date for filter data
min_date = main_df["order_date"].min()
max_date = main_df["order_date"].max()

with st.sidebar:
    #Add company brand
    st.image("https://companyurlfinder.com/marketing/assets/img/logos/olist.com.png")
    
    #Make start_date & end_date from date_input
    start_date, end_date = st.date_input(
        label='Date Range', 
        min_value=min_date,
        max_value=max_date,
        value= [min_date, max_date]
        )

main_df = main_df[(main_df["order_date"] >= str(start_date)) & 
                (main_df["order_date"] <= str(end_date))]

#Make function with radio button on sidebar to call analysis function
def sidebar_function():
    with st.sidebar:
        selected= option_menu(
            menu_title= "Analyze What?",
            options=["Orders","Customers","Sellers"],
            icons=["cart-fill","people-fill","shop-window"],
            menu_icon="clipboard-data-fill",
            default_index=0
            )

    if selected =="Orders":
        orders_analysis()
    if selected=="Customers":
        customers_analysis()
    if selected=="Sellers":
        sellers_analysis()
sidebar_function()

st.sidebar.caption('Copyright © Ilham Fadilah - 2023')