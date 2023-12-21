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

# Set count_customers function to return customers_in_cities and customers_in_states
def count_customers(df):
    customers_in_cities = df.groupby(by="customer_city").agg(
        count_customer = ('customer_unique_id','nunique')
        ).reset_index()
    
    customers_in_states = df.groupby(by="customer_state").agg(
        count_customer = ('customer_unique_id','nunique')
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
    rfm_df["customer_segment"] = np.where(
        rfm_df['RFM_Score'] > 4.5, "Top Customer", (np.where(
            rfm_df['RFM_Score'] > 4, "High Value Customer",(np.where(
                rfm_df['RFM_Score'] > 3, "Medium Value Customer", np.where(
                    rfm_df['RFM_Score'] > 1.6, 'Low Value Customer', 'Lost Customer')))))
    )

    return rfm_df

# Set count_sellers function to return sellers_in_cities and sellers_in_states
def count_sellers(df):
    sellers_in_cities = df.groupby(by="seller_city").agg(
        count_seller = ('seller_id','nunique')
        ).reset_index()
    
    sellers_in_states = df.groupby(by="seller_state").agg(
        count_seller = ('seller_id','nunique')
        ).reset_index()
    
    return sellers_in_cities, sellers_in_states

# Set customers_order function to return count_sum_order
def sellers_order(df):
    seller_count_sum_order = df.groupby(by="seller_id").agg(
        count_order = ('order_id','nunique'), 
        sum_order_value = ('total_order_value', 'sum')
        ).reset_index()
    
    return seller_count_sum_order

#Set palette colors
colors=["#3187d4",'#b3bcc4','#b3bcc4','#b3bcc4','#b3bcc4','#b3bcc4','#b3bcc4','#b3bcc4','#b3bcc4','#b3bcc4']

#Load data from cleaned dataframe
main_df = pd.read_csv('https://raw.githubusercontent.com/hamf99/ecommerce-public-dataset/main/dashboard/main_data_for_dashboard.csv')

#Looping for change column to datetime type
dt_columns = ['order_date', 'approved_date', 'shipped_date', 'delivery_date']
main_df.sort_values(by="order_date", inplace=True)
main_df.reset_index(inplace=True)

for column in dt_columns:
    main_df[column] = pd.to_datetime(main_df[column])

#Set min_date and max_date for filter data
min_date = main_df["order_date"].min()
max_date = main_df["order_date"].max()

#Add sidebar
with st.sidebar:
    #Add company brand
    st.image("https://seeklogo.com/images/O/olist-logo-9DCE4443F8-seeklogo.com.png")
    
    #Make start_date & end_date from date_input
    start_date, end_date = st.date_input(
        label='Date Range', 
        min_value=min_date,
        max_value=max_date,
        value= [min_date, max_date]
        )

#Add one day to end_date and subs
start_date = start_date - pd.DateOffset(days=1)
end_date = end_date + pd.DateOffset(days=1)

#Filtered main_df by start_date and end_date
main_df = main_df[(main_df["order_date"] >= start_date) & 
                (main_df["order_date"] <= end_date)]

# Make the title dashboard
st.markdown('<h1 style="text-align: center;">Ecommerce Olist Analysis</h1>', unsafe_allow_html=True)

################################### ORDERS ###################################
def orders_analysis():
    daily_orders_df = daily_orders(main_df)
    order_by_product_category_df = order_product_category(main_df)

    #Count Orders and Total Order Value per Day
    st.subheader("Daily Orders")

    col1, col2 = st.columns(2)
    with col1:
        total_orders = daily_orders_df.count_order.sum()
        st.metric("Total Orders", value=total_orders)
 
    with col2:
        total_order_value = format_currency(daily_orders_df.sum_order_value.sum(), "R$", locale='pt_BR') 
        st.metric("Total Order Value", value=total_order_value)
    
    xmax = daily_orders_df.order_date[np.argmax(daily_orders_df.count_order)]
    ymax = daily_orders_df.count_order.max()

    fig, ax = plt.subplots(figsize=(25, 10))
    ax.plot(daily_orders_df["order_date"],
            daily_orders_df["count_order"],
            marker='o', 
            linewidth=3,
            color= "#3187d4"
            )
    ax.set_title("Number of Order per Day", loc="center", fontsize=30, pad=20)
    ax.tick_params(axis='y', labelsize=25)
    ax.tick_params(axis='x', labelsize=20)
    ax.annotate(f"At {xmax.strftime('%Y-%m-%d')}\n have {ymax} orders", 
                xy=(xmax, ymax), xycoords='data',
                xytext=(xmax + (end_date - start_date)/6, ymax), #Give annotate with 1:6 scale of x
                textcoords='data', size=20, va="center", ha="center",
                bbox=dict(boxstyle="round4", fc="w"),
                arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.2", fc="w")
                )
    st.pyplot(fig)

    x_max = daily_orders_df.order_date[np.argmax(daily_orders_df.sum_order_value)]
    y_max = round(daily_orders_df.sum_order_value.max(), 2)
    
    fig, ax = plt.subplots(figsize=(25, 10))
    ax.plot(daily_orders_df["order_date"],
            daily_orders_df["sum_order_value"],
            marker='o', 
            linewidth=3,
            color= "#3187d4"
            )
    ax.set_title("Total Order Value per Day", loc="center", fontsize=30, pad=20)
    ax.set_ylabel("R$", fontsize=20, labelpad=10)
    ax.tick_params(axis='y', labelsize=25)
    ax.tick_params(axis='x', labelsize=20)
    ax.annotate(f"At {x_max.strftime('%Y-%m-%d')}\n have value\n R$ {y_max}", 
            xy=(x_max, y_max), xycoords='data',
            xytext=(x_max + (end_date - start_date)/6, y_max), #Give annotate with 1:6 scale of x
            textcoords='data', size=20, va="center", ha="center",
            bbox=dict(boxstyle="round4", fc="w"),
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.2", fc="w")
            )
    st.pyplot(fig)

    #Best and Worst Performing Product Category
    st.subheader("Best and Worst Performing Product Category")
 
    tab1, tab2 = st.tabs(['Count Order', 'Total Order Value'])
    
    with tab1:
        col1, col2 = st.columns(2)
 
        with col2:
            min_order = order_by_product_category_df.num_of_order.min()
            st.metric("Lowest Number of Order by Product Category", value=min_order)
 
        with col1:
            max_order = order_by_product_category_df.num_of_order.max()
            st.metric("Highest Number of Order by Product Category", value=max_order)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25,10))

        sns.barplot(
            x="num_of_order",
            y="product_category",
            data= order_by_product_category_df.sort_values('num_of_order', ascending=False).head(10),
            palette= colors,
            ax=ax[0]
            )
        ax[0].set_ylabel(None)
        ax[0].set_xlabel('Number of Order', fontsize=15, labelpad=10)
        ax[0].set_title("Highest Number Ordered", loc="center", fontsize=20, pad=10)
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
        ax[1].set_xlabel('Number of Order', fontsize=15, labelpad=10)
        ax[1].invert_xaxis()
        ax[1].yaxis.set_label_position("right")
        ax[1].yaxis.tick_right()
        ax[1].set_title("Lowest Number Ordered", loc="center", fontsize=20, pad=10)
        ax[1].tick_params(axis='y', labelsize=18)
        ax[1].tick_params(axis='x', labelsize=18)
        plt.suptitle("Best and Worst Performing Product Category by Number Ordered", fontsize=25)

        st.pyplot(fig)

    with tab2:
        col1, col2 = st.columns(2)
 
        with col1:
            max_order_value = format_currency(order_by_product_category_df.sum_order_value.max(), "R$", locale='pt_BR')
            st.metric("Highest Total Order Value by Product Category", value=max_order_value)

        with col2:
            min_order_value = format_currency(order_by_product_category_df.sum_order_value.min(), "R$", locale='pt_BR')
            st.metric("Lowest Total Order Value by Product Category", value=min_order_value)
 
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25,10))

        sns.barplot(
            x="sum_order_value",
            y="product_category",
            data= order_by_product_category_df.sort_values('sum_order_value', ascending=False).head(10),
            palette= colors,
            ax=ax[0]
            )
        ax[0].set_ylabel(None)
        ax[0].set_xlabel('Total Order Value (Million R$)', fontsize=15, labelpad=10)
        ax[0].set_title("Highest Total Order Value", loc="center", fontsize=20, pad=10)
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
        ax[1].set_xlabel('Total Order Value (R$)', fontsize=15, labelpad=10)
        ax[1].invert_xaxis()
        ax[1].yaxis.set_label_position("right")
        ax[1].yaxis.tick_right()
        ax[1].set_title("Lowest Total Order Value", loc="center", fontsize=20, pad=10)
        ax[1].tick_params(axis='y', labelsize=18)
        ax[1].tick_params(axis='x', labelsize=18)
        plt.suptitle("Best and Worst Performing Product Category by Total Order Value", fontsize=25)

        st.pyplot(fig)

################################### CUSTOMERS ###################################
def customers_analysis():
    customers_in_cities, customers_in_states = count_customers(main_df)
    cust_count_sum_order = customers_order(main_df)
    rfm_df = create_rfm_df(main_df)

    #Distribution of Customers by City and State
    st.subheader("Distribution of Customers by City and State")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_count_cust = main_df.customer_unique_id.nunique()
        st.metric("Total Number of Customers", value=total_count_cust)

    with col2:
        highest_count_cust_city = customers_in_cities.count_customer.max()
        st.metric("Highest by City", value=highest_count_cust_city)

    with col3:
        highest_count_cust_state = customers_in_states.count_customer.max()
        st.metric("Highest by State", value=highest_count_cust_state)
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))

    sns.barplot(x="customer_city", 
                y="count_customer", 
                data= customers_in_cities.sort_values('count_customer', ascending=False).head(10), 
                palette= colors, 
                ax=ax[0])
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].tick_params(axis='x', labelrotation=45)
    ax[0].set_title("Based on City", loc="center", fontsize=18, pad=10)
    ax[0].tick_params(axis ='y', labelsize=15)
    ax[0].tick_params(axis ='x', labelsize=15)

    sns.barplot(x="customer_state", 
                y="count_customer", 
                data= customers_in_states.sort_values('count_customer', ascending=False).head(10),
                palette= colors, 
                ax=ax[1])
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].tick_params(axis='x', labelrotation=45)
    ax[1].set_title("Based on State", loc="center", fontsize=18, pad=10)
    ax[1].tick_params(axis='y', labelsize=15)
    ax[1].tick_params(axis ='x', labelsize=15)

    plt.suptitle("Distribution of Number of Customers by City and State", fontsize=20)
    st.pyplot(fig)

    #Customer with Largest Order
    st.subheader("Customer with Largest Number of Order and Total Order Value")
    tab1, tab2 = st.tabs(['Count Order','Total Order Value'])
 
    with tab1:
        col1, col2, col3 = st.columns(3)
 
        with col1:
            max_cust_count_order = cust_count_sum_order.count_order.max()
            st.metric("Highest Number of Order", value=max_cust_count_order)

        with col2:
            min_cust_count_order = cust_count_sum_order.count_order.min()
            st.metric("Lowest Number of Order", value=min_cust_count_order)

        with col3:
            avg_cust_count_order = cust_count_sum_order.count_order.mean().astype(int)
            st.metric("Average Number of Order", value=avg_cust_count_order)
        
        fig, ax = plt.subplots(figsize=(25, 10))
        sns.barplot(x="count_order", 
                y="customer_unique_id", 
                data= cust_count_sum_order.sort_values('count_order',ascending=False).head(10), 
                palette= colors,
                ax=ax)
        ax.set_ylabel('Customer Unique ID', fontsize=18, labelpad=10)
        ax.set_xlabel('Number of Order', fontsize=18, labelpad=10)
        ax.set_title("Customer with Largest Number of Order", loc="center", fontsize=20, pad=10)
        ax.bar_label(ax.containers[0], label_type='center')
        ax.tick_params(axis ='y', labelsize=15)
        ax.tick_params(axis ='x', labelsize=15)
        st.pyplot(fig)
 
    with tab2:
        col1, col2, col3 = st.columns(3)
 
        with col1:
            max_cust_order_value = format_currency(cust_count_sum_order.sum_order_value.max(), "R$", locale='pt_BR')
            st.metric("Highest Total Order Value", value=max_cust_order_value)

        with col2:
            min_cust_order_value = format_currency(cust_count_sum_order.sum_order_value.min(), "R$", locale='pt_BR')
            st.metric("Lowest Total Order Value", value=min_cust_order_value)
        
        with col3:
            avg_cust_order_value = format_currency(cust_count_sum_order.sum_order_value.mean(), "R$", locale='pt_BR')
            st.metric("Average Total Order Value", value=avg_cust_order_value)
        
        fig, ax = plt.subplots(figsize=(25, 10))

        sns.barplot(x="sum_order_value", 
                    y="customer_unique_id", 
                    data= cust_count_sum_order.sort_values('sum_order_value',ascending=False).head(10), 
                    palette= colors,
                    ax=ax)
        ax.set_ylabel('Customer Unique ID', fontsize=18, labelpad=10)
        ax.set_xlabel('Total Order Value (R$)', fontsize=18, labelpad=10)
        ax.set_title("Customer with Largest Total Order Value", loc="center", fontsize=20, pad=10)
        ax.bar_label(ax.containers[0], label_type='center')
        ax.tick_params(axis ='y', labelsize=15)
        ax.tick_params(axis ='x', labelsize=15)
        st.pyplot(fig)

    #RFM Analysis
    st.subheader("RFM Analysis")

    col1, col2, col3 = st.columns(3)
    with col1:
        avg_recency = (rfm_df.recency.mean()).astype(int)
        st.metric("Average Recency (Days)", value=avg_recency)

    with col2:
        avg_frequency = (rfm_df.frequency.mean()).astype(int)
        st.metric("Average Frequency", value=avg_frequency)

    with col3:
        avg_monetary = format_currency(rfm_df.monetary.mean(), "R$", locale='pt_BR')
        st.metric("Average Monetary", value=avg_monetary)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(36, 12))

    sns.barplot(x="customer_unique_id", 
                y="recency", 
                data= rfm_df.sort_values(by='recency', ascending=True).head(10), 
                palette=colors, 
                ax=ax[0])
    ax[0].set_ylabel('Days', fontsize=14)
    ax[0].set_xlabel(None)
    ax[0].set_title("Based on Recency", loc="center", fontsize=16)
    ax[0].tick_params(axis ='y', labelsize=15)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=15)

    sns.barplot(x="customer_unique_id", 
                y="frequency", 
                data= rfm_df.sort_values(by='frequency', ascending=False).head(10), 
                palette=colors, 
                ax=ax[1])
    ax[1].set_ylabel('Number of Order', fontsize=14)
    ax[1].set_xlabel(None)
    ax[1].set_title("Based on Frequency", loc="center", fontsize=16)
    ax[1].tick_params(axis='y', labelsize=15)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=15)

    sns.barplot(x="customer_unique_id", 
                y="monetary", 
                data= rfm_df.sort_values(by='monetary', ascending=False).head(10), 
                palette=colors, 
                ax=ax[2])
    ax[2].set_ylabel('R$', fontsize=14)
    ax[2].set_xlabel(None)
    ax[2].set_title("Based on Monetary", loc="center", fontsize=16)
    ax[2].tick_params(axis ='y', labelsize=15)
    ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=15)

    plt.suptitle("Best Customer Based on Each Parameter of RFM", fontsize=25)
    st.pyplot(fig)

    #Customer Segmentation Based on RFM Score

    st.subheader("Customer Segmentation Based on RFM Score")

    fig, ax = plt.subplots(figsize=(6,8))
    ax.pie(rfm_df.customer_segment.value_counts(), 
           labels= rfm_df.customer_segment.value_counts().index,
           autopct='%1.2f%%',
           explode= [0.3,0.4,0.2],
           colors= sns.color_palette('Set2')
           )
    st.pyplot(fig)

    #Detailed Data RFM Analysis
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_sortby= st.selectbox('Sort by:', ['recency','frequency', 'monetary', 'RFM_Score'])
        #Make radio button to select sort order
        sort_order = st.radio("Sort Order:", ["Ascending", "Descending"])
               
        #Sorting DataFrame based on selected category and sorting order
        ascending_order = (sort_order == "Ascending")
        sorted_rfm_df = rfm_df.sort_values(by=selected_sortby, ascending=ascending_order)
        
        #Make a slider for filter dataframe based on RFM Score value
        min_score_value = rfm_df.RFM_Score.min()
        max_score_value = rfm_df.RFM_Score.max()
        
        min_score, max_score = st.slider(
            'RFM Score Range:', 
            min_value=0.00,
            max_value=5.00,
            value= [min_score_value, max_score_value]
            )
        
        filtered_rfm_score_df = sorted_rfm_df[(sorted_rfm_df.RFM_Score >= min_score) & 
                                              (sorted_rfm_df.RFM_Score <= max_score)]
        
        
    with col2:
        st.write('Filtering by Costumer Segment:')
        cat_list = filtered_rfm_score_df.customer_segment.unique()
        val = [None]* len(cat_list) # this list will store info about which category is selected
        for i, cat in enumerate(cat_list):
        # create a checkbox for each category
            val[i] = st.checkbox(cat, value=True) # value is the preselect value for first render

        filtered_rfm_df = filtered_rfm_score_df[filtered_rfm_score_df.customer_segment.isin(cat_list[val])].reset_index(drop=True)

        st.write('Costumer Segment by RFM Score:')
        st.markdown("""
                    <div style="background-color: #edebeb; padding: 8px; border-radius: 8px;">
                    <p>RFM Score > 4.5  : Top Customers</p>
                    <p>RFM Score > 4    : High Value Customer</p>
                    <p>RFM Score > 3    : Medium Value Customers</p>
                    <p>RFM Score > 1.6  : Low Value Customers</p>
                    <p>RFM Score < 1.6  : Lost Customers</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                    )

    st.write("Result: ")
    #Print the dataframe if its has shape, otherwise give no results
    if filtered_rfm_df.shape[0]> 0:
        st.write(f"This dataframe have {filtered_rfm_df.shape[0]} rows")
        st.dataframe(filtered_rfm_df)
    else:
        st.write("No Results")
            
################################### SELLERS ###################################
def sellers_analysis():
    sellers_in_cities, sellers_in_states = count_sellers(main_df)
    seller_count_sum_order = sellers_order(main_df)

    #Distribution of Sellers by City and State
    st.subheader("Distribution of Sellers by City and State")
    col1, col2, col3 = st.columns(3)
    with col1:
        total_count_seller = main_df.seller_id.nunique()
        st.metric("Total Number of Sellers", value=total_count_seller)

    with col2:
        highest_count_seller_city = sellers_in_cities.count_seller.max()
        st.metric("Highest by City", value=highest_count_seller_city)

    with col3:
        highest_count_seller_state = sellers_in_states.count_seller.max()
        st.metric("Highest by State", value=highest_count_seller_state)
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))

    sns.barplot(x="seller_city", 
                y="count_seller", 
                data= sellers_in_cities.sort_values('count_seller', ascending=False).head(10), 
                palette= colors, 
                ax=ax[0])
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].tick_params(axis='x', labelrotation=45)
    ax[0].set_title("Based on City", loc="center", fontsize=18, pad=10)
    ax[0].tick_params(axis ='y', labelsize=15)
    ax[0].tick_params(axis='x', labelsize=15)

    sns.barplot(x="seller_state", 
                y="count_seller", 
                data= sellers_in_states.sort_values('count_seller', ascending=False).head(10),
                palette= colors, 
                ax=ax[1])
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].tick_params(axis='x', labelrotation=50)
    ax[1].set_title("Based on State", loc="center", fontsize=18, pad=10)
    ax[1].tick_params(axis='y', labelsize=15)
    ax[1].tick_params(axis='x', labelsize=15)

    plt.suptitle("Distribution of Number of Sellers by City and State", fontsize=20)
    st.pyplot(fig)

    #Seller with Largest Order
    st.subheader("Seller with Largest Number of Order and Total Order Value")
    tab1, tab2 = st.tabs(['Count Order','Total Order Value'])
 
    with tab1:
        col1, col2, col3 = st.columns(3)
 
        with col1:
            max_seller_count_order = seller_count_sum_order.count_order.max()
            st.metric("Highest Number of Order", value=max_seller_count_order)

        with col2:
            min_seller_count_order = seller_count_sum_order.count_order.min()
            st.metric("Lowest Number of Order", value=min_seller_count_order)

        with col3:
            avg_seller_count_order = seller_count_sum_order.count_order.mean().astype(int)
            st.metric("Average Number of Order", value=avg_seller_count_order)
        
        fig, ax = plt.subplots(figsize=(25, 10))
        sns.barplot(x="count_order", 
                    y="seller_id", 
                    data= seller_count_sum_order.sort_values('count_order',ascending=False).head(10), 
                    palette= colors,
                    ax=ax)
        ax.set_ylabel('Seller ID', fontsize=18, labelpad=10)
        ax.set_xlabel('Number of Order', fontsize=18, labelpad=10)
        ax.set_title("Seller with Largest Number of Order", loc="center", fontsize=20, pad=10)
        ax.bar_label(ax.containers[0], label_type='center')
        ax.tick_params(axis ='y', labelsize=15)
        ax.tick_params(axis ='x', labelsize=15)
        st.pyplot(fig)
 
    with tab2:
        col1, col2, col3 = st.columns(3)
 
        with col1:
            max_seller_order_value = format_currency(seller_count_sum_order.sum_order_value.max(), "R$", locale='pt_BR')
            st.metric("Highest Total Order Value", value=max_seller_order_value)

        with col2:
            min_seller_order_value = format_currency(seller_count_sum_order.sum_order_value.min(), "R$", locale='pt_BR')
            st.metric("Lowest Total Order Value", value=min_seller_order_value)
        
        with col3:
            avg_seller_order_value = format_currency(seller_count_sum_order.sum_order_value.mean(), "R$", locale='pt_BR')
            st.metric("Average Total Order Value", value=avg_seller_order_value)

        
        fig, ax = plt.subplots(figsize=(25, 10))

        sns.barplot(x="sum_order_value", 
                    y="seller_id", 
                    data= seller_count_sum_order.sort_values('sum_order_value',ascending=False).head(10), 
                    palette= colors,
                    ax=ax)
        ax.set_ylabel('Seller ID', fontsize=18, labelpad=10)
        ax.set_xlabel('Total Order Value (R$)', fontsize=18, labelpad=10)
        ax.set_title("Seller with Largest Total Order Value", loc="center", fontsize=20, pad=10)
        ax.bar_label(ax.containers[0], label_type='center')
        ax.tick_params(axis ='y', labelsize=15)
        ax.tick_params(axis ='x', labelsize=15)
        st.pyplot(fig)

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