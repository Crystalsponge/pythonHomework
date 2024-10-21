from cProfile import label

import matplotlib
import streamlit
import ucimlrepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from ucimlrepo import fetch_ucirepo

# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.target



variables = adult.variables
##Selecting data for graph
ageindex = ["5-15", "15-30","30-45","45-60","60-75","75-90"]
agegroup = pd.cut(X["age"], [5, 15, 30, 45, 60, 75, 90], labels=ageindex)
agecounts = agegroup.value_counts(normalize= True, sort = False) * 5000
agelabel = agecounts.index.to_numpy()

fig, ax = plt.subplots()
ax.hist(X['hours-per-week'], bins=20, color='skyblue')
ax.set_xlabel("Hours per Week")
ax.set_ylabel("Frequency")

missing_data = X.isnull().sum()

categorical_columns = X.select_dtypes(include=['object']).columns
frequency_dict = {}

# Loop through categorical columns and get the frequency count for each
for col in categorical_columns:
    frequency_dict[col] = X[col].value_counts()

# Convert the dictionary to a DataFrame for better readability
frequency_df = pd.DataFrame({key: value for key, value in frequency_dict.items()})

numerical_columns = X.select_dtypes(include = ['int64', 'float64']).columns
summary_stats = X[numerical_columns].agg(['mean', 'median', 'std'])

#streamlit app part
st.sidebar.header("Options")
st.title("DATA ABOUT ADULTS WHO MAKE OVER 50K$ A YEAR")
st.dataframe(X)
st.write("Missing data", missing_data)
st.write("Frequency count: ", frequency_df)
st.dataframe(summary_stats)
chart_type = st.sidebar.selectbox("Select chart type", ["Distribution of age group", "Level of education","Age and Marital status","Hours per week"])
#
#
def generate_chart(chart_type):
    st.write(chart_type + " chart")
    if chart_type == "Distribution of age group":
        st.bar_chart(agecounts, x_label = "Age group", y_label = "People")
        st.write("""The chart shows that the highest proportion of high-income earners is in the 30-45 age group. As age increases, the proportion of high-income earners generally decreases. However, a significant number of individuals maintain high incomes beyond retirement age. This suggests that while early career success is common, financial success can be achieved at different stages of life.""")
    elif chart_type == "Level of education":
        st.bar_chart(X["education"].value_counts(), x_label = "Level of Education", y_label = "People")
        st.write("""The chart shows that the majority of people have completed at least a high school education, with a significant number pursuing higher education. However, there is still a portion of the population with lower levels of education. This suggests that while education levels have increased over time, there is still room for improvement in terms of educational attainment.
""")
    elif chart_type == "Age and Marital status":
        st.scatter_chart(X[["age","marital-status"]], x = "age", y="marital-status")
        st.write("The chart shows that married-civ-spouse is the most common marital status among high-income earners, followed by never-married. Divorce and separation are less common among this group. Widowed is the least common marital status. This suggests a potential association between marriage and higher income levels, but individual circumstances and factors like education, experience, and industry also play a significant role.")
    elif chart_type == "Hours per week":
        st.pyplot(fig)
        st.write("The plot shows that the median number of hours worked per week is around 40, with a wide range of variation in the data. There are a few outliers who work significantly more or fewer hours than the typical worker. The distribution of hours worked per week is slightly skewed to the right, suggesting that there are a few individuals who work significantly more hours than the majority of the sample.")
generate_chart(chart_type)



# metadata
print('Metadata: ')
print(adult.metadata)

# variable information
print("Variable information: ")
print(adult.variables)

