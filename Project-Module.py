import pandas as pd
import altair as alt
import streamlit as st
from pycaret.regression import *

def load_data_from_csv():
    datadf = pd.DataFrame(pd.read_csv('Placement_Data_Full_Class.csv'))
    return datadf

ingress_df = load_data_from_csv()

st.header('MBA Placement Data Visualization')
st.markdown('The visualization answers some questions on the MBA campus recruitment dataset that we have acquired.')


def data_transformations():
    ingress_df['workex_bin'] = ingress_df['workex'].apply(lambda row: 1 if row == 'Yes' else 0)
    ingress_df['status_int'] = ingress_df['status'].apply(lambda row: 1 if row == 'Placed' else 0)
    ingress_df['MBA Spec + Degree Spec'] = ingress_df['specialisation'] + " " + ingress_df['degree_t']
    return ingress_df

transformed_df = data_transformations()

def MLData():
    ml_model_data = transformed_df[['gender','ssc_p','ssc_b','hsc_p','hsc_b','hsc_s','degree_p','workex','etest_p','MBA Spec + Degree Spec','mba_p','salary']]
    return ml_model_data

def MLmodelDefinition():
    model = create_model('rf')
    tuned_model = tune_model(model)
    final_rf_model = finalize_model(tuned_model)
    save_model(final_rf_model, 'rf_model-Modular')

@st.cache
def predict_salary_value(test_data):
    rf_saved = load_model('rf_model-Modular')
    salary_prediction = predict_model(rf_saved, data = test_data)
    return salary_prediction['Label']

def render_visualization(chartval):
    st.write(chartval)


vis1 = st.checkbox("Does Gender Bias exist in campus recruitment?")
vis2 = st.checkbox("Correlation between salary degree percentage and specialization")
vis3 = st.checkbox('Do companies prefer candidates with work experience?')
vis4 = st.checkbox('Average Salaries of a degree and specialization combination')
vis5 = st.checkbox('How does work experience play a role in determining candidate salary?')
ml = st.checkbox('Probable candidate salary using ML-model:')

if(vis1):
    chart1 = alt.Chart(ingress_df).mark_bar().encode(
    x='gender:N',
    y='count(status_int):Q',
    color='gender:N',
    ).properties(
        height = 500,
        width = 500,
        title = "Does Gender Bias exist in campus recruitment?"
    ).interactive()
    render_visualization(chart1)

if(vis2):
    chart2 = alt.Chart(ingress_df).mark_circle().encode(
    alt.X('degree_p:Q', scale=alt.Scale(zero=False)),
    alt.Y('mba_p:Q', scale=alt.Scale(zero=False, padding=1)),
    color='MBA Spec + Degree Spec:N',
    size='salary:Q'
    ).properties(
        width = 800,
        height = 400,
        title = "Correlation between salary degree percentage and specialization:"
    ).interactive()
    render_visualization(chart2)

if(vis3):
    chart3 = alt.Chart(ingress_df).mark_bar().encode(
    x = 'workex:N',
    y = alt.Y('count(status_int):Q', title = 'Number of candidates placed'),
    color = 'workex:N'
    ).properties(
        width = 400,
        height = 500,
        title = 'Do companies prefer candidates with work experience?'
    ).interactive()
    render_visualization(chart3)

if(vis4):
    chart4 = alt.Chart(ingress_df).mark_bar().encode(
    x = alt.X('MBA Spec + Degree Spec:N', sort = '-y'),
    y = 'average(salary):Q',
    color = 'MBA Spec + Degree Spec:N',
    ).properties(
        width = 800,
        height = 400,
        title = 'Average Salaries of a degree and specialization combination:'
    ).interactive()
    render_visualization(chart4)

if(vis5):
    chart5 = alt.Chart(ingress_df).mark_point().encode(
    x='mba_p:Q',
    y='salary:Q',
    color='workex:N'
    ).properties(
        width = 500,
        height = 500,
        title = 'How does work experience play a role in determining candidate salary?'
    ).interactive()
    render_visualization(chart5)

if(ml):
    mlData = MLData()
    #clf1 = setup(data = mlData, target = 'salary')
    #MLmodelDefinition()
    st.write("Enter Candidate Details:")
    inp_gender = st.radio('Gender', ('F', 'M'), index=0)
    inp_ssc_p = st.slider('Secondary School Percentage', 40.0, 100.0, 50.0, step=0.1)
    inp_ssc_b = st.radio('Secondary School Education Board', ('Central','Others'), index=0)
    inp_hsc_p = st.slider('Higher Secondary School Percentage', 40.0, 100.0, 50.0, step=0.1)
    inp_hsc_b = st.radio('Higher Secondary School Education Board', ('Central','Others'), index=0)
    inp_hsc_s = st.radio('Higher Secondary stream', ('Science','Commerce','Arts'), index=0)
    inp_degree_p = st.slider('Degree Percentage', 40.0, 100.0, 50.0, step=0.1)
    inp_workex = st.radio('Prior Work experience', ('Yes','No'), index=0)
    inp_etest_p = st.slider('Employability Test Percentage', 40.0, 100.0, 50.0, step=0.1)
    inp_MBA_Degree_Spec = st.radio('MBA and Degree specialication', ('Mkt&HR Sci&Tech','Mkt&Fin Sci&Tech','Mkt&HR Comm&Mgmt','Mkt&Fin Comm&Mgmt','Mkt&HR Others','Mkt&Fin Others'), index=0)
    inp_mba_p = st.slider('MBA Percentage', 40.0, 100.0, 50.0, step=0.1)
    test_data = pd.DataFrame({'gender':[inp_gender],'ssc_p':[inp_ssc_p], 'ssc_b':[inp_ssc_b], 'hsc_p':[inp_hsc_p], 'hsc_b': [inp_hsc_b],'hsc_s':[inp_hsc_s],'degree_p':[inp_degree_p],'workex':[inp_workex],'etest_p':[inp_etest_p],'MBA Spec + Degree Spec':[inp_MBA_Degree_Spec],'mba_p':[inp_mba_p]})
    st.write("\n")
    rf_saved = load_model('rf_model-Modular')
    salary_prediction = predict_model(rf_saved, data = test_data)
    st.write('Probable candidate salary/annum = Rs.%0.2f'% (salary_prediction['Label'] * 12))
    # clf1 = setup(data = mlData, target = 'salary')



