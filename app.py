import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Helper functions
def format_indian_number(num):
    return "{:,}".format(num).replace(",", "_").replace("_", ",")  # Indian numbering format

def get_fy_quarter(date):
    year = date.year if date.month >= 4 else date.year - 1
    if date.month in [4, 5, 6]:
        return f"FY{year+1}Q1"
    elif date.month in [7, 8, 9]:
        return f"FY{year+1}Q2"
    elif date.month in [10, 11, 12]:
        return f"FY{year+1}Q3"
    else:
        return f"FY{year+1}Q4"

def process_uploaded_files(employee_wage, notice_exit, pip, requisitions):
    # Convert necessary date columns to datetime
    for df, date_cols in [(notice_exit, ['exit_date', 'notice_start_date', 'expected_exit_date']),
                           (pip, ['pip_start_date', 'pip_end_date']),
                           (requisitions, ['posting_date', 'target_joining_date'])]:
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Determine missing salaries
    salary_by_band = employee_wage.groupby('band')['Monthly_Salary'].mean()
    for df in [notice_exit, pip]:
        df.loc[df['Monthly_Salary'].isna(), 'Monthly_Salary'] = df['band'].map(salary_by_band)

    return employee_wage, notice_exit, pip, requisitions

# Streamlit UI
st.title("Wage Cost Projection App")

# Upload Files
col1, col2 = st.columns(2)
with col1:
    employee_wage_file = st.file_uploader("Upload Employee Wage Data", type=['xlsx'])
    notice_exit_file = st.file_uploader("Upload Employees Notice/Exit Data", type=['xlsx'])
with col2:
    pip_file = st.file_uploader("Upload Employees on PIP Data", type=['xlsx'])
    requisitions_file = st.file_uploader("Upload Open Requisitions Data", type=['xlsx'])

if st.button("Run Wage Projection"):
    if not all([employee_wage_file, notice_exit_file, pip_file, requisitions_file]):
        st.error("Please upload all four files.")
    else:
        # Read Excel files
        employee_wage = pd.read_excel(employee_wage_file)
        notice_exit = pd.read_excel(notice_exit_file)
        pip = pd.read_excel(pip_file)
        requisitions = pd.read_excel(requisitions_file)
        
        # Process data
        employee_wage, notice_exit, pip, requisitions = process_uploaded_files(employee_wage, notice_exit, pip, requisitions)
        
        # Calculate projections (placeholder logic, actual logic needs to be added)
        current_date = datetime.today()
        ongoing_quarter = get_fy_quarter(current_date)
        
        # Placeholder: Creating empty DataFrame for wage cost projections
        months = [(current_date + pd.DateOffset(months=i)).strftime('%b-%Y') for i in range(7)]
        quarters = [get_fy_quarter(current_date + pd.DateOffset(months=3 * i)) for i in range(7)]
        
        departments = employee_wage['Department'].unique()
        
        # Placeholder: Generate random data for wage projections
        monthly_table = pd.DataFrame(index=departments, columns=months, data=np.random.randint(500000, 10000000, size=(len(departments), len(months))))
        quarterly_table = pd.DataFrame(index=departments, columns=quarters, data=np.random.randint(2000000, 40000000, size=(len(departments), len(quarters))))
        
       
        
        # Format Numbers
        monthly_table = monthly_table.applymap(format_indian_number)
        quarterly_table = quarterly_table.applymap(format_indian_number)
        
        # Display tables
        st.subheader("Monthly Wage Cost Projection")
        
        st.subheader("Quarterly Wage Cost Projection")
