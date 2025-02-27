import streamlit as st
import pandas as pd
import numpy as np

# Function to load and preprocess data
def load_data():
    uploaded_files = {}

    # File upload UI (2 rows, 2 files per row)
    col1, col2 = st.columns(2)
    with col1:
        uploaded_files["employee_wage_data"] = st.file_uploader("Upload Employee Wage Data", type=["xlsx"])
        uploaded_files["employees_notice_exit"] = st.file_uploader("Upload Employees Notice/Exit Data", type=["xlsx"])
    with col2:
        uploaded_files["employees_on_pip"] = st.file_uploader("Upload Employees on PIP", type=["xlsx"])
        uploaded_files["open_requisitions"] = st.file_uploader("Upload Open Requisitions Data", type=["xlsx"])

    # Check if all files are uploaded
    if all(uploaded_files.values()):
        st.success("All files uploaded successfully!")
        return uploaded_files
    return None

# Function to process and project wage costs
def process_wage_projections(files):
    # Load data
    wage_data = pd.read_excel(files["employee_wage_data"])
    notice_exit = pd.read_excel(files["employees_notice_exit"])
    pip_data = pd.read_excel(files["employees_on_pip"])
    open_requisitions = pd.read_excel(files["open_requisitions"])

    # Ensure 'target_joining_date' is in datetime format
    open_requisitions['target_joining_date'] = pd.to_datetime(open_requisitions['target_joining_date'], errors='coerce')

    # Get current date and determine ongoing financial quarter
    today = pd.Timestamp.today()
    current_month = today.month
    current_year = today.year

    # Determine the current financial quarter
    if current_month in [4, 5, 6]:  # Q1: Apr-Jun
        current_fy_quarter = f"FY{current_year + 1 % 100}Q1"
    elif current_month in [7, 8, 9]:  # Q2: Jul-Sep
        current_fy_quarter = f"FY{current_year + 1 % 100}Q2"
    elif current_month in [10, 11, 12]:  # Q3: Oct-Dec
        current_fy_quarter = f"FY{current_year + 1 % 100}Q3"
    else:  # Q4: Jan-Mar
        current_fy_quarter = f"FY{current_year % 100}Q4"

    # Create the next 6 months and quarters
    months = pd.date_range(start=today, periods=7, freq='M').strftime('%b-%Y')
    quarters = [f"FY{(current_year + (i // 4)) % 100}Q{(i % 4) + 1}" for i in range(7)]

    # Prepare data structure for projections
    department_costs_monthly = {month: {} for month in months}
    department_costs_quarterly = {q: {} for q in quarters}

    # Process employees for wage projections
    for _, row in wage_data.iterrows():
        department = row["Department"]
        monthly_salary = row["Monthly_Salary"] * 100000  # Convert from lakh INR to INR
        joining_date = pd.to_datetime(row["joining_date"], errors="coerce")

        for month in months:
            if joining_date <= pd.to_datetime(month, format='%b-%Y'):
                department_costs_monthly[month][department] = department_costs_monthly[month].get(department, 0) + monthly_salary

        for quarter in quarters:
            if joining_date <= pd.to_datetime(quarter[:6] + "01", format="FY%yQ%m%d"):
                department_costs_quarterly[quarter][department] = department_costs_quarterly[quarter].get(department, 0) + monthly_salary

    # Adjust costs for notice exit employees (pro-rated)
    for _, row in notice_exit.iterrows():
        department = row["Department"]
        monthly_salary = row["Monthly_Salary"] * 100000
        exit_date = pd.to_datetime(row["exit_date"], errors="coerce")

        if pd.notna(exit_date):
            for month in months:
                month_start = pd.to_datetime(month, format='%b-%Y')
                if exit_date < month_start:
                    department_costs_monthly[month][department] -= monthly_salary

            for quarter in quarters:
                quarter_start = pd.to_datetime(quarter[:6] + "01", format="FY%yQ%m%d")
                if exit_date < quarter_start:
                    department_costs_quarterly[quarter][department] -= monthly_salary

    # Adjust costs for PIP employees (two scenarios)
    department_costs_pip_100 = department_costs_monthly.copy()
    department_costs_pip_50 = department_costs_monthly.copy()

    for _, row in pip_data.iterrows():
        department = row["Department"]
        monthly_salary = row["Monthly_Salary"] * 100000
        pip_end_date = pd.to_datetime(row["pip_start_date"], errors="coerce") + pd.DateOffset(days=60)

        if pd.notna(pip_end_date):
            for month in months:
                month_start = pd.to_datetime(month, format='%b-%Y')
                if pip_end_date < month_start:
                    department_costs_pip_100[month][department] -= monthly_salary
                    department_costs_pip_50[month][department] -= (monthly_salary * 0.5)

    # Convert to DataFrame for display
    monthly_df_100 = pd.DataFrame(department_costs_pip_100).fillna(0).astype(int)
    monthly_df_50 = pd.DataFrame(department_costs_pip_50).fillna(0).astype(int)

    # Function to format numbers in Indian numbering system
    def format_inr(num):
        return "{:,.0f}".format(num).replace(",", "X").replace("X", ",")

    # Apply formatting to all numbers
    monthly_df_100 = monthly_df_100.applymap(format_inr)
    monthly_df_50 = monthly_df_50.applymap(format_inr)

    # Add total row and make it bold
    monthly_df_100.loc["Total"] = "**" + monthly_df_100.sum(numeric_only=True).astype(str) + "**"
    monthly_df_50.loc["Total"] = "**" + monthly_df_50.sum(numeric_only=True).astype(str) + "**"

    # Display the formatted tables
    st.subheader("Projected Monthly Wage Cost (All PIP Exit)")
    st.table(monthly_df_100)

    st.subheader("Projected Monthly Wage Cost (50% PIP Exit)")
    st.table(monthly_df_50)

# Streamlit UI
st.title("Wage Cost Projection App")
uploaded_files = load_data()

if uploaded_files:
    if st.button("Run Wage Cost Projection"):
        process_wage_projections(uploaded_files)
