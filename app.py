import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
from dateutil.relativedelta import relativedelta
import locale

# Set page configuration
st.set_page_config(page_title="Wage Cost Projection Tool", layout="wide")

# Configure Indian number formatting
def format_indian_currency(number):
    """Format a number to Indian currency format (with commas)"""
    s = f"{int(round(number, 0)):,}"
    # Replace with Indian number system (commas after 3, 2, 2 digits)
    parts = s.split(',')
    if len(parts) > 1:
        last_part = parts.pop()
        first_part = parts.pop(0)
        remaining_parts = ','.join(parts)
        if remaining_parts:
            return f"{first_part},{remaining_parts},{last_part}"
        else:
            return f"{first_part},{last_part}"
    else:
        return s

# Create title and description
st.title("Wage Cost Projection Tool")
st.markdown("Upload Excel files to generate wage cost projections by department.")

# Create file upload section with two rows
col1, col2 = st.columns(2)

with col1:
    st.subheader("Employee Data Files")
    employees_file = st.file_uploader("1. Upload Employee Wage Data", type=["xlsx"])
    exited_employees_file = st.file_uploader("2. Upload Employees on Notice/Exited", type=["xlsx"])

with col2:
    st.subheader("Additional Files")
    pip_employees_file = st.file_uploader("3. Upload Employees on PIP", type=["xlsx"])
    requisitions_file = st.file_uploader("4. Upload Open Requisitions", type=["xlsx"])

# Check if all files are uploaded
all_files_uploaded = employees_file and exited_employees_file and pip_employees_file and requisitions_file

if all_files_uploaded:
    st.success("All files uploaded successfully!")
    
    # Load the data to get departments for dropdown
    employees_df = pd.read_excel(employees_file)
    
    # Get unique departments for the dropdown
    departments = sorted(employees_df['Department'].unique().tolist())
    
    # Department selection
    selected_department = st.selectbox(
        "Select Department for Wage Cost Projection:",
        departments,
        index=departments.index('Pre-Sales') if 'Pre-Sales' in departments else 0
    )
    
    process_button = st.button(f"Generate Wage Cost Projections for {selected_department}")
else:
    st.info("Please upload all required files to proceed.")
    process_button = False
    selected_department = None

# Processing section
if process_button and selected_department:
    st.subheader(f"Processing Data for {selected_department} Department...")
    
    # Load data from Excel files if not already loaded
    if 'employees_df' not in locals():
        with st.spinner("Loading employee data..."):
            employees_df = pd.read_excel(employees_file)
            
    with st.spinner("Loading exit data..."):
        exited_df = pd.read_excel(exited_employees_file)
        
    with st.spinner("Loading PIP data..."):
        pip_df = pd.read_excel(pip_employees_file)
        
    with st.spinner("Loading requisitions..."):
        req_df = pd.read_excel(requisitions_file)

    # Data preprocessing
    with st.spinner("Preprocessing data..."):
        # Convert date columns to datetime format
        date_columns = ['joining_date']
        for col in date_columns:
            if col in employees_df.columns:
                employees_df[col] = pd.to_datetime(employees_df[col], errors='coerce')
                
        # Convert date columns for exited employees
        exit_date_columns = ['joining_date', 'exit_date', 'notice_start_date', 'expected_exit_date']
        for col in exit_date_columns:
            if col in exited_df.columns:
                exited_df[col] = pd.to_datetime(exited_df[col], errors='coerce')
                
        # Convert date columns for PIP employees
        pip_date_columns = ['joining_date', 'pip_start_date', 'pip_end_date']
        for col in pip_date_columns:
            if col in pip_df.columns:
                pip_df[col] = pd.to_datetime(pip_df[col], errors='coerce')
                
        # Convert date columns for requisitions
        req_date_columns = ['posting_date', 'target_joining_date']
        for col in req_date_columns:
            if col in req_df.columns:
                req_df[col] = pd.to_datetime(req_df[col], errors='coerce')
        
        # Add expected exit date for PIP employees (60 days after PIP start date)
        pip_df['expected_exit_date_pip'] = pip_df['pip_start_date'] + timedelta(days=60)
        
    # Generate projection months
    current_date = datetime(2025, 2, 1)  # Starting from February 2025
    projection_months = []
    projection_dates = []
    
    for i in range(6):  # 6 months projection
        month_date = current_date + relativedelta(months=i)
        month_name = month_date.strftime('%b %Y')
        projection_months.append(month_name)
        projection_dates.append(month_date)
    
    # Filter data for selected department only
    dept_employees = employees_df[employees_df['Department'] == selected_department].copy()
    dept_exited = exited_df[exited_df['Department'] == selected_department].copy()
    dept_pip = pip_df[pip_df['Department'] == selected_department].copy()
    dept_req = req_df[req_df['Department'] == selected_department].copy()
    
    # Calculate monthly costs for each projection month
    monthly_costs = []
    monthly_explanations = []
    
    for idx, projection_date in enumerate(projection_dates):
        month_end = projection_date + relativedelta(day=31)
        if month_end.day != calendar.monthrange(month_end.year, month_end.month)[1]:
            month_end = projection_date + relativedelta(day=calendar.monthrange(projection_date.year, projection_date.month)[1])
        
        days_in_month = calendar.monthrange(projection_date.year, projection_date.month)[1]
        month_cost = 0
        explanation = f"### Wage Cost Calculation for {projection_months[idx]}\n\n"
        
        # Generate a list of employee IDs to exclude (those in exit or PIP lists)
        employees_to_exclude = set(dept_exited['employee_id'].tolist() + dept_pip['employee_id'].tolist())
        
        # 1. Regular employees (exclude those in exit or PIP lists)
        active_employees = dept_employees[~dept_employees['employee_id'].isin(employees_to_exclude)]
        
        # Calculate cost for regular employees for the full month
        regular_cost = active_employees['Monthly_Salary'].sum() * 100000  # Convert lakhs to INR
        month_cost += regular_cost
        
        explanation += f"**Regular Employees:** {len(active_employees)} active employees contribute ₹{format_indian_currency(regular_cost)} for the full month.\n\n"
        if len(active_employees) > 0:
            explanation += "Employee IDs: " + ", ".join(active_employees['employee_id'].astype(str).tolist()) + "\n\n"
        
        # 2. Employees who are exiting (pro-rated)
        exiting_employees_cost = 0
        explanation += "**Exiting Employees (Pro-rated costs):**\n\n"
        
        if not dept_exited.empty:
            for _, emp in dept_exited.iterrows():
                # Handle both exit_date and expected_exit_date
                
                # If employee has already exited before the projection month, skip
                if pd.notna(emp['exit_date']) and emp['exit_date'] < projection_date:
                    continue
                
                # Case 1: Employee has an actual exit_date in this month
                if pd.notna(emp['exit_date']) and emp['exit_date'].year == projection_date.year and emp['exit_date'].month == projection_date.month:
                    # Calculate pro-rated cost based on days worked in this month
                    days_worked = emp['exit_date'].day
                    pro_rated_cost = (emp['Monthly_Salary'] * 100000) * (days_worked / days_in_month)
                    month_cost += pro_rated_cost
                    exiting_employees_cost += pro_rated_cost
                    
                    explanation += f"- Employee {emp['employee_id']} ({emp['full_name']}) works until {emp['exit_date'].strftime('%d %b %Y')} ({days_worked} days): ₹{format_indian_currency(pro_rated_cost)}\n"
                
                # Case 2: Employee has an expected_exit_date in this month
                elif pd.notna(emp['expected_exit_date']) and emp['expected_exit_date'].year == projection_date.year and emp['expected_exit_date'].month == projection_date.month:
                    # Calculate pro-rated cost based on days worked in this month
                    days_worked = emp['expected_exit_date'].day
                    pro_rated_cost = (emp['Monthly_Salary'] * 100000) * (days_worked / days_in_month)
                    month_cost += pro_rated_cost
                    exiting_employees_cost += pro_rated_cost
                    
                    explanation += f"- Employee {emp['employee_id']} ({emp['full_name']}) (on notice) works until {emp['expected_exit_date'].strftime('%d %b %Y')} ({days_worked} days): ₹{format_indian_currency(pro_rated_cost)}\n"
                
                # Case 3: Employee has exit_date or expected_exit_date in future months
                elif (pd.notna(emp['exit_date']) and emp['exit_date'] > month_end) or (pd.notna(emp['expected_exit_date']) and emp['expected_exit_date'] > month_end):
                    full_cost = emp['Monthly_Salary'] * 100000
                    month_cost += full_cost
                    exiting_employees_cost += full_cost
                    
                    explanation += f"- Employee {emp['employee_id']} ({emp['full_name']}) works full month: ₹{format_indian_currency(full_cost)}\n"
                
                # Case 4: Employee has no exit dates set but is in the exit list
                elif pd.isna(emp['exit_date']) and pd.isna(emp['expected_exit_date']):
                    full_cost = emp['Monthly_Salary'] * 100000
                    month_cost += full_cost
                    exiting_employees_cost += full_cost
                    
                    explanation += f"- Employee {emp['employee_id']} ({emp['full_name']}) (on exit list but no date set) works full month: ₹{format_indian_currency(full_cost)}\n"
        
        if exiting_employees_cost == 0:
            explanation += "- No exiting employees for this month.\n"
        
        explanation += f"\nTotal Exiting Employees Cost: ₹{format_indian_currency(exiting_employees_cost)}\n\n"
        
        # 3. Employees on PIP (pro-rated if expected to exit in this month)
        pip_employees_cost = 0
        explanation += "**PIP Employees (Pro-rated costs):**\n\n"
        
        if not dept_pip.empty:
            for _, emp in dept_pip.iterrows():
                # Calculate expected exit date (60 days after PIP start)
                if pd.isna(emp['pip_start_date']):
                    continue
                
                expected_exit_date = emp['expected_exit_date_pip']
                
                # Skip if already exited before the projection month
                if expected_exit_date < projection_date:
                    continue
                
                # If employee has an expected exit date in this month
                if expected_exit_date.year == projection_date.year and expected_exit_date.month == projection_date.month:
                    # Calculate pro-rated cost based on days worked in this month
                    days_worked = expected_exit_date.day
                    pro_rated_cost = (emp['Monthly_Salary'] * 100000) * (days_worked / days_in_month)
                    month_cost += pro_rated_cost
                    pip_employees_cost += pro_rated_cost
                    
                    explanation += f"- Employee {emp['employee_id']} ({emp['full_name']}) on PIP works until {expected_exit_date.strftime('%d %b %Y')} ({days_worked} days): ₹{format_indian_currency(pro_rated_cost)}\n"
                
                # If employee exits in a future month, add full salary for this month
                elif expected_exit_date > month_end:
                    full_cost = emp['Monthly_Salary'] * 100000
                    month_cost += full_cost
                    pip_employees_cost += full_cost
                    
                    explanation += f"- Employee {emp['employee_id']} ({emp['full_name']}) on PIP works full month: ₹{format_indian_currency(full_cost)}\n"
        
        if pip_employees_cost == 0:
            explanation += "- No PIP employees for this month.\n"
            
        explanation += f"\nTotal PIP Employees Cost: ₹{format_indian_currency(pip_employees_cost)}\n\n"
        
        # 4. New hires from open requisitions (pro-rated for joining month)
        new_hires_cost = 0
        explanation += "**New Hires (Pro-rated costs):**\n\n"
        
        if not dept_req.empty:
            for _, req in dept_req.iterrows():
                # Consider only if joining date is set and status is not 'filled'
                if pd.isna(req['target_joining_date']) or req['status'].lower() == 'filled':
                    continue
                
                # If joining in this month
                if req['target_joining_date'].year == projection_date.year and req['target_joining_date'].month == projection_date.month:
                    # Calculate pro-rated cost based on days worked
                    days_worked = days_in_month - req['target_joining_date'].day + 1
                    pro_rated_cost = (req['Expected_Monthly_Salary'] * 100000) * (days_worked / days_in_month)
                    month_cost += pro_rated_cost
                    new_hires_cost += pro_rated_cost
                    
                    explanation += f"- Requisition {req['requisition_id']} joining on {req['target_joining_date'].strftime('%d %b %Y')} ({days_worked} days): ₹{format_indian_currency(pro_rated_cost)}\n"
                
                # If already joined in previous months
                elif req['target_joining_date'] < projection_date:
                    full_cost = req['Expected_Monthly_Salary'] * 100000
                    month_cost += full_cost
                    new_hires_cost += full_cost
                    
                    explanation += f"- Requisition {req['requisition_id']} (joined on {req['target_joining_date'].strftime('%d %b %Y')}) works full month: ₹{format_indian_currency(full_cost)}\n"
        
        if new_hires_cost == 0:
            explanation += "- No new hires for this month.\n"
            
        explanation += f"\nTotal New Hires Cost: ₹{format_indian_currency(new_hires_cost)}\n\n"
        
        # Calculate total monthly cost
        explanation += f"**Total Monthly Cost for {projection_months[idx]}: ₹{format_indian_currency(month_cost)}**"
        
        monthly_costs.append(month_cost)
        monthly_explanations.append(explanation)
    
    # Display results
    st.header(f"{selected_department} Department Monthly Wage Cost Projection")
    
    # Create a DataFrame for the projection
    projection_df = pd.DataFrame({
        'Month': projection_months,
        'Cost (₹)': monthly_costs
    })
    
    # Display the projection table
    st.dataframe(
        projection_df.style
        .format({'Cost (₹)': lambda x: format_indian_currency(x)})
        .set_properties(**{'font-weight': 'bold'})
    )
    
    # Display detailed explanations for each month
    st.header("Monthly Cost Calculation Details")
    
    # Create tabs for each month's explanation
    tabs = st.tabs(projection_months)
    
    for i, tab in enumerate(tabs):
        with tab:
            st.markdown(monthly_explanations[i])

# Run the app
if __name__ == "__main__":
    pass