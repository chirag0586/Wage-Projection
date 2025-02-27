import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Wage Cost Projection App",
    page_icon="💰",
    layout="wide"
)

# Helper function to format numbers in Indian numbering system
def format_inr(n):
    if pd.isna(n):
        return "0"
    try:
        n = int(round(n))
    except:
        return str(n)
    
    s = str(n)
    if len(s) <= 3:
        return s
    else:
        # Format as per Indian numbering system (e.g., 1,23,45,678)
        result = s[-3:]
        s = s[:-3]
        
        while s:
            if len(s) >= 2:
                result = s[-2:] + "," + result
                s = s[:-2]
            else:
                result = s + "," + result
                s = ""
                
        return result

# Function to calculate days in month
def days_in_month(year, month):
    return calendar.monthrange(year, month)[1]

# Function to prorate salary based on days worked in a month
def prorate_salary(monthly_salary, start_date, end_date, month_start, month_end):
    # Ensure dates are datetime objects
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    if isinstance(month_start, str):
        month_start = pd.to_datetime(month_start)
    if isinstance(month_end, str):
        month_end = pd.to_datetime(month_end)
    
    # Calculate effective start and end dates
    effective_start = max(start_date, month_start)
    effective_end = min(end_date, month_end)
    
    # If effective_start > effective_end, employee didn't work in this month
    if effective_start > effective_end:
        return 0
    
    # Calculate days worked and total days in month
    days_worked = (effective_end - effective_start).days + 1
    total_days = (month_end - month_start).days + 1
    
    # Calculate prorated salary
    return monthly_salary * days_worked / total_days

# Function to get fiscal year quarter
def get_fiscal_quarter(date):
    # Convert to datetime if string
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    # Indian fiscal year starts in April
    if date.month >= 4 and date.month <= 6:
        return f"FY{str(date.year)[2:]}Q1"
    elif date.month >= 7 and date.month <= 9:
        return f"FY{str(date.year)[2:]}Q2"
    elif date.month >= 10 and date.month <= 12:
        return f"FY{str(date.year)[2:]}Q3"
    else:  # Jan to Mar
        return f"FY{str(date.year)[2:]}Q4"

# Function to get month name with year
def get_month_year(date):
    if isinstance(date, str):
        date = pd.to_datetime(date)
    return date.strftime("%b %Y")







# Function to download dataframe as Excel
def download_excel(df, filename):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=True, sheet_name='Sheet1')
    
    b64 = base64.b64encode(output.getvalue()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel file</a>'
    return href




# Create a debugging function to calculate pre-sales department costs
# This code should be added to the original app after data loading but before the main calculations

def debug_presales_calculations(employee_data, exits_data, pip_data, requisitions_data):
    """
    Debug function to show detailed calculations for Pre-Sales department
    for the months of February through July.
    """
    import pandas as pd
    from datetime import datetime, timedelta
    import calendar
    
    # Define the months we want to analyze
    months_to_analyze = {
        2: 'February',
        3: 'March', 
        4: 'April',
        5: 'May',
        6: 'June',
        7: 'July'
    }
    
    # Current year (2025 based on the app description)
    current_year = 2025
    
    # Filter for Pre-Sales department employees
    presales_employees = employee_data[employee_data['Department'] == 'Pre-Sales'].copy()
    presales_exits = exits_data[exits_data['Department'] == 'Pre-Sales'].copy()
    presales_pip = pip_data[pip_data['Department'] == 'Pre-Sales'].copy()
    presales_reqs = requisitions_data[requisitions_data['Department'] == 'Pre-Sales'].copy()
    
    print(f"Debug Analysis for Pre-Sales Department (February-July {current_year})")
    print("-" * 80)
    
    # Print basic stats
    print(f"Total Pre-Sales employees: {len(presales_employees)}")
    print(f"Pre-Sales employees exiting: {len(presales_exits)}")
    print(f"Pre-Sales employees on PIP: {len(presales_pip)}")
    print(f"Pre-Sales open requisitions: {len(presales_reqs)}")
    print("-" * 80)
    
    # For each month, calculate the detailed breakdown
    for month_num, month_name in months_to_analyze.items():
        print(f"\
{month_name} {current_year} Calculation Details:")
        print("-" * 50)
        
        # Calculate days in the month
        days_in_month = calendar.monthrange(current_year, month_num)[1]
        month_start = datetime(current_year, month_num, 1)
        month_end = datetime(current_year, month_num, days_in_month)
        
        # Track total cost for the month
        total_month_cost = 0
        
        # 1. Regular employees (not exiting, not on PIP)
        regular_employees = presales_employees.copy()
        
        # Remove employees who are in exits_data or pip_data
        if not presales_exits.empty:
            regular_employees = regular_employees[~regular_employees['employee_id'].isin(presales_exits['employee_id'])]
        
        if not presales_pip.empty:
            regular_employees = regular_employees[~regular_employees['employee_id'].isin(presales_pip['employee_id'])]
        
        regular_cost = regular_employees['Monthly_Salary'].sum()
        total_month_cost += regular_cost
        
        print(f"Regular employees ({len(regular_employees)}):")
        for _, emp in regular_employees.iterrows():
            print(f"  - {emp['full_name']}: Full month ({days_in_month} days) = ₹{emp['Monthly_Salary']:,.2f}")
        
        # 2. Exiting employees (pro-rated based on exit date)
        if not presales_exits.empty:
            print(f"\
Exiting employees ({len(presales_exits)}):")
            exit_cost = 0
            
            for _, emp in presales_exits.iterrows():
                # Determine the actual exit date
                if pd.notna(emp['exit_date']):
                    exit_date = pd.to_datetime(emp['exit_date'])
                elif pd.notna(emp['expected_exit_date']):
                    exit_date = pd.to_datetime(emp['expected_exit_date'])
                else:
                    # If no exit date is specified, assume they're staying the full month
                    exit_date = month_end + timedelta(days=1)
                
                # Calculate days worked in this month
                if exit_date <= month_start:
                    # Already exited before this month
                    days_worked = 0
                elif exit_date >= month_end:
                    # Exiting after this month ends
                    days_worked = days_in_month
                else:
                    # Exiting during this month
                    days_worked = (exit_date - month_start).days + 1
                
                # Calculate pro-rated salary
                if days_worked > 0:
                    prorated_salary = emp['Monthly_Salary'] * (days_worked / days_in_month)
                    exit_cost += prorated_salary
                    print(f"  - {emp['full_name']}: {days_worked}/{days_in_month} days = ₹{prorated_salary:,.2f}")
                    print(f"    (Exit date: {exit_date.strftime('%Y-%m-%d')})")
                else:
                    print(f"  - {emp['full_name']}: Already exited before {month_name}")
            
            total_month_cost += exit_cost
        
        # 3. PIP employees (assuming Scenario 1: all exit after 60 days)
        if not presales_pip.empty:
            print(f"\
PIP employees ({len(presales_pip)}):")
            pip_cost = 0
            
            for _, emp in presales_pip.iterrows():
                pip_start = pd.to_datetime(emp['pip_start_date'])
                expected_exit = pip_start + timedelta(days=60)  # Scenario 1: All exit after 60 days
                
                # Calculate days worked in this month
                if expected_exit <= month_start:
                    # Already exited before this month
                    days_worked = 0
                elif expected_exit >= month_end:
                    # Exiting after this month ends
                    days_worked = days_in_month
                else:
                    # Exiting during this month
                    days_worked = (expected_exit - month_start).days + 1
                
                # Calculate pro-rated salary
                if days_worked > 0:
                    prorated_salary = emp['Monthly_Salary'] * (days_worked / days_in_month)
                    pip_cost += prorated_salary
                    print(f"  - {emp['full_name']}: {days_worked}/{days_in_month} days = ₹{prorated_salary:,.2f}")
                    print(f"    (PIP start: {pip_start.strftime('%Y-%m-%d')}, Expected exit: {expected_exit.strftime('%Y-%m-%d')})")
                else:
                    print(f"  - {emp['full_name']}: Already exited before {month_name} due to PIP")
            
            total_month_cost += pip_cost
        
        # 4. New hires from open requisitions
        if not presales_reqs.empty:
            print(f"\
Expected new hires ({len(presales_reqs)}):")
            new_hire_cost = 0
            
            for _, req in presales_reqs.iterrows():
                joining_date = pd.to_datetime(req['target_joining_date'])
                
                # Calculate days worked in this month
                if joining_date > month_end:
                    # Joining after this month
                    days_worked = 0
                elif joining_date <= month_start:
                    # Joined before or at the start of this month
                    days_worked = days_in_month
                else:
                    # Joining during this month
                    days_worked = (month_end - joining_date).days + 1
                
                # Calculate pro-rated salary
                if days_worked > 0:
                    prorated_salary = req['Expected_Monthly_Salary'] * (days_worked / days_in_month)
                    new_hire_cost += prorated_salary
                    print(f"  - Req #{req['requisition_id']} ({req['position']}): {days_worked}/{days_in_month} days = ₹{prorated_salary:,.2f}")
                    print(f"    (Expected joining: {joining_date.strftime('%Y-%m-%d')})")
                else:
                    print(f"  - Req #{req['requisition_id']} ({req['position']}): Not joining in {month_name}")
            
            total_month_cost += new_hire_cost
        
        print(f"\
Total Pre-Sales cost for {month_name} {current_year}: ₹{total_month_cost:,.2f}")
    
    return "Debugging complete"

# Note: This function should be called after loading all the data files
# Add this line after data loading in the original code:
# debug_presales_calculations(employee_data, exits_data, pip_data, requisitions_data)

print("Debugging function created. This should be added to the original app code.")
print("Call this function after loading all data files but before running the main calculations.")
print("It will provide detailed breakdown of Pre-Sales department costs for February through July.")

debug_presales_calculations(employee_data, exits_data, pip_data, requisitions_data)


# Main app title
st.title("Wage Cost Projection App")
st.markdown("Upload four Excel files to generate monthly and quarterly wage cost projections.")

# File upload section
st.header("Upload Data Files")

col1, col2 = st.columns(2)

with col1:
    employee_file = st.file_uploader("1. Employee Wage Data", type=["xlsx", "csv"])
    notice_file = st.file_uploader("2. Employees on Notice/Exited", type=["xlsx", "csv"])

with col2:
    pip_file = st.file_uploader("3. Employees on PIP", type=["xlsx", "csv"])
    requisition_file = st.file_uploader("4. Open Requisitions", type=["xlsx", "csv"])

# Function to load data from uploaded files
def load_data(file):
    if file is None:
        return None
    
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Process data when all files are uploaded
if employee_file and notice_file and pip_file and requisition_file:
    st.success("All files uploaded successfully!")
    
    # Load data
    with st.spinner("Loading data..."):
        df_employee = load_data(employee_file)
        df_notice = load_data(notice_file)
        df_pip = load_data(pip_file)
        df_requisition = load_data(requisition_file)
    
    # Display data preview if requested
    with st.expander("Preview Uploaded Data"):
        st.subheader("Employee Wage Data")
        st.dataframe(df_employee.head())
        
        st.subheader("Employees on Notice/Exited")
        st.dataframe(df_notice.head())
        
        st.subheader("Employees on PIP")
        st.dataframe(df_pip.head())
        
        st.subheader("Open Requisitions")
        st.dataframe(df_requisition.head())
    
    # Date range selection for projection
    st.header("Projection Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Projection Start Date", datetime.now().replace(day=1))
    
    with col2:
        # Default to 12 months from start date
        end_date = st.date_input("Projection End Date", 
                                (datetime.now().replace(day=1) + timedelta(days=365)).replace(day=1) - timedelta(days=1))
    
    # Convert to datetime for calculations
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # PIP assumption
    pip_exit_percentage = st.slider("Percentage of PIP employees expected to exit", 0, 100, 50)
    
    # Run projection button
    if st.button("Generate Wage Cost Projections"):
        with st.spinner("Calculating projections..."):
            # Data preprocessing
            # Convert date columns to datetime
            date_columns = ['joining_date', 'exit_date', 'notice_start_date', 'expected_exit_date', 
                           'pip_start_date', 'pip_end_date', 'target_joining_date']
            
            for df in [df_employee, df_notice, df_pip, df_requisition]:
                for col in date_columns:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Generate list of months for projection
            projection_months = []
            current_month = start_date
            while current_month <= end_date:
                projection_months.append(current_month)
                # Move to next month
                if current_month.month == 12:
                    current_month = pd.Timestamp(year=current_month.year + 1, month=1, day=1)
                else:
                    current_month = pd.Timestamp(year=current_month.year, month=current_month.month + 1, day=1)
            
            # Create empty dataframe for monthly projections
            departments = pd.concat([
                df_employee['Department'], 
                df_notice['Department'], 
                df_pip['Department'], 
                df_requisition['Department']
            ]).unique()
            
            # Initialize monthly projection dataframe
            monthly_projection = pd.DataFrame(index=departments)
            
            # Calculate average salary by band for filling missing values
            band_avg_salary = df_employee.groupby('band')['Monthly_Salary'].mean()
            
            # Process each month
            for month_start in projection_months:
                # Calculate month end
                next_month = month_start + pd.DateOffset(months=1)
                month_end = next_month - pd.DateOffset(days=1)
                
                # Column name for this month
                month_col = get_month_year(month_start)
                
                # Initialize department costs for this month
                dept_costs = {dept: 0 for dept in departments}
                
                # 1. Process regular employees (not on notice or PIP)
                for _, emp in df_employee.iterrows():
                    # Skip if employee is in notice or PIP list
                    if emp['employee_id'] in df_notice['employee_id'].values or emp['employee_id'] in df_pip['employee_id'].values:
                        continue
                    
                    # Check if employee has joined by this month
                    if emp['joining_date'] <= month_end:
                        # Calculate prorated salary if joined during this month
                        if emp['joining_date'] >= month_start:
                            salary = prorate_salary(emp['Monthly_Salary'], emp['joining_date'], month_end, month_start, month_end)
                        else:
                            salary = emp['Monthly_Salary']
                        
                        # Add to department cost
                        dept_costs[emp['Department']] += salary
                
                # 2. Process employees on notice/exited
                for _, emp in df_notice.iterrows():
                    # Check if employee has joined by this month
                    if emp['joining_date'] <= month_end:
                        # Determine exit date (actual or expected)
                        exit_date = emp['exit_date'] if pd.notna(emp['exit_date']) else emp['expected_exit_date']
                        
                        # Skip if employee has already exited before this month
                        if pd.notna(exit_date) and exit_date < month_start:
                            continue
                        
                        # Calculate prorated salary
                        if pd.notna(exit_date) and exit_date <= month_end:
                            # Employee exits during this month
                            if emp['joining_date'] >= month_start:
                                # Joined and exited in same month
                                salary = prorate_salary(emp['Monthly_Salary'], emp['joining_date'], exit_date, month_start, month_end)
                            else:
                                # Exited this month but joined earlier
                                salary = prorate_salary(emp['Monthly_Salary'], month_start, exit_date, month_start, month_end)
                        elif emp['joining_date'] >= month_start:
                            # Joined this month, no exit yet
                            salary = prorate_salary(emp['Monthly_Salary'], emp['joining_date'], month_end, month_start, month_end)
                        else:
                            # Regular full month
                            salary = emp['Monthly_Salary']
                        
                        # Add to department cost
                        dept_costs[emp['Department']] += salary
                
                # 3. Process employees on PIP
                for _, emp in df_pip.iterrows():
                    # Check if employee has joined by this month
                    if emp['joining_date'] <= month_end:
                        # Determine if employee will exit based on PIP
                        pip_exit = np.random.choice([True, False], p=[pip_exit_percentage/100, 1-pip_exit_percentage/100])
                        
                        # If PIP exit, assume exit after 30 days from pip_start_date
                        if pip_exit and pd.notna(emp['pip_start_date']):
                            expected_exit = emp['pip_start_date'] + pd.DateOffset(days=30)
                            
                            # Skip if employee has already exited before this month
                            if expected_exit < month_start:
                                continue
                            
                            # Calculate prorated salary
                            if expected_exit <= month_end:
                                # Employee exits during this month
                                if emp['joining_date'] >= month_start:
                                    # Joined and exited in same month
                                    salary = prorate_salary(emp['Monthly_Salary'], emp['joining_date'], expected_exit, month_start, month_end)
                                else:
                                    # Exited this month but joined earlier
                                    salary = prorate_salary(emp['Monthly_Salary'], month_start, expected_exit, month_start, month_end)
                            elif emp['joining_date'] >= month_start:
                                # Joined this month, no exit yet
                                salary = prorate_salary(emp['Monthly_Salary'], emp['joining_date'], month_end, month_start, month_end)
                            else:
                                # Regular full month
                                salary = emp['Monthly_Salary']
                        else:
                            # No PIP exit, treat as regular employee
                            if emp['joining_date'] >= month_start:
                                # Joined this month
                                salary = prorate_salary(emp['Monthly_Salary'], emp['joining_date'], month_end, month_start, month_end)
                            else:
                                # Regular full month
                                salary = emp['Monthly_Salary']
                        
                        # Add to department cost
                        dept_costs[emp['Department']] += salary
                
                # 4. Process open requisitions
                for _, req in df_requisition.iterrows():
                    # Check if requisition is filled during or before this month
                    if pd.notna(req['target_joining_date']) and req['target_joining_date'] <= month_end and req['target_joining_date'] >= month_start:
                        # Calculate monthly salary from annual salary
                        monthly_salary = req['Expected_Annual_Salary'] / 12
                        
                        # Calculate prorated salary for joining month
                        salary = prorate_salary(monthly_salary, req['target_joining_date'], month_end, month_start, month_end)
                        
                        # Add to department cost
                        dept_costs[req['Department']] += salary
                    elif pd.notna(req['target_joining_date']) and req['target_joining_date'] < month_start:
                        # Requisition filled before this month, include full salary
                        monthly_salary = req['Expected_Annual_Salary'] / 12
                        dept_costs[req['Department']] += monthly_salary
                
                # Add month column to projection dataframe
                monthly_projection[month_col] = pd.Series(dept_costs)
            
            # Add total row
            monthly_projection.loc['Total'] = monthly_projection.sum()
            
            # Create quarterly projection
            quarterly_projection = pd.DataFrame(index=monthly_projection.index)
            
            # Group months into quarters
            for month_start in projection_months:
                quarter = get_fiscal_quarter(month_start)
                month_col = get_month_year(month_start)
                
                if quarter not in quarterly_projection.columns:
                    quarterly_projection[quarter] = monthly_projection[month_col]
                else:
                    quarterly_projection[quarter] += monthly_projection[month_col]
            
            # Format values in Indian numbering system
            monthly_projection_formatted = monthly_projection.applymap(format_inr)
            quarterly_projection_formatted = quarterly_projection.applymap(format_inr)
            
            # Display projections
            st.header("Wage Cost Projections")
            
            tab1, tab2 = st.tabs(["Monthly Projection", "Quarterly Projection"])
            
            with tab1:
                st.subheader("Monthly Wage Cost Projection (INR)")
                st.dataframe(monthly_projection_formatted, use_container_width=True)
                st.markdown(download_excel(monthly_projection, "monthly_wage_projection.xlsx"), unsafe_allow_html=True)
            
            with tab2:
                st.subheader("Quarterly Wage Cost Projection (INR)")
                st.dataframe(quarterly_projection_formatted, use_container_width=True)
                st.markdown(download_excel(quarterly_projection, "quarterly_wage_projection.xlsx"), unsafe_allow_html=True)
            
            # Display summary statistics
            st.header("Summary Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_current_month = monthly_projection[get_month_year(start_date)].loc['Total']
                st.metric("Current Month Total", f"₹{format_inr(total_current_month)}")
            
            with col2:
                total_projection_end = monthly_projection[get_month_year(projection_months[-1])].loc['Total']
                percent_change = ((total_projection_end - total_current_month) / total_current_month) * 100
                st.metric("End Month Total", f"₹{format_inr(total_projection_end)}", 
                         f"{percent_change:.1f}% from start")
            
            with col3:
                avg_monthly_cost = monthly_projection.loc['Total'].mean()
                st.metric("Average Monthly Cost", f"₹{format_inr(avg_monthly_cost)}")
            
            # Department-wise trend chart
            st.subheader("Department-wise Monthly Cost Trend")
            
            # Prepare data for chart
            chart_data = monthly_projection.drop('Total')
            
            # Display chart
            st.line_chart(chart_data.T)
            
            st.success("Wage cost projections generated successfully!")

else:
    st.info("Please upload all four required files to generate wage cost projections.")

# Add app information
with st.expander("About this App"):
    st.markdown("""
    ### Wage Cost Projection App
    
    This app calculates wage cost projections based on:
    
    1. **Employee Wage Data**: Current employees and their salaries
    2. **Employees on Notice/Exited**: Employees who are leaving or have left
    3. **Employees on PIP**: Employees on Performance Improvement Plan
    4. **Open Requisitions**: Planned new hires
    
    The app calculates monthly and quarterly projections with proper pro-rating for:
    - Employees who join or exit during a month
    - PIP employees who may exit (based on probability)
    - New hires from open requisitions
    
    All values are displayed in INR using the Indian numbering system.
    """)

print("Streamlit app code generated successfully")