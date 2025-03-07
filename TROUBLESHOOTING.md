# Troubleshooting Guide

## Common Issues and Solutions

### Excel File Issues

If you're encountering errors related to the Excel file:

1. **Run the inspection script first:**
   ```
   python inspect_excel.py
   ```
   This will show you the structure of your Excel file including column names.

2. **Column name mismatches:**
   - The dashboard expects columns named: `checkDate`, `Total`, `OCR`, `30 day bill`, `Δ Revenue`, `Δ 30 day`
   - The code will attempt to map your columns based on position, but if your Excel structure is different, you may need to modify the column mapping in app.py

3. **Missing data:**
   - Ensure your Excel file has data in all required columns
   - The code will calculate `Δ Revenue` and `Δ 30 day` if they're missing

### Installation Issues

If you're having trouble with package installation:

1. **Run pip install manually:**
   ```
   pip install pandas numpy matplotlib seaborn plotly dash dash-bootstrap-components openpyxl statsmodels scikit-learn
   ```

2. **Python version compatibility:**
   - This dashboard works best with Python 3.7+
   - Check your Python version with: `python --version`

3. **Upgrade pip:**
   ```
   pip install --upgrade pip
   ```

### Dashboard Runtime Issues

If the dashboard runs but has display issues:

1. **Browser compatibility:**
   - Use Chrome or Firefox for best results
   - Clear your browser cache if visualizations don't render

2. **Data format issues:**
   - Ensure date values in the Excel are in a consistent format
   - Check that percentage values are properly formatted

3. **Port conflicts:**
   - If port 8050 is already in use, modify app.py to use a different port
   - Find line: `app.run_server(debug=True)` 
   - Change to: `app.run_server(debug=True, port=8051)` (or another available port)

### Advanced Solutions

For more advanced issues:

1. **Debug mode:**
   - The dashboard runs in debug mode by default
   - Check the console output for detailed error messages

2. **Manual data processing:**
   - If automatic column mapping fails, you can manually prepare your Excel file:
     - Rename columns to match expected names
     - Ensure proper data types (dates as dates, numbers as numbers)
     - Save as a new Excel file and use that

3. **Log inspection:**
   - Check application logs for detailed error messages
   - Windows: Look in Event Viewer under Application logs