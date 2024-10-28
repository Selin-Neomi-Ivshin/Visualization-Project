
# Road Accidents Visualization Project
---
This project explores road accidents in Israel through interactive visualizations to uncover insights into how environmental, road, and temporal factors influence accident severity and frequency across various regions.

### [Click here](https://visualization-project-3clwzktcentxfmrnfqy8dh.streamlit.app/) to view the visualization (click on 'Yes, get this app back up!' in order to wake up the app)

---
## Project Overview:

- **Interactive Visualizations**: 
  - Developed with Plotly.js to create dynamic, interactive charts.
  - Visualizes accident patterns such as severity, crash types, and road conditions across different environments.

----

## Files in the Repository

1. **interactive_plots.html**  
   - HTML page with embedded interactive graphs.
   - Displays accident trends by severity, road width, and weather conditions.

2. **plots_to_html.py**  
   - Python script to generate visualizations and export them to HTML.

3. **SmallQ.py**  
   - Processes and queries the datasets to prepare data for visualizations. This file generates the visualization app.

4. **union_dataset_Trash.csv**  
   - Merged dataset containing:
     - **Accident Severity**: Light, Severe, Fatal
     - **Road Width**: Categories such as 5 to 7 meters, 7 to 10.5 meters, etc.
     - **Weather Conditions**: Clear, Rainy, Foggy, etc.
     - **Crash Types**: Front-to-front, Side-to-side, Pedestrian injury, etc.

5. **road2.jpg**  
   - A visual asset used to complement the theme of the project.

6. **Report.pdf**:
   - A detailed report summarizing the findings and analyses.  

---
## Key Objectives:

1. **Primary Question**:  
   - How do **environmental conditions** affect the severity and frequency of accidents?

2. **Secondary Questions**:
   - Does the **type of day** (weekday, holiday) influence accident frequency?
   - Is there a relationship between **road width** and **types of accidents**?
   - How do **road conditions** and width correlate with accident severity?
   - What is the **distribution of accidents** across regions by time of day?

---
## Data Sources:

- **Primary Dataset**: Extracted from the **Central Bureau of Statistics** in Israel.
- **Supplementary Dataset**: Additional data from the **Israel Road Authority**.

---
## Usage:

1. **Open the HTML Visualization**:  
   Open `interactive_plots.html` in a browser to explore the visualizations.

2. **Generate New Plots**:  
   Run `plots_to_html.py` to create or update visualizations.

3. **Process Data**:  
   Use `SmallQ.py` to clean and query the data from `union_dataset_Trash.csv`.
