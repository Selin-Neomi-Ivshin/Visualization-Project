import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import calendar
from plotly.subplots import make_subplots
import base64


# Load the dataset
file_path = 'union_dataset_Trash.csv'
df = pd.read_csv(file_path)

# Data Preparation
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = (df['Date'].dt.weekday + 1) % 7  # Shift weekday to start from Sunday (0) to Saturday (6)
df['Week'] = df['Date'].dt.isocalendar().week

# Color mapping for "Day type"
color_map = {
    'Sunday/Thursday': '#3182bd',
    'Monday/Tuesday/Wednesday': '#fec44f',
    'Weekend/Holiday': '#f03b20'
}

# Define the function to find valid combinations
def find_valid_combinations(data):
    severity_levels = data['Accident severity'].unique()
    valid_combinations = []
    # Exclude combinations with '-' or 'Unknown'
    for light in data['Light conditions'].unique():
        if light in ['-', 'Unknown']:
            continue
        for surface in data['Road surface'].unique():
            if surface in ['-', 'Unknown']:
                continue
            for condition in data['Road conditions'].unique():
                if condition in ['-', 'Unknown']:
                    continue
                for weather in data['Weather'].unique():
                    if weather in ['-', 'Unknown']:
                        continue
                    temp_data = data[(data['Light conditions'] == light) &
                                     (data['Road surface'] == surface) &
                                     (data['Road conditions'] == condition) &
                                     (data['Weather'] == weather)]
                    if all(temp_data['Accident severity'].isin([severity]).any() for severity in severity_levels):
                        valid_combinations.append((light, surface, condition, weather))
    return valid_combinations


# Get valid combinations
valid_combinations = find_valid_combinations(df)

# Define a list of colors for the bars
colors = ['blue', 'green', 'red', 'purple', 'orange', 'yellow', 'pink', 'grey', 'olive', 'cyan']
colors_map = dict(zip(valid_combinations, colors))

# Function to generate the calendar plot
def generate_calendar_plot(year, month, month_data):
    cal = calendar.Calendar(firstweekday=6).monthdayscalendar(year, month)
    z = [[0 if day == 0 else 1 for day in week] for week in cal]
    text = [['' if day == 0 else str(day) for day in week] for week in cal]
    hovertext = [['' for _ in week] for week in cal]
    colors_ = [['white' if day == 0 else 'lightgray' for day in week] for week in cal]

    for week_idx, week in enumerate(cal):
        for day_idx, day in enumerate(week):
            if day != 0:
                day_info = month_data[month_data['Day'] == day]
                if not day_info.empty:
                    day_type = day_info.iloc[0]['Day type']
                    num_accidents = day_info.shape[0]
                    colors_[week_idx][day_idx] = color_map[day_type]
                    hovertext[week_idx][day_idx] = f"Day type: {day_type}<br>Accidents: {num_accidents}"

    fig = go.Figure(data=go.Heatmap(
        z=z,
        text=text,
        hovertext=hovertext,
        x=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
        y=[f'Week {i + 1}' for i in range(len(cal))],
        colorscale=[[0, 'white'], [1, 'white']],
        showscale=False,
        hoverinfo='text'
    ))

    for week_idx, week in enumerate(cal):
        for day_idx, day in enumerate(week):
            if day != 0:
                fig.add_annotation(
                    x=day_idx,
                    y=week_idx,
                    text=str(day),
                    showarrow=False,
                    font=dict(color='black'),
                    bgcolor=colors_[week_idx][day_idx]
                )

    fig.update_layout(
        title=calendar.month_name[month],
        xaxis_nticks=5,
        yaxis_nticks=len(cal),
        height=250,
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True, autorange='reversed')  # Reverse y-axis to start with Week 1 at the top
    )
    return fig


def generate_normalized_stacked_bar_charts(data, category):
    # Define the specific order for the 'Road width' categories
    road_width_order = ["Up to 5 meters", "5 to 7 meters", "7 to 10.5 meters", "10.5 to 14 meters", "More than 14 meters"]

    # Convert 'Road width' to categorical with the specified order
    data['Road width'] = pd.Categorical(data['Road width'], categories=road_width_order, ordered=True)

    severity_order = ["Fatal", "Severe", "Light"]
    color_discrete_map = {'Fatal': '#f03b20', 'Severe': '#3182bd', 'Light': '#fec44f'}

    if category == "Accident severity":
        # Exclude unknown path types
        data_filtered = data[data['Path type'] != 'Unknown number of lanes']
        path_type_order = data_filtered['Path type'].unique().tolist()
        data_filtered['Path type'] = pd.Categorical(data_filtered['Path type'], categories=path_type_order, ordered=True)

        # Group by 'Path type'
        grouped_data = data_filtered.groupby(['Road width', 'Path type', category]).size().reset_index(name='Accident Count')
        pivot_data = grouped_data.pivot(index=['Road width', 'Path type'], columns=category, values='Accident Count').fillna(0)

        # Ensure the order of the severity levels
        pivot_data = pivot_data[severity_order]

        # Normalize the data to get ratios
        pivot_data_normalized = pivot_data.div(pivot_data.sum(axis=1), axis=0)

        # Filter out values less than 1%
        pivot_data_normalized = pivot_data_normalized.applymap(lambda x: x if x >= 0.01 else 0)

        # Create the normalized stacked bar chart with faceting by 'Path type'
        fig = px.bar(pivot_data_normalized.reset_index(),
                     x='Road width', y=pivot_data_normalized.columns,
                     facet_col='Path type', facet_col_wrap=4,  # Adjusted to 4 for one line
                     category_orders={'Path type': path_type_order, 'Road width': road_width_order, category: severity_order},
                     labels={'value': 'Accidents Ratio'},
                     color_discrete_map=color_discrete_map,
                     title=f"Ratio of Accidents by Road Width, Path Type, and {category.replace('_', ' ').title()}")

        # Ensure consistent font size across all x-axis labels
        fig.update_xaxes(tickfont={'size': 14})

        # Update layout
        fig.update_layout(
            barmode='stack',
            xaxis={'categoryorder': 'array', 'categoryarray': road_width_order},
            yaxis={'tickformat': ',.0%', 'tickfont': {'size': 18}},  # Increase y-axis tick font size
            legend=dict(title=dict(text="Accident severity", font=dict(size=20)), font=dict(size=18))  # Increase legend title and font size
        )
    else:
        # Original 'Crash type' code segment
        grouped_data = data.groupby(['Road width', category]).size().reset_index(name='Accident Count')
        pivot_data = grouped_data.pivot(index='Road width', columns=category, values='Accident Count').fillna(0)
        pivot_data_normalized = pivot_data.div(pivot_data.sum(axis=1), axis=0)
        pivot_data_normalized = pivot_data_normalized.applymap(lambda x: x if x >= 0.01 else 0)

        fig = go.Figure()
        sorted_data = pivot_data_normalized.apply(lambda row: row[row > 0].sort_values(), axis=1)
        for col in sorted_data.columns:
            fig.add_trace(go.Bar(
                y=sorted_data.index,
                x=sorted_data[col],
                name=col,
                orientation='h'
            ))

        # Annotation below the legend
        fig.add_annotation(
            text="You can hide unneeded stacks by clicking on its color",
            xref="paper", yref="paper",
            x=1.05, y=-0.15,
            showarrow=False,
            font=dict(size=16, color="grey"),
            xanchor='left', yanchor='top'
        )

        # Update layout to center the bars and add a range slider
        fig.update_layout(
            barmode='stack',
            title=dict(text=f"Ratio of Accidents by Crash Type and Road Width", font=dict(size=24)),
            xaxis_title=dict(text="Ratio of Crash Type per Road Width", font=dict(size=20)),
            yaxis_title=dict(text="Road Width", font=dict(size=20)),
            xaxis={
                'tickformat': ',.0%',
                'zeroline': True,
                'zerolinewidth': 2,
                'zerolinecolor': 'black',
                'tickfont': {'size': 18},
            },
            yaxis={'categoryorder': 'array', 'categoryarray': sorted_data.index, 'tickfont': {'size': 18}},
            legend=dict(title=dict(text="Crash type", font=dict(size=20)), font=dict(size=18)),
            margin=dict(l=50, r=50, t=80, b=150),
            xaxis_rangeslider_visible=True
        )

    return fig


st.set_page_config(layout="wide")

# Load the background image
bg_image_path = "road2.jpg"

# Encode the image in base64
with open(bg_image_path, "rb") as image_file:
    bg_image_base64 = base64.b64encode(image_file.read()).decode()

# Style for the header
st.markdown(
    f"""
    <style>
    .header {{
        position: relative;
        background-image: url("data:image/jpg;base64,{bg_image_base64}");
        background-size: cover;
        background-position: center;
        filter: brightness(0.7);
        padding: 20px;
        height: 100vh; /* Make the header take full viewport height */
        width: 100%; /* Ensure the header takes the full width */
        color: lightblue;
        font-family: 'Times New Roman', Times, serif;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .header h1 {{
        font-size: 3em;
        color: black; /* Set title color to black */
        margin-top: 0;
        margin-bottom: 20px; /* Add some space below the title */
        text-align: center; /* Center align the title */
    }}
    .header p {{
        font-size: 1.4em; /* Increase font size */
        font-weight: bold; /* Make the text bold */
        width: 100%; /* Make the paragraph take full width */
        margin: 0; /* Remove margins */
        color: white; /* Set paragraph color to white */
        background-color: rgba(0, 0, 0, 0.5); /* Add black background with 50% opacity */
        padding: 20px; /* Add padding for readability */
        border-radius: 10px;
        text-align: left; /* Align text to the left */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# HTML for the header
st.markdown(
    f"""
    <div class="header">
        <h1>How Road Accidents Are Affected By Environmental Characteristics</h1>
        <p>Israel, a small country with about 20,000 kilometers of roads, has a vehicle-to-people ratio of 450 per 1,000. Each vehicle averages 16,000 kilometers annually, totaling 50-70 billion kilometers. This density results in notable road congestion and accidents. This project analyzes road accidents in Israel, addressing key questions: How do environmental conditions affect accident severity? Does the day of the week influence accident numbers? Are crash rates different on various roads? Can we identify accident rate variations by time of day across areas? Our dashboard offers insights into these questions, revealing the relationship between environmental factors and road accidents in Israel.</p>
    </div>
    """,
    unsafe_allow_html=True
)


# Function to create the graph for a specific combination
def create_graph_for_combination(data, combination):
    temp_data = data[(data['Light conditions'] == combination[0]) &
                     (data['Road surface'] == combination[1]) &
                     (data['Road conditions'] == combination[2]) &
                     (data['Weather'] == combination[3])]
    counts = temp_data['Accident severity'].value_counts(normalize=True)
    return counts

# Setup Streamlit UI
st.title("Accident Severity Analysis Under Different Environmental Conditions")
st.subheader("Select combinations of environmental conditions to compare, only those yielding results are displayed. The default combination represents ideal conditions. You can choose as many combinations as you want.")

# Display a dropdown to select combinations
selected_combinations = st.multiselect("Select Combination:", valid_combinations, default=[('Daylight', 'Dry', 'No defect', 'Clear')], format_func=lambda x: f"Light: {x[0]}, Surface: {x[1]}, Condition: {x[2]}, Weather: {x[3]}")

# Create the bar chart
fig = make_subplots(specs=[[{"secondary_y": True}]])

for idx, combo in enumerate(selected_combinations):
    severity_counts = create_graph_for_combination(df, combo)
    show_legend = True  # Show legend only for the first bar of each combination
    for severity, count in severity_counts.items():
        legend_name = ''
        for i in range(len(combo)):
            if i == len(combo) - 1:
                legend_name += combo[i]
            else:
                legend_name += combo[i] + ', '
        fig.add_trace(go.Bar(name= legend_name, x=[severity], y=[count], marker_color=colors[idx % len(colors)], showlegend=show_legend), secondary_y=False)
        show_legend = False  # Subsequent bars won't show in the legend

fig.update_layout(
    title="Accident Severity for Selected Conditions",
    xaxis_title="Accident Severity",
    yaxis_title="Accidents ratio",
    barmode='group',
    legend_title='Light condition, Surface, Road condition, Weather'
)

st.plotly_chart(fig, use_container_width=True)

# Year selection
selected_year = st.selectbox("Select A Year To Filter The Plot By:", sorted(df['Year'].unique()), index=0)

# Filter data by the selected year
year_data = df[df['Year'] == selected_year]

# Bar chart for accident analysis
st.subheader("Average Accidents Number Per Day Type")

# Calculate the number of days for each "Day type"
day_type_days = year_data.groupby('Day type')['Date'].nunique().reset_index(name='Day Count')

# Calculate the number of accidents for each "Day type"
day_type_counts_year = year_data.groupby('Day type').size().reset_index(name='Accident Count')

# Merge the dataframes to perform the normalization
normalized_data = pd.merge(day_type_counts_year, day_type_days, on='Day type')
normalized_data['Normalized Count'] = normalized_data['Accident Count'] / normalized_data['Day Count']

# Update bar chart with normalized values
bar_fig = px.bar(normalized_data, x='Day type', y='Normalized Count', color='Day type',
                 color_discrete_map=color_map,
                 category_orders={'Day type': ['Sunday/Thursday', 'Monday/Tuesday/Wednesday', 'Weekend/Holiday']},
                 height=500,
                 width=600,  # Increase the width of the bar plot
                 labels={'Normalized Count': 'Average number of accidents per day'},
                 custom_data=['Accident Count'])

bar_fig.update_traces(
    hovertemplate='<b>%{x}</b><br>Normalized Accidents: %{y:.2f}<br>Total Accidents: %{customdata[0]}'
)

bar_fig.update_layout(
    xaxis_title={'text': 'Day type', 'font': {'size': 20}},  # Increase x-axis title size
    yaxis_title={'text': 'Average number of accidents per day', 'font': {'size': 20}},  # Increase y-axis title size
    xaxis_tickfont=dict(size=15),  # Increase the size of the bar titles
    legend=dict(title_font=dict(size=20), font=dict(size=18))  # Increase legend title and font size
)

if selected_year == 2020:
    bar_fig.update_yaxes(dtick=0.1)
elif selected_year == 2021:
    bar_fig.update_yaxes(dtick=0.2)
elif selected_year == 2022:
    bar_fig.update_yaxes(dtick=0.5)

# JavaScript for sticky behavior
st.markdown(
    """
    <style>
    .sticky-bar {
        position: sticky;
        top: 0;
        z-index: 100;
        background-color: white;
    }
    </style>
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        const barPlot = document.getElementById('bar-plot');
        const calendars = document.getElementById('calendars');

        function onScroll() {
            const rect = calendars.getBoundingClientRect();
            if (rect.top <= 0 && rect.bottom >= window.innerHeight) {
                barPlot.classList.add('sticky-bar');
            } else {
                barPlot.classList.remove('sticky-bar');
            }
        }

        window.addEventListener('scroll', onScroll);
    });
    </script>
    """,
    unsafe_allow_html=True,
)

# Display the bar chart and the calendar plots side by side
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])  # Increase the width of the first column

with col1:
    st.markdown('<div id="bar-plot">', unsafe_allow_html=True)
    st.plotly_chart(bar_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

calendar_columns = [col2, col3, col4]
calendar_idx = 0

st.markdown('<div id="calendars">', unsafe_allow_html=True)
for month in range(1, 13):
    month_data = year_data[year_data['Month'] == month]
    calendar_fig = generate_calendar_plot(selected_year, month, month_data)
    with calendar_columns[calendar_idx % 3]:
        st.plotly_chart(calendar_fig, use_container_width=True)
    calendar_idx += 1
st.markdown('</div>', unsafe_allow_html=True)

# Accident Severity Faceting
st.subheader("We Want To See The Effect Of Road Width On Accident Severity Or Crash Type:")

# Add a filter to select between "Accident severity" and "Crash type"
category = st.selectbox("Select The Feature You Want To Analyze:", ["Accident severity", "Crash type"])

fig = generate_normalized_stacked_bar_charts(df, category)
st.plotly_chart(fig)


def plot_accidents_by_time(df, faceted=False):
    # Prepare the DataFrame
    df_temp = df.copy()

    # Convert 'Hour' column to numeric float values
    def convert_hour_to_float(hour_str):
        try:
            hours, minutes = map(int, hour_str.split(':'))
            return hours + minutes / 60.0
        except:
            return None

    df_temp['Hour'] = df_temp['Hour'].apply(convert_hour_to_float)

    # Filter out invalid 'Hour' values
    df_temp = df_temp[(df_temp['Hour'] >= 0) & (df_temp['Hour'] < 24)].dropna(subset=['Hour'])

    # Group by 'Hour' and 'Area' to count the number of accidents
    hourly_accidents = df_temp.groupby(['Hour', 'Area']).size().reset_index(name='Accident Count')

    if faceted:
        # Create faceted line plot
        facet_fig = px.line(hourly_accidents, x='Hour', y='Accident Count',
                            facet_col='Area', facet_col_wrap=2,
                            title='Accidents by Time of Day and Area',
                            labels={'Hour': 'Time of Day (Hour)', 'Accident Count': 'Number of Accidents'})
        facet_fig.update_xaxes(matches='x', tickmode='linear', dtick=1, title_font=dict(size=20), tickfont=dict(size=18))
        facet_fig.update_yaxes(matches='y', dtick=6, title_font=dict(size=16), tickfont=dict(size=14))
        max_count = hourly_accidents['Accident Count'].max()
        facet_fig.update_yaxes(range=[0, max_count + (3 - max_count % 3)])
        facet_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=18)))
    else:
        # Create combined line plot
        facet_fig = px.line(hourly_accidents, x='Hour', y='Accident Count', color='Area',
                            title='Accidents by Time of Day and Area',
                            labels={'Hour': 'Time of Day (Hour)', 'Accident Count': 'Number of Accidents'})
        facet_fig.update_xaxes(tickmode='linear', dtick=1, title_font=dict(size=20), tickfont=dict(size=18))
        facet_fig.update_yaxes(dtick=6, title_font=dict(size=16), tickfont=dict(size=14))
        max_count = hourly_accidents['Accident Count'].max()
        facet_fig.update_yaxes(range=[0, max_count + (3 - max_count % 3)])

    # Update layout with the unified title and larger legend title settings
    facet_fig.update_layout(
        title=dict(text='Accidents by Time of Day and Area', font=dict(size=24)),
        legend=dict(title='Area', title_font=dict(size=22), font=dict(size=18))  # Increase legend title font size
    )

    return facet_fig


# Usage in Streamlit
st.subheader("Number Of Accidents In Different Areas In Israel By Hour")

# Toggle between faceted and combined views
view_type = st.radio("Select View Type:", ('Combined', 'Faceted'))
faceted_view = view_type == 'Faceted'

fig = plot_accidents_by_time(df, faceted=faceted_view)
st.plotly_chart(fig, use_container_width=True)