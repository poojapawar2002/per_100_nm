import pickle

with open('voyage_dict.pkl', 'rb') as f:
    voyage_dict = pickle.load(f)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np

st.set_page_config(layout="wide")

# Vessel name mapping
vessel_names = {
    9967433: "MH Perseus",
    9967524: "PISCES",
    9967419: "CAPELLA",
    9967495: "CETUS",
    9967457: "CASSIOPEIA",
    9967536: "PYXIS",
    9967469: "Cenataurus",
    9967483: "CHARA",
    9967445: "CARINA"
}

df = pd.read_csv("Updated_autolog_complete_input_ideal_power_foc_7000series_except1004.csv")

additional_vessel_df = pd.read_csv("../Data/inputfiles/autolog_input_with_dist_1004.csv")

additional_vessel_df["StartDateUTC"] = pd.to_datetime(additional_vessel_df["StartDateUTC"], format="%d-%m-%Y %H:%M")
additional_vessel_df["EndDateUTC"] = pd.to_datetime(additional_vessel_df["EndDateUTC"], format="%d-%m-%Y %H:%M")
additional_vessel_df['MeanDraft'] = (additional_vessel_df['DraftAftTele'] + additional_vessel_df['DraftFwdTele']) / 2

df = df[df["IMO"]!=9967457]

df = df[df['ideal_power'] > 0]

df["ideal_foc"] = df["ideal_foc_hr"]

df['MeanDraft'] = (df['DraftAftTele'] + df['DraftFwdTele']) / 2

df["StartDateUTC"] = pd.to_datetime(df["StartDateUTC"], format="%d-%m-%Y %H:%M")
df["EndDateUTC"] = pd.to_datetime(df["EndDateUTC"], format="%d-%m-%Y %H:%M")

df.sort_values(by=["IMO", "StartDateUTC"], inplace=True)

# Select Vessel
imos = df["IMO"].unique().tolist()
selected_imo = st.selectbox("Select Vessel", imos, format_func=lambda x: vessel_names.get(x, x))

# Filter dataframe for that IMO
df_imo = df[df['IMO'] == selected_imo].copy()

# Get voyage intervals
voyage_intervals = voyage_dict.get(selected_imo, [])

# Prepare list to hold voyage summary
voyage_summary = []

for idx, (start, end) in enumerate(voyage_intervals):
    # Filter data within the voyage time range
    voyage_df = df_imo[
        (df_imo['StartDateUTC'] >= start) & 
        (df_imo['EndDateUTC'] <= end)
    ]
    
    # Calculate totals
    total_DistanceOGAct = voyage_df['DistanceOGAct'].sum()
    
    # Skip voyages with 0 distance
    if total_DistanceOGAct == 0:
        continue
    
    # Calculate detailed metrics for the entire voyage
    # voyage_df["MeanDraft"] = (voyage_df["DraftAftTele"] + voyage_df["DraftFwdTele"]) / 2
    
    total_FuelMassCons = (voyage_df['MEFuelMassCons']/1000).sum()
    total_ideal_foc = voyage_df['ideal_foc'].sum()
    avg_draft = voyage_df["MeanDraft"].mean()
    avg_wind_speed = voyage_df['TrueWindSpeedWP'].mean()
    wavg_ME1ShaftPower = (voyage_df['ME1ShaftPower'] * voyage_df["ME1RunningHoursMinute"]).sum() / voyage_df["ME1RunningHoursMinute"].sum()
    wavg_ideal_power = (voyage_df['ideal_power'] * voyage_df["ME1RunningHoursMinute"]).sum() / voyage_df["ME1RunningHoursMinute"].sum()
    total_running_hours = voyage_df["ME1RunningHoursMinute"].sum()/60

    # Append to summary list
    voyage_summary.append({
        "Voyage #": idx + 1,
        "Voyage Start": start,
        "Voyage End": end,
        "Total DistanceOGAct": round(total_DistanceOGAct, 2),
        "Total Running Hours": round(total_running_hours, 2),
        "Mean Draft": round(avg_draft, 2),
        "Average SpeedOG": round(voyage_df['SpeedOG'].mean(), 2),
        "Average calculated SpeedTW": round(voyage_df['calculated_stw'].mean(), 2),
        "Average Wind Speed": round(avg_wind_speed, 2),
        "WAvg ME1ShaftPower": round(wavg_ME1ShaftPower, 2),
        "WAvg Ideal Power": round(wavg_ideal_power, 2),
        "Total MEFuelMassCons": round(total_FuelMassCons, 2),
        "Total Ideal FOC": round(total_ideal_foc, 2),
        "Data Points": len(voyage_df)
    })

# Create summary DataFrame
summary_df = pd.DataFrame(voyage_summary)

# Display in Streamlit
st.write(f"Voyage Summary for Vessel: {vessel_names.get(selected_imo, selected_imo)}")
st.dataframe(summary_df)

# Section for per 100 nm breakdown
st.write("## Per 100 nautical miles breakdown for Selected Voyage")

# Select voyage for detailed breakdown
if len(voyage_summary) > 0:
    voyage_numbers = [f"Voyage {voyage['Voyage #']}" for voyage in voyage_summary]
    selected_voyage = st.selectbox("Select Voyage for 100 nautical miles breakdown", voyage_numbers)
    selected_voyage_num = int(selected_voyage.split()[1])
    
    # Find the corresponding voyage interval (accounting for skipped voyages)
    selected_voyage_idx = None
    for voyage in voyage_summary:
        if voyage['Voyage #'] == selected_voyage_num:
            selected_voyage_idx = voyage['Voyage #'] - 1
            break

if selected_voyage_idx is not None and selected_voyage_idx < len(voyage_intervals):
    start, end = voyage_intervals[selected_voyage_idx]
    
    # Filter data for selected voyage
    voyage_df = df_imo[
        (df_imo['StartDateUTC'] >= start) & 
        (df_imo['EndDateUTC'] <= end)
    ].copy().sort_values('StartDateUTC')
    
    if len(voyage_df) > 0:
        # Calculate cumulative DistanceOGAct
        voyage_df['CumulativeDistanceOGAct'] = voyage_df['DistanceOGAct'].cumsum()
        
        # Create 100-knot sections
        sections_100_nm = []
        current_section = 1
        section_start_idx = 0
        
        def calculate_section_metrics(section_df):
            """Helper function to calculate all section metrics"""
            # section_df["MeanDraft"] = (section_df["DraftAftTele"] + section_df["DraftFwdTele"]) / 2
            
            section_DistanceOGAct = section_df['DistanceOGAct'].sum()
            section_avg_draft = section_df["MeanDraft"].mean()
            section_avg_wind_speed = section_df['TrueWindSpeedWP'].mean()
            section_avg_SpeedOG = section_df['SpeedOG'].mean()
            section_FuelMassCons = (section_df['MEFuelMassCons']/1000).sum()
            section_ideal_foc = section_df['ideal_foc'].sum()
            section_wavg_ME1ShaftPower = (section_df['ME1ShaftPower'] * section_df["ME1RunningHoursMinute"]).sum() / section_df["ME1RunningHoursMinute"].sum()
            section_wavg_ideal_power = (section_df['ideal_power'] * section_df["ME1RunningHoursMinute"]).sum() / section_df["ME1RunningHoursMinute"].sum()
            section_start_time = section_df['StartDateUTC'].iloc[0]
            section_end_time = section_df['EndDateUTC'].iloc[-1]
            
            return {
                "Section Start": section_start_time,
                "Section End": section_end_time,
                "DistanceOGAct": round(section_DistanceOGAct, 2),
                "Total Running Hours": round(section_df["ME1RunningHoursMinute"].sum() / 60, 2),
                "Mean Draft": round(section_avg_draft, 2),
                "Average SpeedOG": round(section_avg_SpeedOG, 2),
                "Average Calculated SpeedTW": round(section_df['calculated_stw'].mean(), 2),
                "Average Wind Speed": round(section_avg_wind_speed, 2), 
                "WAvg Power Delivered Actual": round(section_wavg_ME1ShaftPower, 2),
                "WAvg Ideal Power": round(section_wavg_ideal_power, 2),
                "MEFuelMassCons": round(section_FuelMassCons, 2),
                "Ideal FOC": round(section_ideal_foc, 2), 
                "Data Points": len(section_df)
            }
        
        for i, row in voyage_df.iterrows():
            cumulative_speed = row['CumulativeDistanceOGAct']
            
            # Check if we've completed a 100-knot section
            if cumulative_speed >= (current_section * 100):
                # Get data for this section
                section_df = voyage_df.iloc[section_start_idx:voyage_df.index.get_loc(i)+1]
                
                # Calculate section metrics
                metrics = calculate_section_metrics(section_df)
                
                section_data = {f"Voyage {selected_voyage_num} - Section": current_section}
                section_data.update(metrics)
                
                sections_100_nm.append(section_data)
                
                # Move to next section
                current_section += 1
                section_start_idx = voyage_df.index.get_loc(i) + 1
        
        # Handle remaining data (last incomplete section)
        if section_start_idx < len(voyage_df):
            remaining_df = voyage_df.iloc[section_start_idx:]
            
            if len(remaining_df) > 0:
                # Calculate metrics for incomplete section
                metrics = calculate_section_metrics(remaining_df)
                
                # For incomplete section, scale FOC values to 100nm
                actual_distance = metrics["DistanceOGAct"]
                if actual_distance > 0:
                    scaling_factor = 100 / actual_distance
                    metrics["MEFuelMassCons"] = round(metrics["MEFuelMassCons"] * scaling_factor, 2)
                    metrics["Ideal FOC"] = round(metrics["Ideal FOC"] * scaling_factor, 2)
                    # metrics["DistanceOGAct"] = 100  # Set to 100 for incomplete section
                
                section_data = {f"Voyage {selected_voyage_num} - Section": f"{current_section}"}
                section_data.update(metrics)
                
                sections_100_nm.append(section_data)
        
        # Create sections DataFrame
        if sections_100_nm:
            sections_df = pd.DataFrame(sections_100_nm)
            st.write(f"#### {selected_voyage} - 100 nautical miles sections")
            st.dataframe(sections_df)

# NEW SECTION: Time Series Graphs for All Vessels
st.write("## Time Series Analysis - All Vessels Combined")

# Add filters for the graphs
st.write("### Filters for Graphs")
col1, col2, col3, col4 = st.columns(4)

with col1:
    draft_min = st.number_input("Min Mean Draft (m)", min_value=df["MeanDraft"].min(), max_value=df["MeanDraft"].max(), value=df["MeanDraft"].min(), step=0.5)
    draft_max = st.number_input("Max Mean Draft (m)", min_value=df["MeanDraft"].min(), max_value=df["MeanDraft"].max(), value=df["MeanDraft"].max(), step=0.5)

with col2:
    speedog_min = st.number_input("Min Speed OG (knots)", min_value=df["SpeedOG"].min(), max_value=df["SpeedOG"].max(), value=df["SpeedOG"].min(), step=0.5)
    speedog_max = st.number_input("Max Speed OG (knots)", min_value=df["SpeedOG"].min(), max_value=df["SpeedOG"].max(), value=df["SpeedOG"].max(), step=0.5)

with col3:
    speedtw_min = st.number_input("Min Speed TW (knots)", min_value=df["SpeedTW"].min(), max_value=df["SpeedTW"].max(), value=df["SpeedTW"].min(), step=0.5)
    speedtw_max = st.number_input("Max Speed TW (knots)", min_value=df["SpeedTW"].min(), max_value=df["SpeedTW"].max(), value=df["SpeedTW"].max(), step=0.5)

with col4:
    wind_min = st.number_input("Min Wind Speed (m/s)", min_value=df["TrueWindSpeedWP"].min(), max_value=df["TrueWindSpeedWP"].max(), value=df["TrueWindSpeedWP"].min(), step=0.5)
    wind_max = st.number_input("Max Wind Speed (m/s)", min_value=df["TrueWindSpeedWP"].min(), max_value=df["TrueWindSpeedWP"].max(), value=df["TrueWindSpeedWP"].max(), step=0.5)

# Collect all section data for all vessels
all_sections_data = []

# Apply filters
df = df[
    (df['MeanDraft'] >= draft_min) & (df['MeanDraft'] <= draft_max) &
    (df['SpeedOG'] >= speedog_min) & (df['SpeedOG'] <= speedog_max) &
    (df['SpeedTW'] >= speedtw_min) & (df['SpeedTW'] <= speedtw_max) &
    (df['TrueWindSpeedWP'] >= wind_min) & (df['TrueWindSpeedWP'] <= wind_max)
].copy()

additional_vessel_df = additional_vessel_df[
    (additional_vessel_df['MeanDraft'] >= draft_min) & (additional_vessel_df['MeanDraft'] <= draft_max) &
    (additional_vessel_df['SpeedOG'] >= speedog_min) & (additional_vessel_df['SpeedOG'] <= speedog_max) &
    (additional_vessel_df['SpeedTW'] >= speedtw_min) & (additional_vessel_df['SpeedTW'] <= speedtw_max) &
    (additional_vessel_df['TrueWindSpeedWP'] >= wind_min) & (additional_vessel_df['TrueWindSpeedWP'] <= wind_max)
].copy()

# Define 8 distinct colors for the vessels (you can change these)
# vessel_colors = [
#     '#1f77b4',  # muted blue
#     '#ff7f0e',  # safety orange
#     '#2ca02c',  # cooked asparagus green
#     '#d62728',  # brick red
#     '#9467bd',  # muted purple
#     '#8c564b',  # chestnut brown
#     '#e377c2',  # raspberry yogurt pink
#     '#7f7f7f',  # middle gray
#     '#bcbd22',  # curry yellow-green
#     '#17becf'   # blue-teal
# ]


# Process all vessels
for vessel_idx, imo in enumerate(imos):
    df_vessel = df[df['IMO'] == imo].copy()
    vessel_voyage_intervals = voyage_dict.get(imo, [])
    
    for voyage_idx, (start, end) in enumerate(vessel_voyage_intervals):
        # Filter data within the voyage time range
        voyage_df = df_vessel[
            (df_vessel['StartDateUTC'] >= start) & 
            (df_vessel['EndDateUTC'] <= end)
        ].copy().sort_values('StartDateUTC')
        
        # Skip voyages with 0 distance
        total_DistanceOGAct = voyage_df['DistanceOGAct'].sum()
        if total_DistanceOGAct == 0:
            continue
        
        # Calculate cumulative DistanceOGAct
        voyage_df['CumulativeDistanceOGAct'] = voyage_df['DistanceOGAct'].cumsum()
        
        # Create 100-nm sections for this voyage
        current_section = 1
        section_start_idx = 0
        
        def calculate_section_metrics_for_graph(section_df, vessel_imo, voyage_num):
            """Helper function to calculate section metrics for graphing"""
            # section_df["MeanDraft"] = (section_df["DraftAftTele"] + section_df["DraftFwdTele"]) / 2
            
            section_FuelMassCons = (section_df['MEFuelMassCons']/1000).sum()
            section_ideal_foc = section_df['ideal_foc'].sum()
            section_wavg_ME1ShaftPower = (section_df['ME1ShaftPower'] * section_df["ME1RunningHoursMinute"]).sum() / section_df["ME1RunningHoursMinute"].sum()
            section_wavg_ideal_power = (section_df['ideal_power'] * section_df["ME1RunningHoursMinute"]).sum() / section_df["ME1RunningHoursMinute"].sum()
            section_start_time = section_df['StartDateUTC'].iloc[0]
            section_end_time = section_df['EndDateUTC'].iloc[-1]
            section_mid_time = section_start_time + (section_end_time - section_start_time) / 2
            
            # Calculate averages for filtering
            section_avg_draft = section_df["MeanDraft"].mean()
            section_avg_speedog = section_df['SpeedOG'].mean()
            section_avg_speedtw = section_df['calculated_stw'].mean()
            section_avg_wind = section_df['TrueWindSpeedWP'].mean()
            
            return {
                "vessel_imo": vessel_imo,
                "vessel_name": vessel_names.get(vessel_imo, vessel_imo),
                "voyage_num": voyage_num,
                "section_mid_time": section_mid_time,
                "section_start_time": section_start_time,
                "section_end_time": section_end_time,
                "MEFuelMassCons_actual": section_FuelMassCons,
                "MEFuelMassCons_ideal": section_ideal_foc,
                "Power_actual": section_wavg_ME1ShaftPower,
                "Power_ideal": section_wavg_ideal_power,
                "section_num": current_section,
                "avg_draft": section_avg_draft,
                "avg_speedog": section_avg_speedog,
                "avg_speedtw": section_avg_speedtw,
                "avg_wind": section_avg_wind
            }
        
        for i, row in voyage_df.iterrows():
            cumulative_speed = row['CumulativeDistanceOGAct']
            
            # Check if we've completed a 100-nm section
            if cumulative_speed >= (current_section * 100):
                # Get data for this section
                section_df = voyage_df.iloc[section_start_idx:voyage_df.index.get_loc(i)+1]
                
                # Calculate section metrics
                metrics = calculate_section_metrics_for_graph(section_df, imo, voyage_idx + 1)
                all_sections_data.append(metrics)
                
                # Move to next section
                current_section += 1
                section_start_idx = voyage_df.index.get_loc(i) + 1
        
        # Handle remaining data (last incomplete section)
        if section_start_idx < len(voyage_df):
            remaining_df = voyage_df.iloc[section_start_idx:]
            
            if len(remaining_df) > 0:
                # Calculate metrics for incomplete section
                metrics = calculate_section_metrics_for_graph(remaining_df, imo, voyage_idx + 1)
                
                # Scale FOC values to 100nm for incomplete section
                actual_distance = remaining_df['DistanceOGAct'].sum()
                if actual_distance > 0:
                    scaling_factor = 100 / actual_distance
                    metrics["MEFuelMassCons_actual"] = metrics["MEFuelMassCons_actual"] * scaling_factor
                    metrics["MEFuelMassCons_ideal"] = metrics["MEFuelMassCons_ideal"] * scaling_factor
                
                all_sections_data.append(metrics)

# Create graphs if we have data
# Create graphs if we have data
if all_sections_data:
    # Convert to DataFrame
    graph_df = pd.DataFrame(all_sections_data)
    
    # Process additional vessel data (NEW SECTION)
    additional_sections = []
    
    # Get voyage intervals for additional vessel
    additional_voyages = voyage_dict.get(9967457, [])
    
    for voyage_idx, (start, end) in enumerate(additional_voyages):
        # Filter data within voyage time range
        voyage_df = additional_vessel_df[
            (additional_vessel_df['StartDateUTC'] >= start) & 
            (additional_vessel_df['EndDateUTC'] <= end)
        ].copy().sort_values('StartDateUTC')
        
        if len(voyage_df) == 0:
            continue
            
        # Calculate cumulative distance
        voyage_df['CumulativeDistanceOGAct'] = voyage_df['DistanceOGAct'].cumsum()
        
        # Create sections (same logic as main vessels)
        current_section = 1
        section_start_idx = 0
        
        for i, row in voyage_df.iterrows():
            if row['CumulativeDistanceOGAct'] >= (current_section * 100):
                section_df = voyage_df.iloc[section_start_idx:voyage_df.index.get_loc(i)+1]
                
                metrics = {
                    "vessel_imo": 9967457,
                    "vessel_name": vessel_names[9967457],
                    "voyage_num": voyage_idx + 1,
                    "section_mid_time": section_df['StartDateUTC'].iloc[0] + (section_df['EndDateUTC'].iloc[-1] - section_df['StartDateUTC'].iloc[0])/2,
                    "section_start_time": section_df['StartDateUTC'].iloc[0],
                    "section_end_time": section_df['EndDateUTC'].iloc[-1],
                    "MEFuelMassCons_actual": (section_df['MEFuelMassCons']/1000).sum(),
                    "MEFuelMassCons_ideal": np.nan,
                    "Power_actual": (section_df['ME1ShaftPower'] * section_df["ME1RunningHoursMinute"]).sum() / section_df["ME1RunningHoursMinute"].sum(),
                    "Power_ideal": np.nan,
                    "section_num": current_section,
                    "avg_draft": section_df["MeanDraft"].mean(),
                    "avg_speedog": section_df['SpeedOG'].mean(),
                    "avg_speedtw": np.nan,
                    "avg_wind": section_df['TrueWindSpeedWP'].mean()
                }
                additional_sections.append(metrics)
                
                current_section += 1
                section_start_idx = voyage_df.index.get_loc(i) + 1
        
        # # Handle incomplete last section
        # if section_start_idx < len(voyage_df):
        #     remaining_df = voyage_df.iloc[section_start_idx:]
        #     if len(remaining_df) > 0:
        #         metrics = {
        #             # ... same fields as above ...
        #             "section_num": current_section,
        #             # Scale values to 100nm if needed
        #             "MEFuelMassCons_actual": (remaining_df['MEFuelMassCons']/1000).sum() * (100/remaining_df['DistanceOGAct'].sum()) if remaining_df['DistanceOGAct'].sum() > 0 else 0,
        #             # ... other metrics ...
        #         }
        #         additional_sections.append(metrics)
    
    # Combine with main data
    combined_df = pd.concat([graph_df, pd.DataFrame(additional_sections)])
    filtered_df = combined_df
    
    # REST OF YOUR EXISTING GRAPHING CODE CONTINUES HERE
    # (th
       
    if len(filtered_df) > 0:
        # Create custom labels for hover and annotations
        filtered_df['x_label'] = filtered_df.apply(lambda row: f"{row['vessel_name']}-V{int(row['voyage_num'])}-S{int(row['section_num'])}", axis=1)
        filtered_df['hover_text'] = filtered_df.apply(lambda row: f"Vessel: {row['vessel_name']}<br>Voyage {int(row['voyage_num'])}, Section {int(row['section_num'])}<br>{row['section_mid_time'].strftime('%Y-%m-%d %H:%M')}<br>Draft: {row['avg_draft']:.1f}m<br>Speed OG: {row['avg_speedog']:.1f}kn<br>Speed TW: {row['avg_speedtw']:.1f}kn<br>Wind: {row['avg_wind']:.1f}m/s", axis=1)
        
        # Sort by time for proper chronological ordering
        filtered_df = filtered_df.sort_values('section_mid_time')
        
        # Get unique vessels for color assignment
        unique_vessels = filtered_df['vessel_imo'].unique()
        
    
        color_palette = [
    ("#1F77B4", "#AEC7E8"),  # Blue
    ("#FF7F0E", "#FFBB78"),  # Orange
    ("#2CA02C", "#98DF8A"),  # Green
    ("#D62728", "#FF9896"),  # Red
    ("#9467BD", "#C5B0D5"),  # Purple
    ("#8C564B", "#C49C94"),  # Brown
    ("#E377C2", "#F7B6D2"),  # Pink
    ("#7F7F7F", "#C7C7C7"),  # Gray
    ("#BCBD22", "#DBDB8D"),  # Olive
    ("#17BECF", "#9EDAE5"),  # Cyan
    ("#1A55FF", "#9BB8FF"),  # Bright Blue
    ("#FF4C4C", "#FF9999"),  # Coral
    ("#00B159", "#70E6B1"),  # Emerald
    ("#F28500", "#FFC266"),  # Pumpkin
    ("#8E44AD", "#D7BDE2"),  # Amethyst
    ("#2C3E50", "#95A5A6"),
    ("#03402A","#469777"),  # Midnight Blue (with Silver)
]


        vessel_color_map = {}
        for i, vessel in enumerate(unique_vessels):
            base_color, light_color = color_palette[i % len(color_palette)]
            
            # Convert to RGB
            def hex_to_rgb(hex):
                return tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            
            base_rgb = hex_to_rgb(base_color)
            light_rgb = hex_to_rgb(light_color)
            
            vessel_color_map[vessel] = {
                # Actual elements
                'actual_point': f'rgb{base_rgb}',
                'actual_trend': f'rgb{tuple(max(0, int(c * 0.8)) for c in base_rgb)}',  # 20% darker
                
                # Ideal elements (use light version)
                'ideal_point': f'rgb{light_rgb}',
                'ideal_trend': f'rgb{tuple(min(255, int(c * 1.2)) for c in light_rgb)}'  # 20% lighter
            }
        
        # Graph 1: Fuel Consumption (Actual vs Ideal)
        fig1 = go.Figure()
        
        # Add traces for each vessel - Actual FOC
        for vessel in unique_vessels:
            vessel_data = filtered_df[filtered_df['vessel_imo'] == vessel].sort_values('section_mid_time')
            vessel_name = vessel_names.get(vessel, vessel)
            colors = vessel_color_map[vessel]
            
            if len(vessel_data) > 1:  # Need at least 2 points for trendline
                # Convert datetime to numeric for trendline calculation
                x_numeric = pd.to_numeric(vessel_data['section_mid_time'])
                
                # Calculate linear trendline for actual FOC
                z_actual = np.polyfit(x_numeric, vessel_data['MEFuelMassCons_actual'], 1)
                p_actual = np.poly1d(z_actual)
                trendline_actual = p_actual(x_numeric)

                if vessel != 9967457:
                
                    # Calculate linear trendline for ideal FOC
                    z_ideal = np.polyfit(x_numeric, vessel_data['MEFuelMassCons_ideal'], 1)
                    p_ideal = np.poly1d(z_ideal)
                    trendline_ideal = p_ideal(x_numeric)
            
            
            # Actual FOC data points
            fig1.add_trace(go.Scatter(
                x=vessel_data['section_mid_time'],
                y=vessel_data['MEFuelMassCons_actual'],
                mode='markers',
                name=f'{vessel_name} - Actual FOC',
                line=dict(color=colors['actual_point'], width=2),  # Modified
                marker=dict(color=colors['actual_point'], size=8, symbol='circle'),  # Modified
                hovertemplate='%{customdata}<br>Actual FOC: %{y:.2f} MT<extra></extra>',
                customdata=vessel_data['hover_text'],
                legendgroup=f'vessel_{vessel}',
                showlegend=True
            ))

            # Actual FOC trendline
            fig1.add_trace(go.Scatter(
                x=vessel_data['section_mid_time'],
                y=trendline_actual,
                mode='lines',
                name=f'{vessel_name} - Actual Trend',
                line=dict(color=colors['actual_trend'], width=3),  # Modified
                hovertemplate='Actual FOC Trend: %{y:.2f} MT<extra></extra>',
                legendgroup=f'vessel_{vessel}',
                showlegend=True,
                opacity=0.8
            ))

            if vessel != 9967457:
                # Ideal FOC data points
                fig1.add_trace(go.Scatter(
                    x=vessel_data['section_mid_time'],
                    y=vessel_data['MEFuelMassCons_ideal'],
                    mode='markers',
                    name=f'{vessel_name} - Ideal FOC',
                    line=dict(color=colors['ideal_point'], width=2, dash='dash'),  # Modified
                    marker=dict(color=colors['ideal_point'], size=8, symbol='diamond'),  # Modified
                    hovertemplate='%{customdata}<br>Ideal FOC: %{y:.2f} MT<extra></extra>',
                    customdata=vessel_data['hover_text'],
                    legendgroup=f'vessel_{vessel}',
                    showlegend=True
                ))

                # Ideal FOC trendline
                fig1.add_trace(go.Scatter(
                    x=vessel_data['section_mid_time'],
                    y=trendline_ideal,
                    mode='lines',
                    name=f'{vessel_name} - Ideal Trend',
                    line=dict(color=colors['ideal_trend'], width=3),  # Modified
                    hovertemplate='Ideal FOC Trend: %{y:.2f} MT<extra></extra>',
                    legendgroup=f'vessel_{vessel}',
                    showlegend=True,
                    opacity=0.8
                ))
        
        fig1.update_layout(
            title='Fuel Consumption Over Time - All Vessels',
            xaxis_title='Time',
            yaxis_title='Fuel Consumption (MT per 100nm)',
            hovermode='closest',
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            height=700,
            margin=dict(r=200)
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Graph 2: Power (Actual vs Ideal)
        fig2 = go.Figure()
        
        # Add traces for each vessel - Actual Power
        for vessel in unique_vessels:
            vessel_data = filtered_df[filtered_df['vessel_imo'] == vessel].sort_values('section_mid_time')
            vessel_name = vessel_names.get(vessel, vessel)
            colors = vessel_color_map[vessel]
            
            if len(vessel_data) > 1:  # Need at least 2 points for trendline
                # Convert datetime to numeric for trendline calculation
                x_numeric = pd.to_numeric(vessel_data['section_mid_time'])
                
                # Calculate linear trendline for actual Power
                z_actual = np.polyfit(x_numeric, vessel_data['Power_actual'], 1)
                p_actual = np.poly1d(z_actual)
                trendline_actual = p_actual(x_numeric)

                if  vessel != 9967457:
                
                    # Calculate linear trendline for ideal Power
                    z_ideal = np.polyfit(x_numeric, vessel_data['Power_ideal'], 1)
                    p_ideal = np.poly1d(z_ideal)
                    trendline_ideal = p_ideal(x_numeric)
            
            # Actual Power data points
            
            fig2.add_trace(go.Scatter(
                x=vessel_data['section_mid_time'],
                y=vessel_data['Power_actual'],
                mode='markers',
                name=f'{vessel_name} - Actual Power',
                line=dict(color=colors['actual_point'], width=2),  # Modified
                marker=dict(color=colors['actual_point'], size=8, symbol='circle'),  # Modified
                hovertemplate='%{customdata}<br>Actual Power: %{y:.2f} kW<extra></extra>',
                customdata=vessel_data['hover_text'],
                legendgroup=f'vessel_{vessel}',
                showlegend=True
            ))

            # Actual Power trendline
            fig2.add_trace(go.Scatter(
                x=vessel_data['section_mid_time'],
                y=trendline_actual,
                mode='lines',
                name=f'{vessel_name} - Actual Trend',
                line=dict(color=colors['actual_trend'], width=3),  # Modified
                hovertemplate='Actual Power Trend: %{y:.2f} kW<extra></extra>',
                legendgroup=f'vessel_{vessel}',
                showlegend=True,
                opacity=0.8
            ))

            if vessel != 9967457:

                # Ideal Power data points
                fig2.add_trace(go.Scatter(
                    x=vessel_data['section_mid_time'],
                    y=vessel_data['Power_ideal'],
                    mode='markers',
                    name=f'{vessel_name} - Ideal Power',
                    line=dict(color=colors['ideal_point'], width=2, dash='dash'),  # Modified
                    marker=dict(color=colors['ideal_point'], size=8, symbol='diamond'),  # Modified
                    hovertemplate='%{customdata}<br>Ideal Power: %{y:.2f} kW<extra></extra>',
                    customdata=vessel_data['hover_text'],
                    legendgroup=f'vessel_{vessel}',
                    showlegend=True
                ))

                # Ideal Power trendline
                fig2.add_trace(go.Scatter(
                    x=vessel_data['section_mid_time'],
                    y=trendline_ideal,
                    mode='lines',
                    name=f'{vessel_name} - Ideal Trend',
                    line=dict(color=colors['ideal_trend'], width=3),  # Modified
                    hovertemplate='Ideal Power Trend: %{y:.2f} kW<extra></extra>',
                    legendgroup=f'vessel_{vessel}',
                    showlegend=True,
                    opacity=0.8
                ))
        
        fig2.update_layout(
            title='Power Over Time - All Vessels',
            xaxis_title='Time',
            yaxis_title='Power (kW)',
            hovermode='closest',
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            height=700,
            margin=dict(r=200)
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Display summary of filtered data
        st.write(f"**Showing {len(filtered_df)} data points from {len(unique_vessels)} vessels after applying filters**")
        
    else:
        st.write("No data points match the current filter criteria. Please adjust the filters.")

else:
    st.write("No voyage data found for any vessels.")