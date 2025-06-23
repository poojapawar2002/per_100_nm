import pickle

with open('voyage_dict.pkl', 'rb') as f:
    voyage_dict = pickle.load(f)

import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

df = pd.read_csv("autolog_complete_input_ideal_power_foc_7000series_except1004.csv")

df["ideal_foc"] = df["ideal_foc_hr"]



df["StartDateUTC"] = pd.to_datetime(df["StartDateUTC"], format="%d-%m-%Y %H:%M")
df["EndDateUTC"] = pd.to_datetime(df["EndDateUTC"], format="%d-%m-%Y %H:%M")

df.sort_values(by=["IMO", "StartDateUTC"], inplace=True)

# Select IMO
imos = list(voyage_dict.keys())
selected_imo = st.selectbox("Select IMO", imos)

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
    voyage_df["MeanDraft"] = (voyage_df["DraftAftTele"] + voyage_df["DraftFwdTele"]) / 2
    
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
st.write(f"Voyage Summary for IMO: {selected_imo}")
st.dataframe(summary_df)

# Section for per 100 nm breakdown
# st.write("## FuelMassCons Analysis per 100 nm Sections")

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
            section_df["MeanDraft"] = (section_df["DraftAftTele"] + section_df["DraftFwdTele"]) / 2
            
            
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
                
                section_data = {f"Voyage {selected_voyage_num} - Section": f"{current_section} (Incomplete)"}
                section_data.update(metrics)
                
                sections_100_nm.append(section_data)
        
        # Create sections DataFrame
        if sections_100_nm:
            sections_df = pd.DataFrame(sections_100_nm)
            st.write(f"#### {selected_voyage} - 100 nautical miles sections")
            st.dataframe(sections_df)
            
            # Show summary statistics
            # complete_sections = [s for s in sections_100_nm if "Incomplete" not in str(s[f"Voyage {selected_voyage_num} - Section"])]
        #     if complete_sections:
        #         avg_FuelMassCons_per_100_nm = sum([s["MEFuelMassCons"] for s in complete_sections]) / len(complete_sections)
        #         st.write(f"**Average MEFuelMassCons per 100-knot section: {avg_FuelMassCons_per_100_nm:.2f}**")
        #         st.write(f"**Number of complete 100-knot sections: {len(complete_sections)}**")
        else:
            st.write("No complete 100-knot sections found for this voyage.")
else:
    st.write("No voyages with distance > 0 found for this IMO.")