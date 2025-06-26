import dash
from dash.dependencies import Input, Output
from dash import State, callback_context, dcc, html, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import base64

# to check if location is a country
import pycountry 
import json
import requests

# Dash Leaflet
import dash_leaflet as dl
import dash_leaflet.express as dlx

# vendor mapping
import os
import geopandas as gpd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from collections import defaultdict

# vendor-event mapping and pdf download
import io
import random
import math
from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

data_file_path = "Additional Files (Documentation, data, etc.)/processed_data_rescat.csv"

data_processed = pd.read_csv(data_file_path)

# Drop rows without geocoded coordinates
data_processed = data_processed.dropna(subset=['latitude', 'longitude'])

# Add code from topic_modeling.py to convert to sgt, drop rows with null lat or lon also.
# Convert timestamp to SG local time
def convert_to_sgt(timestamp):
    try:
        dt = pd.to_datetime(timestamp, errors='coerce')
        if pd.isna(dt):
            return None
        if dt.tz is None:
            if timestamp.strip().endswith('00:00:00'):
                sgt = pytz.timezone('Asia/Singapore')
                dt = sgt.localize(dt)
            else:
                dt = pytz.utc.localize(dt)
        sgt = pytz.timezone('Asia/Singapore')
        dt_sgt = dt.astimezone(sgt)
        return dt_sgt.strftime('%Y-%m-%d %H:%M:%S %Z')
    except Exception as e:
        return None

data_processed['published_date_sgt'] = data_processed['published_date'].apply(convert_to_sgt)

# Converting date column from string to proper date format
data_processed['published_date_sgt'] = pd.to_datetime(data_processed['published_date_sgt'])
data_processed = data_processed[['published_date', 'published_date_sgt'] + [col for col in data_processed.columns if col not in ['published_date', 'published_date_sgt']]]

# create dictionary of locations
locations_dict = data_processed[['extracted_location', 'latitude', 'longitude']]
# Nested dictionary with following structure (lat and lon values for each country)
# 
# {
#   'Lat': {'Country1': lat_value1, 'Country2': lat_value2, ...},
#   'Long': {'Country1': long_value1, 'Country2': long_value2, ...}
# }
# 
locations_dict_nested = locations_dict.set_index('extracted_location')[['latitude', 'longitude']].T.to_dict('dict')

# Extract events with negative sentiments (for testing purpose)
negative_events_df = data_processed[data_processed['sentiment'] == 'negative']
negative_count = len(negative_events_df)
total_count = len(data_processed)

# Get current date
sgt = pytz.timezone('Asia/Singapore')
current_sgt_date_str = datetime.now(sgt).strftime("%B %d %Y , %I:%M %p")

####################################################################################################################################
# VENDOR DATA PREP

# Load SAP country code to name mapping from CSV
sap_country_csv = "Additional Files (Documentation, data, etc.)/Country_Codes-export.csv"
sap_country_df = pd.read_csv(sap_country_csv)
sap_country_map = dict(zip(sap_country_df['Country Code'].str.strip(), sap_country_df['Country/Region'].str.strip()))

# Load vendor data
vendor_file = "Additional Files (Documentation, data, etc.)/acra_vendor_with_addresss.xlsx"
vendor_df = pd.read_excel(vendor_file)

# Filter and assign tiers based on Total Net Order Value
vendor_df = vendor_df[(vendor_df['uen_status_desc'] == 'Registered') &
                      (vendor_df['Total Net Order Value'].notna()) &
                      (vendor_df['Vendor Country'].notna())]

vendor_df = vendor_df.copy()
# Split data into 3 quantiles based on total net order value (to assign tier) => to be updated later.
vendor_df['tier'] = pd.qcut(vendor_df['Total Net Order Value'], q=3, labels=['bronze', 'silver', 'gold'])

# Ensure Vendor Country is str for mapping
vendor_df['Vendor Country'] = vendor_df['Vendor Country'].astype(str).str.strip()
vendor_df['Mapped Country'] = vendor_df['Vendor Country'].map(sap_country_map)

# Prepare geocoding query (concat country and HQ name)
vendor_df['geocode_query'] = vendor_df['Vendor Name'].fillna('') + ', ' + vendor_df['Mapped Country'].fillna('')

# Load cache file if exists
cache_file = "vendor_geocoded_cache.csv"
if os.path.exists(cache_file):
    cache_df = pd.read_csv(cache_file)
else:
    cache_df = pd.DataFrame(columns=['Vendor Code', 'lat', 'lon'])

# Merge cached coordinates
vendor_df = pd.merge(vendor_df, cache_df, on='Vendor Code', how='left')

# Geocoding with fallback to country name only
geolocator = Nominatim(user_agent="vendor-mapper")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2, error_wait_seconds=2)

def get_location_with_fallback(row):
    query = row['geocode_query']
    country_name = row['Mapped Country'].strip() if pd.notna(row['Mapped Country']) else None

    if pd.notna(row.get('lat')) and pd.notna(row.get('lon')):
        return pd.Series([row['lat'], row['lon']])  # already in cache

    if not query or not country_name:
        return pd.Series((None, None))

    try:
        location = geocode(query, timeout=10)
        if location:
            return pd.Series([location.latitude, location.longitude])
    except Exception as e:
        print(f"Geocoding failed for: {query} | Reason: {e}")

    try:
        location = geocode(country_name, timeout=10)
        if location:
            return pd.Series([location.latitude, location.longitude])
    except Exception as e:
        print(f"Fallback geocoding failed for: {country_name} | Reason: {e}")

    return pd.Series((None, None))

vendor_df[['lat', 'lon']] = vendor_df.apply(get_location_with_fallback, axis=1)
vendor_df = vendor_df.dropna(subset=['lat', 'lon'])

# Update and save cache
updated_cache = vendor_df[['Vendor Code', 'lat', 'lon']].drop_duplicates()
updated_cache.to_csv(cache_file, index=False)

##
# Assign tier priority per lat/lon (for highest tier coloring on map)
vendor_df['latlon'] = vendor_df[['lat', 'lon']].apply(lambda row: f"{row['lat']:.6f},{row['lon']:.6f}", axis=1)
tier_rank = {'gold': 3, 'silver': 2, 'bronze': 1}

# Group vendors by latlon, determine highest tier and names for each location
grouped_vendor_df = vendor_df.groupby('latlon').agg({
    'Vendor Name': list,
    'tier': lambda tiers: max(tiers, key=lambda t: tier_rank[t]),
    'lat': 'first',
    'lon': 'first'
}).reset_index()

grouped_vendor_df['hover_text'] = grouped_vendor_df['Vendor Name'].apply(lambda names: '<br>'.join(names))

tier_icon_map = {
    'gold': '/assets/gold-badge.png',
    'silver': '/assets/silver-badge.png',
    'bronze': '/assets/bronze-badge.png'
}

grouped_vendor_df['icon'] = grouped_vendor_df['tier'].map(tier_icon_map)
grouped_vendor_df['popup'] = grouped_vendor_df['Vendor Name'].apply(lambda x: '<br>'.join(x) if isinstance(x, list) else str(x))

####################################################################################################################################
# VENDOR-EVENT MAPPING FUNCTIONS

def create_vendor_event_mapping():
    """Create vendor-event mapping: Natural disasters with CONSISTENT radius"""
    
    # Country standardization mapping (keep your existing one)
    country_standardization = {
        'United States': ['USA', 'US', 'United States of America'],
        'China': ['CN', 'People\'s Republic of China', 'PRC'],
        'United Kingdom': ['UK', 'Britain', 'Great Britain', 'England', 'Scotland', 'Wales'],
        'Germany': ['DE', 'Deutschland'],
        'France': ['FR'],
        'Singapore': ['SG'],
        'Switzerland': ['CH'],
        'Austria': ['AT'],
        'Thailand': ['TH'],
        'Norway': ['NO'],
        'Slovakia': ['SK'],
        'Taiwan': ['TW', 'Republic of China'],
        'Brazil': ['BR'],
    }
    
    # Create reverse mapping for lookup
    country_lookup = {}
    for standard, variants in country_standardization.items():
        country_lookup[standard] = standard
        for variant in variants:
            country_lookup[variant] = standard
    
    # Get vendor and event data
    vendor_countries = vendor_df[['Vendor Code', 'Vendor Name', 'Mapped Country', 'lat', 'lon', 'tier', 'Total Net Order Value']].copy()
    vendor_countries['Standard_Country'] = vendor_countries['Mapped Country'].map(country_lookup)
    
    event_locations = negative_events_df[['url', 'title', 'event_topic', 'extracted_location', 'latitude', 'longitude', 'published_date_sgt']].copy()
    event_locations['Standard_Location'] = event_locations['extracted_location'].map(country_lookup)
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    
    # STEP 1: Natural disaster radius mapping with CONSISTENT radius
    natural_disaster_matches = []
    
    # Filter for natural disasters only
    disaster_events = event_locations[
        event_locations['event_topic'].str.lower().str.contains('natural disaster', na=False)
    ]
    
    disaster_mapped_pairs = set()  # Track what's been mapped by disasters
    
    for _, disaster in disaster_events.iterrows():
        if pd.notna(disaster['latitude']) and pd.notna(disaster['longitude']):
            # Generate CONSISTENT radius using disaster URL as seed
            disaster_radius = get_consistent_disaster_radius(disaster['url'])
            
            for _, vendor in vendor_countries.iterrows():
                if pd.notna(vendor['lat']) and pd.notna(vendor['lon']):
                    distance = haversine_distance(vendor['lat'], vendor['lon'], disaster['latitude'], disaster['longitude'])
                    
                    if distance <= disaster_radius:
                        pair_key = (vendor['Vendor Code'], disaster['url'])
                        disaster_mapped_pairs.add(pair_key)
                        
                        natural_disaster_matches.append({
                            'Vendor_Code': vendor['Vendor Code'],
                            'Vendor_Name': vendor['Vendor Name'],
                            'Vendor_Country': vendor['Mapped Country'],
                            'Vendor_Tier': vendor['tier'],
                            'Vendor_Order_Value': vendor['Total Net Order Value'],
                            'Vendor_Lat': vendor['lat'],
                            'Vendor_Lon': vendor['lon'],
                            'Event_URL': disaster['url'],
                            'Event_Title': disaster['title'],
                            'Event_Topic': disaster['event_topic'],
                            'Event_Location': disaster['extracted_location'],
                            'Event_Lat': disaster['latitude'],
                            'Event_Lon': disaster['longitude'],
                            'Event_Date': disaster['published_date_sgt'],
                            'Mapping_Type': f'Natural Disaster Radius ({disaster_radius}km)',
                            'Distance_KM': round(distance, 1)
                        })
    
    # STEP 2: Direct country matching for NON-DISASTER events (unchanged)
    direct_matches = []
    for _, vendor in vendor_countries.iterrows():
        vendor_country = vendor['Standard_Country'] or vendor['Mapped Country']
        
        matching_events = event_locations[
            ((event_locations['Standard_Location'] == vendor_country) |
            (event_locations['extracted_location'] == vendor_country) |
            (event_locations['extracted_location'] == vendor['Mapped Country'])) &
            (~event_locations['event_topic'].str.lower().str.contains('natural disaster', na=False))
        ]
        
        for _, event in matching_events.iterrows():
            pair_key = (vendor['Vendor Code'], event['url'])
            
            if pair_key in disaster_mapped_pairs:
                continue
                
            direct_matches.append({
                'Vendor_Code': vendor['Vendor Code'],
                'Vendor_Name': vendor['Vendor Name'],
                'Vendor_Country': vendor['Mapped Country'],
                'Vendor_Tier': vendor['tier'],
                'Vendor_Order_Value': vendor['Total Net Order Value'],
                'Vendor_Lat': vendor['lat'],
                'Vendor_Lon': vendor['lon'],
                'Event_URL': event['url'],
                'Event_Title': event['title'],
                'Event_Topic': event['event_topic'],
                'Event_Location': event['extracted_location'],
                'Event_Lat': event['latitude'],
                'Event_Lon': event['longitude'],
                'Event_Date': event['published_date_sgt'],
                'Mapping_Type': 'Direct Country Match',
                'Distance_KM': 0.0
            })
    
    # Combine and process mappings (unchanged)
    all_mappings = natural_disaster_matches + direct_matches
    mapping_df = pd.DataFrame(all_mappings)
    
    if not mapping_df.empty:
        mapping_df = mapping_df.drop_duplicates(subset=['Vendor_Code', 'Event_URL'])
        tier_order = {'gold': 1, 'silver': 2, 'bronze': 3}
        mapping_df['tier_rank'] = mapping_df['Vendor_Tier'].map(tier_order)
        mapping_df = mapping_df.sort_values(['tier_rank', 'Distance_KM', 'Vendor_Order_Value'], ascending=[True, True, False])
        mapping_df = mapping_df.drop('tier_rank', axis=1)
        mapping_df['Risk_Level'] = mapping_df.apply(assess_risk_level_enhanced, axis=1)
        
    return mapping_df

def assess_risk_level_enhanced(row):
    """Simple risk assessment: vendor tier + event recency"""
    base_risk_score = 2
    tier_multiplier = {'gold': 1.5, 'silver': 1.2, 'bronze': 1.0}
    tier_mult = tier_multiplier.get(row['Vendor_Tier'], 1.0)
    
    days_ago = (datetime.now(sgt) - row['Event_Date']).days
    if days_ago <= 7:
        recency_mult = 1.4
    elif days_ago <= 30:
        recency_mult = 1.2
    elif days_ago <= 90:
        recency_mult = 1.0
    else:
        recency_mult = 0.8
    
    final_score = base_risk_score * tier_mult * recency_mult
    
    if final_score >= 3.0:
        return 'High'
    elif final_score >= 2.0:
        return 'Medium'
    else:
        return 'Low'

def create_mapping_summary():
    """Create summary statistics"""
    mapping_df = create_vendor_event_mapping()
    
    if mapping_df.empty:
        return {
            'total_mappings': 0, 'unique_vendors': 0, 'unique_events': 0,
            'risk_breakdown': {}, 'tier_breakdown': {}, 'mapping_type_breakdown': {}
        }
    
    return {
        'total_mappings': len(mapping_df),
        'unique_vendors': mapping_df['Vendor_Code'].nunique(),
        'unique_events': mapping_df['Event_URL'].nunique(),
        'risk_breakdown': mapping_df['Risk_Level'].value_counts().to_dict(),
        'tier_breakdown': mapping_df['Vendor_Tier'].value_counts().to_dict(),
        'mapping_type_breakdown': mapping_df['Mapping_Type'].value_counts().to_dict(),
        'avg_distance': mapping_df['Distance_KM'].mean()
    }

def export_mapping_to_csv():
    """Export to CSV"""
    mapping_df = create_vendor_event_mapping()
    if mapping_df.empty:
        return None
    output = io.StringIO()
    mapping_df.to_csv(output, index=False)
    csv_content = output.getvalue()
    output.close()
    return csv_content

def export_mapping_to_pdf():
    """Export to PDF with enhanced formatting and colors"""
    mapping_df = create_vendor_event_mapping()
    if mapping_df.empty:
        return None
    
    buffer = io.BytesIO()
    # Use landscape orientation for more columns
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), 
                          leftMargin=0.5*inch, rightMargin=0.5*inch,
                          topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=1  # Center
    )
    story.append(Paragraph("Vendor-Event Risk Mapping Report", title_style))
    story.append(Spacer(1, 20))
    
    # Summary section
    summary = create_mapping_summary()
    summary_text = f"""
    <b>Executive Summary:</b><br/>
    • Total Mappings: {summary['total_mappings']}<br/>
    • Vendors Affected: {summary['unique_vendors']}<br/>
    • Risk Events: {summary['unique_events']}<br/>
    • Average Distance: {summary['avg_distance']:.1f}km<br/>
    """
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    cell_style = ParagraphStyle(
    name='TableCell',
    fontName='Helvetica',
    fontSize=8,
    leading=10,
    alignment=0,  # Left align
    wordWrap='CJK',  # Enables word wrapping
    )
    
    # Enhanced table data with more columns
    table_data = [['Vendor Name', 'Country', 'Tier', 'Event Title', 'Event Topic', 'Risk Level', 'Distance (km)', 'Mapping Type']]
    
    # for _, row in mapping_df.head(100).iterrows():  # Limit to 100 rows for PDF
    for _, row in mapping_df.iterrows():
        table_data.append([
            Paragraph(str(row['Vendor_Name']), cell_style),
            Paragraph(str(row['Vendor_Country']), cell_style),
            Paragraph(str(row['Vendor_Tier']).title(), cell_style),
            Paragraph(str(row['Event_Title']), cell_style),
            Paragraph(str(row['Event_Topic']), cell_style),
            Paragraph(str(row['Risk_Level']), cell_style),
            Paragraph(f"{row['Distance_KM']:.1f}", cell_style),
            Paragraph(str(row['Mapping_Type']), cell_style),
        ])
    
    # Create table with enhanced styling
    table = Table(
    table_data,
    colWidths=[1.6*inch, 1.0*inch, 0.7*inch, 2.1*inch, 1.3*inch, 0.8*inch, 0.9*inch, 1.2*inch]
    )   
    
    # Base table style
    table_style = [
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('FONTSIZE', (0, 1), (-1, -1), 8),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('ROWHEIGHT', (0, 0), (-1, -1), 0.35*inch),
    ]   

    # Add color coding for risk levels, CELL BY CELL!
    for i, row_data in enumerate(table_data[1:], 1):  # Skip header
        # risk_level = row_data[5]  # Risk Level column
        risk_level = row_data[5].getPlainText() if isinstance(row_data[5], Paragraph) else str(row_data[5])
        risk_color = get_risk_color_code(risk_level)
        # Convert hex to Color
        hex_color = risk_color.lstrip('#')
        rgb_color = colors.Color(
            int(hex_color[0:2], 16)/255.0,
            int(hex_color[2:4], 16)/255.0, 
            int(hex_color[4:6], 16)/255.0,
            alpha=0.3
        )

        table_style.append(('BACKGROUND', (0, i), (-1, i), rgb_color))
    
    table.setStyle(TableStyle(table_style))
    story.append(table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def get_risk_color(risk_level):
    """Get color for risk badges"""
    colors_map = {'High': 'warning', 'Medium': 'info', 'Low': 'success'}
    return colors_map.get(risk_level, 'secondary')

def get_tier_color(tier):
    """Get color for tier badges"""
    colors_map = {'Gold': 'warning', 'Silver': 'secondary', 'Bronze': 'dark'}
    return colors_map.get(tier, 'light')

def get_risk_color_code(risk_level):
    """Get color codes for risk levels"""
    color_map = {
        'High': '#FF0000',      # Red  
        'Medium': '#FFA500',    # Orange
        'Low': '#FFD700'        # Yellow/Gold
    }
    return color_map.get(risk_level, '#FFFFFF') #default color

def get_risk_badge_color(risk_level):
    """Get Bootstrap badge colors for risk levels"""
    color_map = {
        'High': 'danger', 
        'Medium': 'warning',
        'Low': 'success'
    }
    return color_map.get(risk_level, 'secondary')

def create_enhanced_button_layout():
    return html.Div([
        html.Div([
            # Top row - Main action
            html.Div([
                dbc.Button(
                    [html.I(className="fas fa-cogs me-2"), "Generate vendor-events info"],
                    id="generate-mapping-btn", 
                    color="dark",
                    size="lg",
                    style={'minWidth': '200px', 'fontWeight': 'bold'}
                ),
            ], style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            # Bottom row - Secondary actions  
            html.Div([
                dbc.ButtonGroup([
                    dbc.Button(
                        [html.I(className="fas fa-table me-2"), "View Table"], 
                        id="view-table-btn", 
                        color="info", 
                        outline=True,
                        style={'minWidth': '130px'}
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-download me-2"), "Export CSV"], 
                        id="export-csv-btn", 
                        color="success", 
                        outline=True,
                        style={'minWidth': '130px'}
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-file-pdf me-2"), "Export PDF"], 
                        id="export-pdf-btn", 
                        color="danger", 
                        outline=True,
                        style={'minWidth': '130px'}
                    ),
                ], size="sm")
            ], style={'textAlign': 'center'}),
            
            # Hidden download components
            dcc.Download(id="download-csv"),
            dcc.Download(id="download-pdf"),
            
        ], style={'backgroundColor': '#1f2c56', 'padding': '30px', 'borderRadius': '10px', 'margin': '10px'})
    ])

def create_enhanced_mapping_table(mapping_df):
    """Create enhanced table with color-coded risk levels and more columns"""
    if mapping_df.empty:
        return dbc.Alert("No mapping data available.", color="warning")
    
    # Create table with enhanced styling
    table_rows = []
    for _, row in mapping_df.iterrows():
        risk_color = get_risk_color_code(row['Risk_Level'])
        
        table_row = html.Tr([
            html.Td(str(row['Vendor_Name'])[:40], style={'fontWeight': 'bold'}),
            html.Td(str(row['Vendor_Country'])),
            html.Td(dbc.Badge(str(row['Vendor_Tier']).title(), color=get_tier_color(str(row['Vendor_Tier']).title()))),
            html.Td(str(row['Event_Title'])[:40] if pd.notna(row['Event_Title']) else "N/A"),
            html.Td(str(row['Event_Topic'])[:35] if pd.notna(row['Event_Topic']) else "N/A"),
            html.Td(dbc.Badge(row['Risk_Level'], color=get_risk_badge_color(row['Risk_Level']))),
            html.Td(f"{row['Distance_KM']:.1f}km"),
        ], style={'backgroundColor': risk_color + '20'})  # 20 = light transparency in hex
        
        table_rows.append(table_row)
    
    table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Vendor Name", style={'minWidth': '200px'}),
                html.Th("Country"),
                html.Th("Tier"),
                html.Th("Event Title", style={'minWidth': '200px'}),
                html.Th("Event Topic", style={'minWidth': '180px'}),
                html.Th("Risk Level"),
                html.Th("Distance"),
            ], style={'backgroundColor': '#343a40', 'color': 'white'})
        ]),
        html.Tbody(table_rows)
    ], striped=False, bordered=True, hover=True, responsive=True, size="sm")
    
    return html.Div([
        html.P(f"Showing all {len(mapping_df)} vendor-event mappings", 
               style={'fontWeight': 'bold', 'marginBottom': '15px'}),
        html.Div([
            html.P("Risk Level Legend:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            html.Div([
                dbc.Badge("High", color="danger", className="me-2"),
                dbc.Badge("Medium", color="warning", className="me-2"),
                dbc.Badge("Low", color="success", className="me-2"),
            ], style={'marginBottom': '15px'})
        ]),
        html.Div(table, style={'maxHeight': '500px', 'overflowY': 'auto'})
    ])
####################################################################################################################################
# VENDOR TOWER FUNCTIONS

def get_country_center_position(geojson_feature, offset_direction='northeast'):
    """
    Get country center position with a small offset for tower placement
    """
    try:
        # Handle different geometry types
        geometry = geojson_feature['geometry']
        coordinates = geometry['coordinates']
        
        # Flatten coordinates for MultiPolygon
        if geometry['type'] == 'MultiPolygon':
            # Take the largest polygon (first one usually)
            coordinates = coordinates[0][0]
        elif geometry['type'] == 'Polygon':
            coordinates = coordinates[0]
        else:
            return None, None
            
        # Extract all lat/lon points
        lons = [coord[0] for coord in coordinates]
        lats = [coord[1] for coord in coordinates]
        
        # Calculate center (centroid)
        center_lat = (min(lats) + max(lats)) / 2
        center_lon = (min(lons) + max(lons)) / 2
        
        # Apply small offset based on direction (about 1-2 degrees)
        offset_amount = 1.5  # degrees
        
        offset_directions = {
            'northeast': (offset_amount, offset_amount),     # +lat, +lon
            'northwest': (offset_amount, -offset_amount),    # +lat, -lon
            'southeast': (-offset_amount, offset_amount),    # -lat, +lon
            'southwest': (-offset_amount, -offset_amount),   # -lat, -lon
            'north': (offset_amount, 0),                     # +lat, 0
            'south': (-offset_amount, 0),                    # -lat, 0
            'east': (0, offset_amount),                      # 0, +lon
            'west': (0, -offset_amount),                     # 0, -lon
            'none': (0, 0)                                   # no offset (exact center)
        }
        
        lat_offset, lon_offset = offset_directions.get(offset_direction, (0, 0))
        
        final_lat = center_lat + lat_offset
        final_lon = center_lon + lon_offset
        
        print(f"  Center: ({center_lat:.3f}, {center_lon:.3f}) -> Offset ({final_lat:.3f}, {final_lon:.3f})")
        
        return final_lat, final_lon
        
    except Exception as e:
        print(f"Error getting center position: {e}")
        return None, None

def aggregate_vendors_by_country():
    """
    Aggregate vendor data by country and tier - with better country mapping
    """
    try:
        # EXPANDED country mapping to handle more cases
        country_mapping = {
            'United States': 'United States of America',
            'USA': 'United States of America',  # Fix for USA
            'US': 'United States of America',
            'Hong Kong': 'China',
            'South Korea': 'Republic of Korea',
            'North Korea': 'Democratic People\'s Republic of Korea',
            'Russia': 'Russian Federation',
            'Iran': 'Islamic Republic of Iran',
            'Venezuela': 'Bolivarian Republic of Venezuela',
            'Bolivia': 'Plurinational State of Bolivia',
            'Tanzania': 'United Republic of Tanzania',
            'Macedonia': 'North Macedonia',
            'Moldova': 'Republic of Moldova',
            'Congo': 'Republic of the Congo',
            'DR Congo': 'Democratic Republic of the Congo',
            'Ivory Coast': "Côte d'Ivoire",
            'Cape Verde': 'Cabo Verde',
            'Swaziland': 'Eswatini',
            'UK': 'United Kingdom',
            'Britain': 'United Kingdom',
            'England': 'United Kingdom',
            'Scotland': 'United Kingdom',
            'Wales': 'United Kingdom',
            'Northern Ireland': 'United Kingdom'
        }
        
        # Apply country mapping
        vendor_df_mapped = vendor_df.copy()
        vendor_df_mapped['GeoJSON_Country'] = vendor_df_mapped['Mapped Country'].replace(country_mapping)
        
        # Convert tier column to string to avoid categorical issues
        vendor_df_mapped['tier'] = vendor_df_mapped['tier'].astype(str)
        
        # Debug: Print unique countries before aggregation
        print("=== VENDOR COUNTRY MAPPING DEBUG ===")
        print("Original vendor countries:")
        for country in sorted(vendor_df_mapped['Mapped Country'].unique()):
            mapped = vendor_df_mapped[vendor_df_mapped['Mapped Country'] == country]['GeoJSON_Country'].iloc[0]
            count = len(vendor_df_mapped[vendor_df_mapped['Mapped Country'] == country])
            print(f"  {country} -> {mapped} ({count} vendors)")
        
        # Aggregate by country and tier
        country_vendor_summary = (
            vendor_df_mapped.groupby(['GeoJSON_Country', 'tier'])
            .agg({
                'Vendor Name': list,
                'Vendor Code': 'count'
            })
            .rename(columns={'Vendor Code': 'count'})
            .reset_index()
        )
        
        # Pivot to get counts per tier
        country_summary = country_vendor_summary.pivot(
            index='GeoJSON_Country', 
            columns='tier', 
            values='count'
        ).fillna(0).astype(int)
        
        # Reset index to make GeoJSON_Country a regular column
        country_summary = country_summary.reset_index()
        
        # Also get all vendor names per country
        country_vendors = (
            vendor_df_mapped.groupby('GeoJSON_Country')['Vendor Name']
            .apply(list)
            .reset_index()
        )
        
        # Merge the data
        try:
            final_summary = country_summary.merge(country_vendors, on='GeoJSON_Country', how='left')
        except Exception as merge_error:
            print(f"Merge failed, using alternative approach: {merge_error}")
            country_summary.set_index('GeoJSON_Country', inplace=True)
            country_vendors.set_index('GeoJSON_Country', inplace=True)
            final_summary = country_summary.join(country_vendors, how='left').reset_index()
        
        print("=== FINAL VENDOR SUMMARY ===")
        for _, row in final_summary.iterrows():
            total = int(row.get('gold', 0)) + int(row.get('silver', 0)) + int(row.get('bronze', 0))
            print(f"{row['GeoJSON_Country']}: {total} vendors (G:{row.get('gold', 0)}, S:{row.get('silver', 0)}, B:{row.get('bronze', 0)})")
        
        return final_summary
        
    except Exception as e:
        print(f"Error in aggregate_vendors_by_country: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['GeoJSON_Country', 'bronze', 'silver', 'gold', 'Vendor Name'])

def create_vendor_tower_svg(vendor_list, country_code):
    """
    Generate SVG for vendor tower with INDIVIDUAL vendor segments stacked
    Gold on top, then silver, then bronze. 
    For >100 vendors: shows 3 clean blocks (one per tier) instead of individual segments.
    """
    # Sort vendors: gold first (top), then silver, then bronze
    tier_order = {'gold': 1, 'silver': 2, 'bronze': 3}
    sorted_vendors = sorted(vendor_list, key=lambda x: tier_order.get(x, 3))
    
    # Count each tier
    total_vendors = len(sorted_vendors)
    gold_count = sorted_vendors.count('gold')
    silver_count = sorted_vendors.count('silver') 
    bronze_count = sorted_vendors.count('bronze')
    
    # SCALING for large vendor counts
    if total_vendors > 100:
        # For large countries: show 3 clean blocks (one per tier)
        display_segments = []
        if gold_count > 0:
            display_segments.append(('gold', gold_count))
        if silver_count > 0:
            display_segments.append(('silver', silver_count))
        if bronze_count > 0:
            display_segments.append(('bronze', bronze_count))
        
        scale_text = f"({total_vendors} vendors: G{gold_count} S{silver_count} B{bronze_count})"
        use_blocks = True
    else:
        # For small countries: show individual segments
        display_segments = [(tier, 1) for tier in sorted_vendors]
        scale_text = f"({total_vendors} vendors)"
        use_blocks = False
    
    # FIXED dimensions
    if use_blocks:
        # Larger blocks for consolidated view
        base_height = 15  # Base height for each tier block
        segment_spacing = 2  # Space between tier blocks
    else:
        # Thin segments for individual vendors
        base_height = 3  # Thin height per individual vendor
        segment_spacing = 0.5  # Small space between individual segments
    
    tower_width = 10  # Fixed width
    
    # Calculate total height
    if use_blocks:
        # Variable height per block based on vendor count (with min/max)
        total_height = 0
        for tier, count in display_segments:
            block_height = max(10, min(40, base_height + (count // 20)))  # Scale block size slightly
            total_height += block_height
        total_height += (len(display_segments) - 1) * segment_spacing if len(display_segments) > 1 else 0
    else:
        # Fixed height per individual vendor
        total_height = len(display_segments) * base_height + (len(display_segments) - 1) * segment_spacing
    
    # SVG dimensions
    svg_height = total_height + 35  # Extra space for labels
    svg_width = tower_width + 10
    
    # Tier colors
    tier_colors = {
        'gold': f'url(#goldGrad_{country_code})',
        'silver': f'url(#silverGrad_{country_code})',
        'bronze': f'url(#bronzeGrad_{country_code})'
    }
    
    tier_strokes = {
        'gold': '#CC8400',
        'silver': '#606060', 
        'bronze': '#654321'
    }
    
    svg = f'''
    <svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="goldGrad_{country_code}" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#FFD700;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#FFA500;stop-opacity:1" />
            </linearGradient>
            <linearGradient id="silverGrad_{country_code}" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#C0C0C0;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#808080;stop-opacity:1" />
            </linearGradient>
            <linearGradient id="bronzeGrad_{country_code}" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#CD853F;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#8B4513;stop-opacity:1" />
            </linearGradient>
        </defs>
    '''
    
    # Draw segments from bottom to top (reverse order so gold ends up on top)
    current_y = svg_height - 30  # Start from bottom, leave space for labels
    
    for i, (tier, count) in enumerate(reversed(display_segments)):
        if use_blocks:
            # Variable height blocks with count labels
            block_height = max(10, min(40, base_height + (count // 20)))
            current_y -= block_height
            
            fill_color = tier_colors.get(tier, tier_colors['bronze'])
            stroke_color = tier_strokes.get(tier, tier_strokes['bronze'])
            
            svg += f'''
            <rect x="5" y="{current_y}" width="{tower_width}" height="{block_height}" 
                  fill="{fill_color}" stroke="{stroke_color}" stroke-width="0.5" rx="1"/>
            '''
        else:
            # Individual thin segments
            current_y -= base_height
            
            fill_color = tier_colors.get(tier, tier_colors['bronze'])
            stroke_color = tier_strokes.get(tier, tier_strokes['bronze'])
            
            svg += f'''
            <rect x="5" y="{current_y}" width="{tower_width}" height="{base_height}" 
                  fill="{fill_color}" stroke="{stroke_color}" stroke-width="0.3" rx="0.5"/>
            '''
        
        # Add spacing between segments (except for the last one)
        if i < len(display_segments) - 1:
            current_y -= segment_spacing
    
    # Country label and info at bottom
    svg += f'''
        <text x="{svg_width//2}" y="{svg_height - 15}" 
              text-anchor="middle" font-family="Arial, sans-serif" 
              font-size="9" font-weight="bold" fill="#333">{country_code}</text>
        <text x="{svg_width//2}" y="{svg_height - 5}" 
              text-anchor="middle" font-family="Arial, sans-serif" 
              font-size="6" fill="#666">{scale_text}</text>
    </svg>
    '''
    
    return svg

#######################################################################################################
# Create legend functions
def create_map_legends():
    """Create legend components for event icons and vendor tiers"""
    
    # Event icons legend
    event_legend_items = [
        {'icon': '/assets/disease_outbreak.png', 'label': 'Disease Outbreak'},
        {'icon': '/assets/war.png', 'label': 'War'},
        {'icon': '/assets/military_conflict.png', 'label': 'Military Conflict'},
        {'icon': '/assets/natural_disaster.png', 'label': 'Natural Disaster'},
        {'icon': '/assets/banking_financial_crisis.png', 'label': 'Banking/Financial Crisis'},
        {'icon': '/assets/cybersecurity_breach.png', 'label': 'Cybersecurity Breach'},
        {'icon': '/assets/strike.png', 'label': 'Strike'},
        {'icon': '/assets/supply_chain_disruption.png', 'label': 'Supply Chain Disruption'},
        {'icon': '/assets/technology_failure.png', 'label': 'Technology Failure'},
        {'icon': '/assets/trade_policy_change.png', 'label': 'Policy Change'},
    ]
    
    event_legend = html.Div([
        html.H6("Event Types", style={'color': 'white', 'marginBottom': '10px', 'fontWeight': 'bold'}),
        html.Div([
            html.Div([
                html.Img(src=item['icon'], style={'width': '20px', 'height': '20px', 'marginRight': '8px', 'verticalAlign': 'middle'}),
                html.Span(item['label'], style={'color': 'white', 'fontSize': '12px', 'verticalAlign': 'middle'})
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'})
            for item in event_legend_items
        ])
    ], style={
        'backgroundColor': 'rgba(31, 44, 86, 0.9)',
        'padding': '15px',
        'borderRadius': '8px',
        'marginBottom': '10px',
        'maxHeight': '300px',
        'overflowY': 'auto'
    })
    
    # Vendor tiers legend
    vendor_legend = html.Div([
        html.H6("Vendor Tiers", style={'color': 'white', 'marginBottom': '10px', 'fontWeight': 'bold'}),
        html.Div([
            # Gold tier
            html.Div([
                html.Img(src='/assets/gold-badge.png', style={'width': '20px', 'height': '20px', 'marginRight': '8px'}),
                html.Span("Gold Tier", style={'color': 'white', 'fontSize': '12px'})
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'}),
            
            # Silver tier
            html.Div([
                html.Img(src='/assets/silver-badge.png', style={'width': '20px', 'height': '20px', 'marginRight': '8px'}),
                html.Span("Silver Tier", style={'color': 'white', 'fontSize': '12px'})
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'}),
            
            # Bronze tier
            html.Div([
                html.Img(src='/assets/bronze-badge.png', style={'width': '20px', 'height': '20px', 'marginRight': '8px'}),
                html.Span("Bronze Tier", style={'color': 'white', 'fontSize': '12px'})
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'})
        ])
    ], style={
        'backgroundColor': 'rgba(31, 44, 86, 0.9)',
        'padding': '15px',
        'borderRadius': '8px'
    })
    
    return event_legend, vendor_legend  
#######################################################################################################################################################
# zoom-based functions

# Load geocode cache at the top of your file (after other imports)
def load_geocode_cache():
    """Load the geocode cache from JSON file"""
    try:
        cache_file_path = "geocode_cache.json"
        print(f"Attempting to load cache from: {cache_file_path}")
        
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        print(f"Successfully loaded cache with {len(cache_data)} entries")

        return cache_data
        
    except FileNotFoundError:
        print(f"ERROR: geocode_cache.json not found at {cache_file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in geocode_cache.json: {e}")
        return {}
    except Exception as e:
        print(f"ERROR: Failed to load geocode cache: {e}")
        return {}

# Load the cache once at startup
GEOCODE_CACHE = load_geocode_cache()

def safe_geocode_lookup(country_name, geocode_cache):
    """Enhanced lookup that handles name variations"""
    
    # Direct lookup first
    if country_name in geocode_cache:
        return geocode_cache[country_name]
    
    # Handle variations
    variations = {
        'United States of America': 'United States',
        'USA': 'United States', 
        'US': 'United States',
        'UK': 'United Kingdom'
    }
    
    if country_name in variations:
        mapped_name = variations[country_name]
        if mapped_name in geocode_cache:
            return geocode_cache[mapped_name]
    
    # Case insensitive lookup
    for cache_key, cache_value in geocode_cache.items():
        if cache_key.lower() == country_name.lower():
            return cache_value
    
    return None

# Event icon mapping for filtering 
EVENT_ICON_MAP = {
    'disease outbreak': '/assets/disease_outbreak.png',
    'war': '/assets/war.png',
    'military conflict': '/assets/military_conflict.png',
    'natural disaster': '/assets/natural_disaster.png',
    'banking or financial crisis': '/assets/banking_financial_crisis.png',
    'cybersecurity breach': '/assets/cybersecurity_breach.png',
    'strike': '/assets/strike.png',
    'supply chain disruption': '/assets/supply_chain_disruption.png',
    'technology failure': '/assets/technology_failure.png',
    'policy change': '/assets/trade_policy_change.png',
}

def get_unique_event_topics(events):
    """Group events by unique topic and count them"""
    topic_counts = {}
    
    for event in events:
        topic = event['event_topic']
        if topic in topic_counts:
            topic_counts[topic] += 1
        else:
            topic_counts[topic] = 1
    
    # Sort by count (highest first)
    return dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True))

def filter_events_with_icons(df, event_icon_map):
    """Filter dataframe to only include events that have matching icons"""
    def has_matching_icon(topic):
        topic_lower = str(topic).lower()
        return any(keyword in topic_lower for keyword in event_icon_map.keys())
    
    return df[df['event_topic'].apply(has_matching_icon)]

# Zoom-based display functions
def create_country_statistics(choropleth_df):
    """Create country-level event statistics for zoomed out view - only for events with icons"""
    # Filter to only events that have matching icons
    events_with_icons = filter_events_with_icons(choropleth_df, EVENT_ICON_MAP)
    
    # Group events by country and event type
    country_stats = events_with_icons.groupby(['mapped_location', 'event_topic']).size().reset_index(name='count')

    # Converts GeoJSON names back to cache names
    geojson_to_cache = {
        'United States of America': 'United States',
        'China': 'China'
    }

    country_summary = {}
    for _, row in country_stats.iterrows():
        geojson_country = row['mapped_location']  # This is "United States of America"
        
        # Convert to cache-compatible name for lookup
        cache_country = geojson_to_cache.get(geojson_country, geojson_country)
        
        event_type = row['event_topic']
        count = row['count']
        
        if cache_country not in country_summary:
            country_summary[cache_country] = {}
        country_summary[cache_country][event_type] = count
    
    return country_summary

def create_event_type_clusters(choropleth_df, zoom_level):
    """Create event type clusters for medium zoom levels - separate clusters by event type"""
    if zoom_level <= 2.1 or zoom_level > 5:
        return []
    
    # Filter to only events that have matching icons
    events_with_icons = filter_events_with_icons(choropleth_df, EVENT_ICON_MAP)
    
    # Group events by location first, then by event type - LARGER radius for more spread
    cluster_radius = 3.0  
    location_groups = defaultdict(lambda: defaultdict(list))
    
    for _, event in events_with_icons.iterrows():
        lat, lon = event['latitude'], event['longitude']
        if pd.notna(lat) and pd.notna(lon):
            # Round coordinates to create location clusters
            cluster_key = (round(lat/cluster_radius)*cluster_radius, round(lon/cluster_radius)*cluster_radius)
            event_type = event['event_topic'].lower()
            location_groups[cluster_key][event_type].append(event)
    
    # ENHANCED color generation with more distinct colors
    def get_event_type_color(event_type):
        """Generate consistent color for any event type using hash with more distinct colors"""
        import hashlib
        
        # EXPANDED color palette with more distinct, vibrant colors
        color_palette = [
            '#FF0000',  # Bright Red
            '#0066FF',  # Bright Blue  
            '#FF9900',  # Bright Orange
            '#00CC00',  # Bright Green
            '#9900FF',  # Bright Purple
            '#FF0099',  # Bright Pink
            '#00CCFF',  # Bright Cyan
            '#FFCC00',  # Bright Yellow
            '#FF3366',  # Bright Rose
            '#6600CC',  # Bright Violet
            '#00FF99',  # Bright Mint
            '#FF6600',  # Bright Orange Red
            '#3366FF',  # Bright Royal Blue
            '#CC0099',  # Bright Magenta
            '#99FF00',  # Bright Lime
            '#FF3300',  # Bright Scarlet
            '#0099FF',  # Bright Sky Blue
            '#FF9933',  # Bright Peach
            '#6699FF',  # Bright Periwinkle
            '#FF6699',  # Bright Salmon
        ]
        
        # Use hash of event type to consistently pick the same color
        hash_value = int(hashlib.md5(event_type.encode()).hexdigest(), 16)
        color_index = hash_value % len(color_palette)
        return color_palette[color_index]
    
    # Create separate markers for each event type at each location
    type_cluster_markers = []
    
    for (cluster_lat, cluster_lon), event_types in location_groups.items():
        # MUCH BETTER offset positioning for multiple event types
        num_types = len(event_types)
        
        if num_types == 1:
            # Single type - no offset needed
            offset_positions = [(cluster_lat, cluster_lon)]
        elif num_types == 2:
            # Two types - spread much wider horizontally
            offset_positions = [
                (cluster_lat, cluster_lon - 1.5),
                (cluster_lat, cluster_lon + 1.5)
            ]
        elif num_types == 3:
            # Three types - larger triangle formation
            offset_positions = [
                (cluster_lat + 1.2, cluster_lon),          # Top
                (cluster_lat - 0.6, cluster_lon - 1.2),   # Bottom left
                (cluster_lat - 0.6, cluster_lon + 1.2)    # Bottom right
            ]
        elif num_types == 4:
            # Four types - larger square formation
            offset_positions = [
                (cluster_lat + 1.0, cluster_lon - 1.0),   # Top left
                (cluster_lat + 1.0, cluster_lon + 1.0),   # Top right
                (cluster_lat - 1.0, cluster_lon - 1.0),   # Bottom left
                (cluster_lat - 1.0, cluster_lon + 1.0)    # Bottom right
            ]
        else:
            # 5+ types - larger circular formation with much more separation
            offset_radius = 2.0  # Much larger radius for maximum separation
            offset_positions = []
            for i in range(num_types):
                angle = (360 / num_types) * i
                offset_lat = cluster_lat + offset_radius * np.cos(np.radians(angle))
                offset_lon = cluster_lon + offset_radius * np.sin(np.radians(angle))
                offset_positions.append((offset_lat, offset_lon))
        
        for i, (event_type, events) in enumerate(event_types.items()):
            count = len(events)
                
            color = get_event_type_color(event_type)  # Auto-generate unique color
            
            # Use pre-calculated offset position
            offset_lat, offset_lon = offset_positions[i]
            
            # Add BIGGER count badge in center of circle
            type_cluster_markers.append(
                dl.Marker(
                    position=(offset_lat, offset_lon),  # Center position
                    icon=dict(
                        iconUrl=f"data:image/svg+xml;base64,{create_count_badge_svg(count, color)}",
                        iconSize=[28, 28],  # Much bigger badge
                        iconAnchor=[14, 14]
                    ),
                    zIndexOffset=1001,
                    children=[
                        dl.Tooltip(
                            html.Div([
                                html.Span(f"📍 {event_type.title()}", 
                                         style={'fontWeight': 'bold', 'fontSize': '14px', 'color': 'black'}),
                                html.Br(),
                                html.Span(f"Count: {count} events", 
                                         style={'fontSize': '12px', 'color': 'black'}),
                            ])
                        ),
                    ]
                )
            )
    
    return type_cluster_markers

def create_count_badge_svg(count, color):
    """Create SVG for count badge with colored background"""
    # Determine text color based on background color brightness
    def get_text_color(hex_color):
        # Convert hex to RGB
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        # Calculate brightness (0-255)
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        return 'white' if brightness < 128 else 'black'
    
    text_color = get_text_color(color)
    
    svg = f'''
    <svg width="28" height="28" xmlns="http://www.w3.org/2000/svg">
        <circle cx="14" cy="14" r="12" fill="{color}" stroke="white" stroke-width="3"/>
        <text x="14" y="18" text-anchor="middle" font-family="Arial, sans-serif" 
              font-size="12" font-weight="bold" fill="{text_color}">{count}</text>
    </svg>
    '''
    return base64.b64encode(svg.encode('utf-8')).decode('utf-8')

def create_cluster_count_svg(count):
    """Create SVG for cluster count display"""
    svg = f'''
    <svg width="30" height="30" xmlns="http://www.w3.org/2000/svg">
        <text x="15" y="18" text-anchor="middle" font-family="Arial, sans-serif" 
              font-size="12" font-weight="bold" fill="white">{count}</text>
    </svg>
    '''
    return base64.b64encode(svg.encode('utf-8')).decode('utf-8')

def create_country_stat_markers(country_summary):
    """Create country statistics markers for zoomed out view using geocode cache"""
    stat_markers = []
    
    # Now create markers for countries we can find
    for country, stats in country_summary.items():
        cache_entry = safe_geocode_lookup(country, GEOCODE_CACHE)
        
        if cache_entry and cache_entry.get('lat') and cache_entry.get('lon'):
            lat, lon = cache_entry['lat'], cache_entry['lon']
            
            total_events = sum(stats.values())
            dynamic_width = max(80, min(200, len(country) * 6 + 20))

            # Create country statistics marker
            stat_markers.append(
                dl.Marker(
                    position=(lat, lon),
                    icon=dict(
                        iconUrl=f"data:image/svg+xml;base64,{create_country_stat_svg(country, total_events)}",
                        iconSize=[dynamic_width, 60],
                        iconAnchor=[dynamic_width//2, 30]
                    ),
                    zIndexOffset=500
                )
            )
            print(f"Created stat marker for {country} at ({lat}, {lon})")
        else:
            print(f"Warning: No coordinates found for {country} in geocode cache")
    
    print(f"Total stat markers created: {len(stat_markers)}")
    return stat_markers

def create_country_stat_svg(country, total_events):
    """Create SVG for country statistics display"""
    # Truncate country name if too long
    display_name = country
    
    # Calculate dynamic width based on country name length
    char_width = 6  # pixels per character
    box_width = max(80, min(200, len(country) * char_width + 20))
    
    svg = f'''
    <svg width="{box_width}" height="60" xmlns="http://www.w3.org/2000/svg">
        <rect x="2" y="2" width="{box_width-4}" height="40" fill="rgba(0,0,0,0.8)" 
              stroke="#FF0000" stroke-width="2" rx="4"/>
        <text x="{box_width//2}" y="18" text-anchor="middle" font-family="Arial, sans-serif" 
              font-size="11" font-weight="bold" fill="#ffd700">{country}</text>
        <text x="{box_width//2}" y="35" text-anchor="middle" font-family="Arial, sans-serif" 
              font-size="12" font-weight="bold" fill="white">{total_events} events</text>
    </svg>
    '''
    
    return base64.b64encode(svg.encode('utf-8')).decode('utf-8')

def create_marker_cluster_for_medium_zoom(choropleth_df):
    """Custom clustering using LayerGroup since MarkerClusterGroup is not available"""
    
    # Filter to only events that have matching icons
    events_with_icons = filter_events_with_icons(choropleth_df, EVENT_ICON_MAP)
    
    # Group events by approximate location (0.8 degree grid for medium clustering)
    location_groups = defaultdict(list)
    for _, event in events_with_icons.iterrows():
        lat, lon = event['latitude'], event['longitude']
        if pd.notna(lat) and pd.notna(lon):
            # Round to nearest 0.8 degrees for clustering
            grid_lat = round(lat / 0.8) * 0.8
            grid_lon = round(lon / 0.8) * 0.8
            location_groups[(grid_lat, grid_lon)].append(event)
    
    markers = []
    for (grid_lat, grid_lon), events in location_groups.items():
        # Multiple events - create cluster marker
        count = len(events)
        # Use average position of events in cluster
        avg_lat = sum(e['latitude'] for e in events) / count
        avg_lon = sum(e['longitude'] for e in events) / count
        
        markers.append(
            dl.Marker(
                position=[avg_lat, avg_lon],
                icon=dict(
                    iconUrl=f"data:image/svg+xml;base64,{create_simple_cluster_svg(count)}",
                    iconSize=[50, 50],
                    iconAnchor=[25, 25]
                ),
                children=[
                    dl.Tooltip(f"{count} events here"),
                    dl.Popup([
                        html.H6(f"📍 {events[0]['extracted_location']}", style={'color': '#888'}),
                        html.Hr(style={'margin': '10px 0'}),
                        *[html.Div([
                            html.Strong(f"{topic} ({topic_count})", style={'color': '#666'}),
                            html.Br(),
                        ]) for topic, topic_count in get_unique_event_topics(events).items()],
                        html.P(f"Total: {count} events", 
                            style={'fontStyle': 'italic', 'color': '#666', 'marginTop': '10px'})
                    ])
                ]
            )
        )
    
    # Return as LayerGroup
    if markers:
        return [dl.LayerGroup(children=markers, id='event-cluster-group')]
    
    return []

def create_simple_cluster_svg(count):
    """Create cluster icon SVG"""
    size = 50
    font_size = 16
    
    # Color based on count
    if count <= 5:
        color = '#1f77b4'  # Blue
    elif count <= 15:
        color = '#ff7f0e'  # Orange
    else:
        color = '#d62728'  # Red
    
    svg = f'''
    <svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <radialGradient id="grad_{count}" cx="30%" cy="30%" r="70%">
                <stop offset="0%" style="stop-color:{color};stop-opacity:1" />
                <stop offset="100%" style="stop-color:{color};stop-opacity:0.8" />
            </radialGradient>
        </defs>
        <circle cx="{size//2}" cy="{size//2}" r="{size//2-3}" fill="url(#grad_{count})" stroke="white" stroke-width="3"/>
        <text x="{size//2}" y="{size//2+font_size//3}" text-anchor="middle" font-family="Arial" 
              font-size="{font_size}" font-weight="bold" fill="white">{count}</text>
    </svg>
    '''
    return base64.b64encode(svg.encode('utf-8')).decode('utf-8')
#######################################################################################################################################################
# Natural disasters radius mapping functions

# generate consistent radius per disaster (TODO: change with real data when possible)
def get_consistent_disaster_radius(disaster_url):
    """Generate consistent radius per disaster event using URL as seed"""
    import hashlib
    # Use disaster URL as seed for consistent radius generation
    seed = int(hashlib.md5(disaster_url.encode()).hexdigest(), 16) % 1000000
    random.seed(seed)
    radius = random.randint(100, 500)
    random.seed()  # Reset random seed
    return radius

def create_disaster_radius_circles(filtered_events):
    """Create radius circles using the SAME radius as vendor mapping"""
    
    # Filter for natural disasters only
    disaster_events = filtered_events[
        filtered_events['event_topic'].str.lower().str.contains('natural disaster', na=False)
    ]
    
    radius_circles = []
    
    # Use the SAME radius generation function as backend
    for index, disaster in disaster_events.iterrows():
        if pd.notna(disaster['latitude']) and pd.notna(disaster['longitude']):
            
            # Use SAME radius generation as backend mapping
            disaster_radius_km = get_consistent_disaster_radius(disaster['url'])
            
            # Convert km to degrees for map display (approximate: 1 degree ≈ 111 km)
            radius_degrees = disaster_radius_km / 111.0
            
            # Create circle overlay for this disaster
            circle = dl.Circle(
                center=[disaster['latitude'], disaster['longitude']],
                radius=radius_degrees * 111000,  # Convert back to meters for leaflet
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.3, # Semi-transparent
                weight=2,
                opacity=0.7,
                children=[
                    dl.Tooltip(f"Natural Disaster - Impact Radius: {disaster_radius_km}km"),
                    dl.Popup([
                        html.H6("🌋 Natural Disaster", style={'color': '#d63031', 'margin': '0 0 10px 0'}),
                        html.P([
                            html.Strong("Location: "), 
                            disaster['extracted_location']
                        ], style={'color': '#2d3436', 'margin': '5px 0'}),
                        html.P([
                            html.Strong("Impact Radius: "), 
                            f"{disaster_radius_km} km"
                        ], style={'color': '#d63031', 'fontWeight': 'bold', 'margin': '5px 0'}),
                        html.P([
                            html.Strong("Date: "), 
                            disaster['published_date_sgt'].strftime('%Y-%m-%d') if pd.notna(disaster['published_date_sgt']) else 'Unknown'
                        ], style={'color': '#636e72', 'margin': '5px 0'}),
                        html.Hr(style={'margin': '10px 0'}),
                        html.A("📰 View Article", 
                              href=disaster['url'], 
                              target="_blank",
                              style={'color': '#0984e3', 'textDecoration': 'none'})
                    ])
                ]
            )
            radius_circles.append(circle)
    
    return radius_circles

#######################################################################################################################################################

# Configure dash app, initialize with Bootstrap theme
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}], external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([

    # Last Updated Text
    html.Div([
        html.H6('Last Updated: ' + current_sgt_date_str + ' (SGT)', id='last-updated-text'),
    ]),

    # Radio Buttons Container with Disaster Radius Toggle
    html.Div([
        html.Label("Map View:", style={'color': 'white', 'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
        dcc.RadioItems(
            id='view-toggle',
            options=[
                {'label': 'Marker View', 'value': 'marker'},
                {'label': 'Choropleth View', 'value': 'choropleth'}
            ],
            value='marker',
            labelStyle={'display': 'inline-block', 'marginRight': '20px', 'color': 'white'},
            style={'marginBottom': '15px'}
        ),
        
        # Disaster radius toggle (only shows in choropleth view)
        html.Div([
            html.Label("Disaster Impact Analysis:", style={'color': 'white', 'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
            dcc.Checklist(
                id='show-disaster-radius',
                options=[
                    {'label': ' Show Natural Disaster Impact Zones', 'value': 'show_radius'}
                ],
                value=[],  # Unchecked by default
                labelStyle={'color': 'white', 'marginLeft': '10px'},
                inputStyle={'marginRight': '8px'}
            ),
            html.Small([
                "💡 ", 
                html.Strong("Tip: "),
                "Toggle to see estimated impact radius for natural disasters (100-500km)"
            ], style={'color': '#aaa', 'fontStyle': 'italic', 'display': 'block', 'marginTop': '5px'})
        ], id='radius-controls', style={'display': 'none'})  # Hidden by default
        
    ], style={
        'backgroundColor': '#1f2c56',
        'borderRadius': '5px',
        'padding': '15px',
        'margin': '10px 10px 0 10px',
        'zIndex': 10,
        'position': 'relative'
    }),

    # Unified Card - Map and Vendor-Event Mapping
    html.Div([
        # Map Container
        html.Div([
            dl.Map(
                id='leaflet-map',
                center=[30, 35],
                zoom=2,
                children=[],
                style={'width': '100%', 'height': '70vh'}
            ),
            # Legends Container - positioned absolutely within map
            html.Div([
                html.Div(id='map-legends-container', children=[])
            ], style={
                'position': 'absolute',
                'top': '10px',
                'right': '10px',
                'zIndex': 1000,
                'maxWidth': '250px'
            })
        ], style={
            'position': 'relative',
            'marginBottom': '20px'  # Add space between map and buttons
        }),
        
        # Vendor-Event Mapping Section
        create_enhanced_button_layout(),
        html.Div(id='mapping-summary', style={'marginBottom': '20px'}),
        
    ], className="create_container1 twelve columns", style={
        'margin': '10px',
        'position': 'relative',
        'height': 'auto'  # Change from fixed height to auto to accommodate content
    }),

    # Modal
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(id='modal-title')),
            dbc.ModalBody(id='modal-body'),
        ],
        id="event-modal",
        is_open=False,
        size="lg",
        scrollable=True,
    ),

    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Vendor-Event Mapping Details")),
            dbc.ModalBody(id='mapping-modal-body'),
        ],
        id="mapping-modal",
        is_open=False,
        size="xl",
        scrollable=True,
    ),

], id="mainContainer",
    style={"display": "flex", "flex-direction": "column"})

# Callback Functions

# Create scattermapbox chart: Default zoom for world but zoom in if a specific location selected, show scatter points sized and colored by event counts, added - toggle between existing map and choropleth based on inputs of radio buttons
@app.callback(
    Output('leaflet-map', 'children'),
    Input('view-toggle', 'value'),
    Input('leaflet-map', 'zoom'),        # to get current zoom level
    Input('show-disaster-radius', 'value')
)
def update_leaflet_map(selected_view, current_zoom, show_radius_list):
    # Get what triggered the callback
    ctx = dash.callback_context
    triggered_input = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Default view settings(world)
    map_center = [20, 0]
    map_zoom = 2

    # If no zoom level provided, use default
    if current_zoom is None:
        current_zoom = 2

    filtered_events = negative_events_df.copy()

    # ALWAYS start with base layers only - this clears everything
    base_layers = [
        dl.TileLayer(url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", opacity=0.7),
        dl.TileLayer(url="https://services.arcgisonline.com/arcgis/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}", opacity=1),
        dl.FullScreenControl()
    ]

    layers = []

    if selected_view == 'marker':
        counts = (
            filtered_events.groupby(['latitude', 'longitude', 'extracted_location'])
            .size().reset_index(name='count')
        )
        max_count = counts['count'].max()

        event_markers = [
            dl.CircleMarker(
                center=(row['latitude'], row['longitude']),
                radius=(row['count'] / max_count) * 30,
                color='red', fill=True, fillOpacity=0.6,
                id=f"event-circle-{i}",  # Add unique ID
                children=[
                    dl.Tooltip(f"{row['extracted_location']}: {row['count']} events"),
                    dl.Popup([html.B(row['extracted_location']), html.Br(), f"Event Count: {row['count']}"])
                ]
            ) for i, (_, row) in enumerate(counts.iterrows())
        ]

        layers.extend(event_markers)

    elif selected_view == 'choropleth':
        valid_countries = {country.name for country in pycountry.countries}

        choropleth_df = negative_events_df[negative_events_df["extracted_location"].isin(valid_countries)]

        geojson_url = 'https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json'
        geojson_countries = requests.get(geojson_url).json()

        country_name_mapping = {
            'United States': 'United States of America',
            'Hong Kong': 'China'
        }

        choropleth_df = choropleth_df.copy()
        choropleth_df['mapped_location'] = choropleth_df['extracted_location'].replace(country_name_mapping)

        choropleth_counts = choropleth_df.groupby('mapped_location').size().reset_index(name='count')

        # Recency-based border coloring
        now = datetime.now(sgt)
        recent_events = (
            choropleth_df.groupby('mapped_location')['published_date_sgt']
            .max()
            .reset_index(name='most_recent_date')
        )

        recent_events['recency_category'] = pd.cut(
            (now - recent_events['most_recent_date']).dt.days,
            bins=[-1, 1, 7, 30, 1e6],
            labels=['1_day', '1_week', '1_month', 'older']
        )

        recency_border_color = {
            '1_day': '#00FFFF',
            '1_week': '#0096FF',
            '1_month': '#ADD8E6',
            'older': '#6F8FAF'
        }

        recent_events['border_color'] = recent_events['recency_category'].map(recency_border_color)

        country_counts_dict = dict(zip(choropleth_counts['mapped_location'], choropleth_counts['count']))

        # Calculate log values for color scaling
        if len(choropleth_counts) > 0:
            log_counts = np.log1p(choropleth_counts['count'])
            min_log = log_counts.min()
            max_log = log_counts.max()
        else:
            min_log = max_log = 0

        reds_scale = [
            (255, 245, 240),
            (254, 224, 210),
            (252, 187, 161),
            (252, 146, 114),
            (251, 106, 74),
            (239, 59, 44),
            (203, 24, 29),
            (165, 15, 21),
            (103, 0, 13),
        ]

        def get_fill_color(count):
            if max_log > min_log:
                log_val = np.log1p(count)
                norm = (log_val - min_log) / (max_log - min_log)
            else:
                norm = 0
            idx = min(int(norm * (len(reds_scale) - 1)), len(reds_scale) - 1)
            r, g, b = reds_scale[idx]
            return f'rgb({r},{g},{b})'

        for feature in geojson_countries['features']:
            name = feature['properties']['name']

            if name in country_counts_dict:
                fill_val = country_counts_dict[name]
                fill_color = get_fill_color(fill_val)

                border_color_row = recent_events[recent_events['mapped_location'] == name]
                border_color = border_color_row['border_color'].values[0] if not border_color_row.empty else '#444444'

                feature['properties']['style'] = {
                    'fillColor': fill_color, 
                    'fillOpacity': 1,
                    'color': border_color,
                    'weight': 2,
                }
            else:
                feature['properties']['style'] = {
                    'fillOpacity': 0,
                    'weight': 0,
                    'color': 'rgba(0,0,0,0)',
                    'fillColor': 'rgba(0,0,0,0)'
                }

        # Filter GeoJSON to only show countries with events (for choropleth display)
        geojson_countries['features'] = [f for f in geojson_countries['features'] if f['properties']['style']['fillOpacity'] > 0]
        layers.append(dl.GeoJSON(data=geojson_countries))

        # ===== ZOOM-BASED EVENT DISPLAY LOGIC =====
        print(f"Current zoom level: {current_zoom}")  # Debug
        
        if current_zoom <= 3:
            # ZOOMED OUT: Show country statistics using geocode cache - only for events with icons
            print("Zoom <= 2: Creating country statistics")
            country_summary = create_country_statistics(choropleth_df)
            print(f"Country summary created: {len(country_summary)} countries")
            stat_markers = create_country_stat_markers(country_summary)
            print(f"Stat markers created: {len(stat_markers)} markers")
            layers.extend(stat_markers)
            
        elif 3.3 <= current_zoom <= 4:
            # MEDIUM ZOOM: Use MarkerClusterGroup 
            print("Zoom 3.3-4: Creating MarkerClusterGroup")
            cluster_markers = create_marker_cluster_for_medium_zoom(choropleth_df)
            layers.extend(cluster_markers)
            
        else:  # current_zoom >= 5
            # ZOOMED IN: Show individual event icons - only for events with matching icons
            print("Zoom >= 5: Creating individual event icons")
            # Filter choropleth events to only those with matching icons
            events_with_icons = filter_events_with_icons(choropleth_df, EVENT_ICON_MAP)
            print(f"Events with icons: {len(events_with_icons)} events")
            
            # Group by coordinates - using only events that have icons
            coord_groups = defaultdict(list)
            for _, row in events_with_icons.iterrows():
                lat, lon = row['latitude'], row['longitude']
                if pd.notna(lat) and pd.notna(lon):
                    coord_groups[(lat, lon)].append(row)

            event_icon_markers = []

            for (lat, lon), group_rows in coord_groups.items():
                if len(group_rows) > 1:
                    # Apply jitter in a circle around the original point (for events with overlapping lat/lon coordinates)
                    n = len(group_rows)
                    angle_step = 360 / n
                    jitter_radius = 0.15  # degrees

                    for i, row in enumerate(group_rows):
                        topic = str(row['event_topic']).lower()
                        matched_icon = None
                        for keyword, icon_path in EVENT_ICON_MAP.items():
                            if keyword in topic:
                                matched_icon = icon_path
                                break
                        if not matched_icon:
                            continue

                        angle = angle_step * i
                        jitter_lat = lat + jitter_radius * np.cos(np.radians(angle))
                        jitter_lon = lon + jitter_radius * np.sin(np.radians(angle))

                        event_icon_markers.append(
                            dl.Marker(
                                position=(jitter_lat, jitter_lon),
                                icon=dict(iconUrl=matched_icon, iconSize=[24, 24], iconAnchor=[15, 15]),
                                children=[
                                    dl.Tooltip(f"{row['event_topic']}"),
                                    dl.Popup([
                                        html.B(row['event_topic']),
                                        html.Br(),
                                        html.A("Open Link", href=row['url'], target="_blank")
                                    ])
                                ]
                            )
                        )
                else:
                    # Single event — no jitter
                    row = group_rows[0]
                    topic = str(row['event_topic']).lower()
                    matched_icon = None
                    for keyword, icon_path in EVENT_ICON_MAP.items():
                        if keyword in topic:
                            matched_icon = icon_path
                            break
                    if not matched_icon:
                        continue

                    event_icon_markers.append(
                        dl.Marker(
                            position=(lat, lon),
                            icon=dict(iconUrl=matched_icon, iconSize=[24, 24], iconAnchor=[15, 15]),
                            children=[
                                dl.Tooltip(f"{row['event_topic']}"),
                                dl.Popup([
                                    html.B(row['event_topic']),
                                    html.Br(),
                                    html.A("Open Link", href=row['url'], target="_blank")
                                ])
                            ]
                        )
                    )

            print(f"Individual event markers created: {len(event_icon_markers)} markers")
            layers.extend(event_icon_markers)

        # ===== Toggle natural disaster radius of impact Layer Control =====
        if 'show_radius' in show_radius_list and current_zoom >= 4:
            print("🔘 Toggle ON: Adding disaster radius circles")
            disaster_circles = create_disaster_radius_circles(choropleth_df)
            layers.extend(disaster_circles)
            print(f"Added {len(disaster_circles)} disaster radius circles")
        else:
            print("⚪ Toggle OFF: No radius circles")

        # ===== VENDOR TOWERS (appears together with clusters @ medium zoom) =====
        if current_zoom >= 3.3:
            # Aggregate vendor data by country
            country_vendor_data = aggregate_vendors_by_country()

            # Create country-based towers with individual vendor segments
            vendor_towers = []
            
            # Apply the same country mapping to vendor_df for tower positioning
            country_mapping = {
                'United States': 'United States of America',
                'USA': 'United States of America',
                'US': 'United States of America',
                'Hong Kong': 'China',
                'South Korea': 'Republic of Korea',
                'North Korea': 'Democratic People\'s Republic of Korea',
                'Russia': 'Russian Federation',
                'Iran': 'Islamic Republic of Iran',
                'Venezuela': 'Bolivarian Republic of Venezuela',
                'Bolivia': 'Plurinational State of Bolivia',
                'Tanzania': 'United Republic of Tanzania',
                'Macedonia': 'North Macedonia',
                'Moldova': 'Republic of Moldova',
                'Congo': 'Republic of the Congo',
                'DR Congo': 'Democratic Republic of the Congo',
                'Ivory Coast': "Côte d'Ivoire",
                'Cape Verde': 'Cabo Verde',
                'Swaziland': 'Eswatini',
                'UK': 'United Kingdom',
                'Britain': 'United Kingdom',
                'England': 'United Kingdom',
                'Scotland': 'United Kingdom',
                'Wales': 'United Kingdom',
                'Northern Ireland': 'United Kingdom'
            }
            
            # Create a copy of vendor_df with mapped countries
            vendor_df_for_towers = vendor_df.copy()
            vendor_df_for_towers['Tower_Country'] = vendor_df_for_towers['Mapped Country'].replace(country_mapping)
            
            # For each country with vendors, create a stacked tower
            for _, country_row in country_vendor_data.iterrows():
                country_name = country_row['GeoJSON_Country']
                
                # Get all vendors in this country
                country_vendors = vendor_df_for_towers[vendor_df_for_towers['Tower_Country'] == country_name]
                
                if country_vendors.empty:
                    continue
                
                # Create list of vendor tiers for stacking (each vendor = one segment)
                vendor_tiers = country_vendors['tier'].astype(str).tolist()
                total_vendors = len(vendor_tiers)
                
                # Use mean coordinates of all vendors in this country
                country_lat = country_vendors['lat'].mean()
                country_lon = country_vendors['lon'].mean()
                
                # Add very small offset to avoid overlapping with vendor icons
                offset_lat = country_lat + 0.05  # Very small offset (0.05 degrees ≈ 5.5km)
                offset_lon = country_lon + 0.05
                
                # Create country code (first 2-3 letters)
                country_code = country_name[:3].upper()
                
                # Generate tower SVG with individual vendor segments
                tower_svg = create_vendor_tower_svg(vendor_tiers, country_code)
                
                # Calculate icon size based on display mode
                if total_vendors > 100:
                    # Block mode: 3 blocks max, larger size
                    tower_pixel_height = 60  # Fixed reasonable height for block mode
                else:
                    # Individual mode: based on actual segments
                    tower_pixel_height = total_vendors * 3 + (total_vendors - 1) * 0.5 + 35
                
                icon_width = 35
                icon_height = max(tower_pixel_height, 45)  # Minimum 45px height for labels
                
                # Create tower marker with click event data using a simpler approach
                tower_marker = dl.Marker(
                    position=[offset_lat, offset_lon],
                    icon=dict(
                        iconUrl=f"data:image/svg+xml;base64,{base64.b64encode(tower_svg.encode('utf-8')).decode('utf-8')}",
                        iconSize=[icon_width, icon_height],
                        iconAnchor=[icon_width//2, icon_height-5],  # Anchor at bottom center
                        className='vendor-tower-icon'
                    ),
                    children=[
                        dl.Tooltip(
                            html.Div([
                                html.H5(f"{country_name} Vendors", style={'margin': '0', 'color': '#333'}),
                                html.P(f"Total: {total_vendors} vendors", style={'margin': '2px 0', 'color': '#333'}),
                                html.P(f"Gold: {vendor_tiers.count('gold')}", style={'margin': '2px 0', 'color': '#333'}),
                                html.P(f"Silver: {vendor_tiers.count('silver')}", style={'margin': '2px 0', 'color': '#333'}),
                                html.P(f"Bronze: {vendor_tiers.count('bronze')}", style={'margin': '2px 0', 'color': '#333'}),
                                html.P("💡 Click tower for full details", style={'margin': '2px 0', 'color': '#0066cc', 'fontStyle': 'italic', 'fontSize': '0.9em'})
                            ])
                        ),
                    ],
                    zIndexOffset=1500,
                    id=f"vendor-tower-{country_name.replace(' ', '-').replace('\'', '')}",
                    n_clicks=0
                )
                
                vendor_towers.append(tower_marker)

            # Add towers to layers
            layers.extend(vendor_towers)

            # VENDOR MARKERS WITH HALOS (only at high zoom)
            tier_colors = {
                'gold': '#FFD700',     # Gold
                'silver': '#C0C0C0',   # Silver  
                'bronze': '#CD853F'    # Bronze
            }
            
            vendor_compound_markers = []
            
            for i, (_, row) in enumerate(grouped_vendor_df.iterrows()):
                tier = row['tier']
                halo_color = tier_colors[tier]  # Direct color mapping
                
                # Background halo circle (same size for all tiers)
                halo_marker = dl.CircleMarker(
                    center=(row['lat'], row['lon']),
                    radius=15,  # Same size for all
                    color=halo_color,
                    weight=0.2,
                    fillOpacity=0.6,
                    fillColor=halo_color,
                    id=f"vendor-halo-{i}"  # Add unique ID
                )
                
                # Foreground badge icon  
                badge_marker = dl.Marker(
                    position=(row['lat'], row['lon']),
                    icon=dict(
                        iconUrl=row['icon'], 
                        iconSize=[35, 35],
                        iconAnchor=[17, 17]
                    ),
                    zIndexOffset=1000,
                    children=[
                        dl.Tooltip(
                            html.Div([item for name in row['Vendor Name'] for item in [html.Span(name), html.Br()]])
                        )
                    ],
                    id=f"vendor-badge-{i}"
                )
                
                vendor_compound_markers.extend([halo_marker, badge_marker])

            layers.extend(vendor_compound_markers)

    return base_layers + layers

# Callback function to control legend visibility
@app.callback(
    Output('map-legends-container', 'children'),
    Input('view-toggle', 'value'),
    prevent_initial_call=False
)
def update_map_legends(selected_view):
    if selected_view == 'choropleth':
        event_legend, vendor_legend = create_map_legends()
        return [event_legend, vendor_legend]
    else:
        return []
    
# Callback function to handle vendor tower clicks via direct marker n_clicks
@app.callback(
    [Output('event-modal', 'is_open', allow_duplicate=True),
     Output('modal-title', 'children', allow_duplicate=True),
     Output('modal-body', 'children', allow_duplicate=True)],
    [Input(f'vendor-tower-{country.replace(" ", "-").replace("\'", "")}', 'n_clicks') 
     for country in ['United-States-of-America', 'Singapore', 'China', 'France', 'Germany', 
                     'Austria', 'Brazil', 'Norway', 'Slovakia', 'Switzerland', 'Taiwan', 'Thailand']],
    [State('event-modal', 'is_open')],
    prevent_initial_call=True
)
def handle_vendor_tower_click(*args):
    # Last argument is the state (is_open), others are n_clicks
    n_clicks_list = args[:-1]
    is_open = args[-1]
    
    # Check if any tower was clicked
    if not any(n_clicks_list) or all(clicks is None or clicks == 0 for clicks in n_clicks_list):
        return dash.no_update, dash.no_update, dash.no_update
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    
    # Extract country name from the triggered component
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if 'vendor-tower-' in triggered_id:
        # Extract country code from the ID
        country_code = triggered_id.replace('vendor-tower-', '')
        
        # Convert back to readable country name
        country_name_mapping = {
            'United-States-of-America': 'United States of America',
            'Singapore': 'Singapore',
            'China': 'China',
            'France': 'France',
            'Germany': 'Germany',
            'Austria': 'Austria',
            'Brazil': 'Brazil',
            'Norway': 'Norway',
            'Slovakia': 'Slovakia',
            'Switzerland': 'Switzerland',
            'Taiwan': 'Taiwan',
            'Thailand': 'Thailand'
        }
        
        display_country = country_name_mapping.get(country_code, country_code.replace('-', ' '))
        
        # Apply the same country mapping to vendor_df for data retrieval
        country_mapping = {
            'United States': 'United States of America',
            'USA': 'United States of America',
            'US': 'United States of America',
            'Hong Kong': 'China',
        }
        
        # Create a copy of vendor_df with mapped countries
        vendor_df_for_modal = vendor_df.copy()
        vendor_df_for_modal['Modal_Country'] = vendor_df_for_modal['Mapped Country'].replace(country_mapping)
        
        # Get vendors for this country
        country_vendors = vendor_df_for_modal[vendor_df_for_modal['Modal_Country'] == display_country]
        
        if country_vendors.empty:
            modal_body = html.P("No vendor details found for this country.")
        else:
            # Group vendors by tier
            gold_vendors = country_vendors[country_vendors['tier'] == 'gold']
            silver_vendors = country_vendors[country_vendors['tier'] == 'silver']
            bronze_vendors = country_vendors[country_vendors['tier'] == 'bronze']
            
            # Create detailed vendor lists
            modal_content = []
            
            # Summary at top
            modal_content.extend([
                html.Div([
                    html.H6(f"Total Vendors: {len(country_vendors)}", style={'margin': '0 0 10px 0', 'color': '#333'}),
                    html.P(f"🥇 Gold: {len(gold_vendors)} | 🥈 Silver: {len(silver_vendors)} | 🥉 Bronze: {len(bronze_vendors)}", 
                           style={'margin': '0 0 15px 0', 'color': '#666'})
                ], style={'borderBottom': '1px solid #ddd', 'paddingBottom': '10px', 'marginBottom': '15px'})
            ])
            
            # Gold vendors section
            if not gold_vendors.empty:
                modal_content.extend([
                    html.H5("🥇 Gold Tier Vendors", style={'color': '#FFD700', 'marginTop': '15px'}),
                    html.Div([
                        html.Div([
                            html.Strong(row['Vendor Name']),
                            html.Br(),
                            html.Span(f"Order Value: ${row['Total Net Order Value']:,.0f}", style={'color': '#666', 'fontSize': '0.9em'}),
                            html.Br(),
                            html.Span(f"Code: {row['Vendor Code']}", style={'color': '#666', 'fontSize': '0.8em'})
                        ], style={'marginBottom': '8px', 'padding': '8px', 'border': '1px solid #FFD700', 'borderRadius': '4px', 'backgroundColor': '#fffbf0'})
                        for _, row in gold_vendors.iterrows()
                    ])
                ])
            
            # Silver vendors section
            if not silver_vendors.empty:
                modal_content.extend([
                    html.H5("🥈 Silver Tier Vendors", style={'color': '#C0C0C0', 'marginTop': '15px'}),
                    html.Div([
                        html.Div([
                            html.Strong(row['Vendor Name']),
                            html.Br(),
                            html.Span(f"Order Value: ${row['Total Net Order Value']:,.0f}", style={'color': '#666', 'fontSize': '0.9em'}),
                            html.Br(),
                            html.Span(f"Code: {row['Vendor Code']}", style={'color': '#666', 'fontSize': '0.8em'})
                        ], style={'marginBottom': '8px', 'padding': '8px', 'border': '1px solid #C0C0C0', 'borderRadius': '4px', 'backgroundColor': '#f8f8f8'})
                        for _, row in silver_vendors.iterrows()
                    ])
                ])
            
            # Bronze vendors section
            if not bronze_vendors.empty:
                modal_content.extend([
                    html.H5("🥉 Bronze Tier Vendors", style={'color': '#CD853F', 'marginTop': '15px'}),
                    html.Div([
                        html.Div([
                            html.Strong(row['Vendor Name']),
                            html.Br(),
                            html.Span(f"Order Value: ${row['Total Net Order Value']:,.0f}", style={'color': '#666', 'fontSize': '0.9em'}),
                            html.Br(),
                            html.Span(f"Code: {row['Vendor Code']}", style={'color': '#666', 'fontSize': '0.8em'})
                        ], style={'marginBottom': '8px', 'padding': '8px', 'border': '1px solid #CD853F', 'borderRadius': '4px', 'backgroundColor': '#faf6f0'})
                        for _, row in bronze_vendors.iterrows()
                    ])
                ])
            
            modal_body = html.Div(modal_content, style={'maxHeight': '500px', 'overflowY': 'auto'})
        
        modal_title = f"Vendor Details - {display_country}"
        
        return True, modal_title, modal_body
    
    return dash.no_update, dash.no_update, dash.no_update

# Vendor-Event Mapping Callbacks
@app.callback(
    Output('mapping-summary', 'children'),
    Input('generate-mapping-btn', 'n_clicks'),
    prevent_initial_call=True
)
def generate_mapping_summary(n_clicks):
    if n_clicks:
        summary = create_mapping_summary()
        if summary['total_mappings'] == 0:
            return dbc.Alert("No vendor-event mappings found.", color="warning")
        
        # Enhanced risk breakdown with color coding
        risk_breakdown = html.Div([
            html.H5("Risk Level Distribution", 
                   style={'color': 'white', 'marginBottom': '15px'}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Badge(
                            f"{risk}: {count}", 
                            color=get_risk_badge_color(risk), 
                            className="me-3 mb-2",
                            style={'fontSize': '0.9em', 'padding': '8px 12px'}
                        )
                        for risk, count in summary['risk_breakdown'].items()
                    ])
                ], width=12)
            ])
        ])
        
        # Enhanced summary cards with better styling
        summary_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(str(summary['total_mappings']), 
                               className="card-title text-center", 
                               style={'color': 'black', 'fontWeight': 'bold'}),
                        html.P("Total vendors-events Mappings", className="card-text text-center text-muted")
                    ])
                ], color="light", outline=True, className="shadow-sm")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(str(summary['unique_vendors']), 
                               className="card-title text-center", 
                               style={'color': 'red', 'fontWeight': 'bold'}),
                        html.P("Vendors Affected", className="card-text text-center text-muted")
                    ])
                ], color="light", outline=True, className="shadow-sm")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(str(summary['unique_events']), 
                               className="card-title text-center", 
                               style={'color': '#dc3545', 'fontWeight': 'bold'}),
                        html.P("Risk Events (unique)", className="card-text text-center text-muted")
                    ])
                ], color="light", outline=True, className="shadow-sm")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{summary['avg_distance']:.1f}km" if summary['avg_distance'] else "N/A", 
                               className="card-title text-center", 
                               style={'color': 'black', 'fontWeight': 'bold'}),
                        html.P("Avg Distance", className="card-text text-center text-muted")
                    ])
                ], color="light", outline=True, className="shadow-sm")
            ], width=3),
        ], className="mb-4", style={'margin-top':'20px'})
        
        return html.Div([risk_breakdown, summary_cards])
    return html.Div()

# callback for toggle visibility
@app.callback(
    Output('radius-controls', 'style'),
    Input('view-toggle', 'value')
)
def toggle_radius_controls(view_type):
    if view_type == 'choropleth':
        return {'display': 'block', 'marginTop': '15px', 'paddingTop': '15px', 'borderTop': '1px solid #444'}
    return {'display': 'none'}

# toggle_mapping modal callback
@app.callback(
    [Output('mapping-modal', 'is_open'), Output('mapping-modal-body', 'children')],
    [Input('view-table-btn', 'n_clicks')],
    [State('mapping-modal', 'is_open')],
    prevent_initial_call=True
)
def toggle_mapping_modal(n_clicks, is_open):
    if n_clicks:
        mapping_df = create_vendor_event_mapping()
        table_content = create_enhanced_mapping_table(mapping_df)
        return True, table_content
    return False, html.Div()

@app.callback(Output('download-csv', 'data'), Input('export-csv-btn', 'n_clicks'), prevent_initial_call=True)
def export_csv(n_clicks):
    if n_clicks:
        mapping_df = create_vendor_event_mapping()
        if mapping_df.empty:
            return dash.no_update
            
        # Select and rename columns for export
        export_df = mapping_df[[
            'Vendor_Name', 'Vendor_Country', 'Vendor_Tier', 'Vendor_Order_Value',
            'Event_Title', 'Event_Topic', 'Event_Location', 'Event_Date',
            'Risk_Level', 'Distance_KM', 'Mapping_Type'
        ]].copy()
        
        export_df.columns = [
            'Vendor Name', 'Vendor Country', 'Vendor Tier', 'Order Value',
            'Event Title', 'Event Topic', 'Event Location', 'Event Date', 
            'Risk Level', 'Distance (km)', 'Mapping Type'
        ]
        
        # Convert to CSV
        output = io.StringIO()
        export_df.to_csv(output, index=False)
        csv_content = output.getvalue()
        output.close()
        
        return dict(content=csv_content, filename="vendor_event_mapping_enhanced.csv")
    return dash.no_update

# pdf download callback
@app.callback(Output('download-pdf', 'data'), Input('export-pdf-btn', 'n_clicks'), prevent_initial_call=True)
def export_pdf(n_clicks):
    if n_clicks:
        pdf_content = export_mapping_to_pdf()
        if pdf_content:
            import base64
            encoded_pdf = base64.b64encode(pdf_content).decode('utf-8')
            return dict(content=encoded_pdf, filename="vendor_event_mapping_report.pdf", base64=True)
    return dash.no_update

if __name__ == '__main__':
    app.run()
