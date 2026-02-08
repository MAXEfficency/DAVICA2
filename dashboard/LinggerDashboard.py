"""
LINGGER'S DASHBOARD: Student Support Ecosystem Dashboard
CA2 Data Visualization Assignment - ST1502
Theme: Dark professional with blue/teal accents (complementary to Thomas's dashboard)
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime
import numpy as np

# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================

# Color scheme - Blue/Teal theme (complementary to Thomas's yellow/orange)
COLORS = {
    'background': '#0a1929',          # Dark navy blue (same as Thomas)
    'surface': '#132f4c',              # Slightly lighter navy
    'card': '#1e3a5f',                 # Card background
    'primary': '#06b6d4',              # Cyan/Teal accent (main difference)
    'secondary': '#0891b2',            # Darker teal  
    'success': '#10b981',              # Green
    'danger': '#ef4444',               # Red
    'warning': '#f59e0b',              # Orange
    'info': '#3b82f6',                 # Blue
    'text_primary': '#f1f5f9',         # Light text
    'text_secondary': '#94a3b8',       # Muted text
    'border': '#334155',               # Border color
    'grid': '#1e293b'                  # Grid lines
}

# Chart template configuration
CHART_TEMPLATE = {
    'layout': {
        'paper_bgcolor': COLORS['background'],
        'plot_bgcolor': COLORS['surface'],
        'font': {'color': COLORS['text_primary'], 'family': 'Inter, sans-serif'},
        'xaxis': {
            'gridcolor': COLORS['grid'],
            'linecolor': COLORS['border'],
            'tickfont': {'color': COLORS['text_secondary']}
        },
        'yaxis': {
            'gridcolor': COLORS['grid'],
            'linecolor': COLORS['border'],
            'tickfont': {'color': COLORS['text_secondary']}
        },
        'hovermode': 'closest',
        'margin': {'l': 60, 'r': 40, 't': 60, 'b': 60}
    }
}

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_prepare_data():
    """Load and prepare the master dataset with all necessary calculations"""
    
    df = pd.read_csv('../cleaned_data/master_dataset.csv')
    
    # Drop rows with NaN in critical columns
    df = df.dropna(subset=['PERIOD', 'GPA', 'STUDENT ID'])
    
    # Data type conversions
    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
    df['COMMENCEMENT DATE'] = pd.to_datetime(df['COMMENCEMENT DATE'], errors='coerce')
    df['COMPLETION DATE'] = pd.to_datetime(df['COMPLETION DATE'], errors='coerce')
    
    # Create age groups
    df['Age_Group'] = pd.cut(df['AGE'], 
                              bins=[0, 25, 35, 45, 100], 
                              labels=['18-25', '26-35', '36-45', '46+'])
    
    # Risk classification
    df['Initial_Risk'] = pd.cut(df['GPA'].where(df['PERIOD'] == 'Sem 1'),
                                 bins=[0, 2.5, 3.0, 4.0],
                                 labels=['High Risk', 'Medium Risk', 'Low Risk'])
    df['Initial_Risk'] = df.groupby('STUDENT ID')['Initial_Risk'].transform('first')
    
    # Pass/Fail status
    df['Pass_Status'] = df['GPA'].apply(lambda x: 'Pass' if x >= 2.0 else 'Fail')
    
    # Handle missing values
    df['ATTENDANCE'] = df['ATTENDANCE'].fillna(0)
    df['SELF-STUDY HRS'] = df['SELF-STUDY HRS'].fillna(0)
    
    # Fill support columns with median
    support_cols = ['TEACHING SUPPORT', 'COMPANY SUPPORT', 'FAMILY SUPPORT', 'COURSE RELEVANCE']
    for col in support_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Attendance categories
    df['Attendance_Category'] = pd.cut(df['ATTENDANCE'],
                                        bins=[0, 60, 75, 85, 100],
                                        labels=['Critical (<60%)', 'Low (60-75%)', 
                                               'Good (75-85%)', 'Excellent (85%+)'])
    
    # Study hours categories
    df['Study_Category'] = pd.cut(df['SELF-STUDY HRS'],
                                   bins=[0, 5, 10, 15, 100],
                                   labels=['Minimal (0-5h)', 'Low (5-10h)', 
                                          'Moderate (10-15h)', 'High (15h+)'])
    
    # Clean period names
    df['Period_Clean'] = df['PERIOD'].str.replace('Sem ', 'Semester ')
    
    # Course code extraction
    df['Course_Code'] = df['STUDENT ID'].str.extract(r'^(\d{4})-')[0]
    
    return df

# ============================================================================
# CHART GENERATION FUNCTIONS
# ============================================================================

def create_nationality_study_effort(df, selected_nationality=None):
    """
    CHART 1: Nationality & Study Effort Comparison - Plotly Express Box Plot
    Shows study hours distribution by nationality status
    """
    
    # Filter if specific nationality selected
    if selected_nationality and selected_nationality != 'all':
        df_filtered = df[df['NATIONALITY_STATUS'] == selected_nationality]
    else:
        df_filtered = df.copy()
    
    # Create box plot
    fig = px.box(df_filtered,
                 x='NATIONALITY_STATUS',
                 y='SELF-STUDY HRS',
                 color='NATIONALITY_STATUS',
                 points='outliers',
                 hover_data=['GPA', 'ATTENDANCE'],
                 color_discrete_map={
                     'SG Citizen': '#3b82f6',
                     'SG PR': '#8b5cf6',
                     'Foreigner': '#06b6d4'
                 },
                 category_orders={'NATIONALITY_STATUS': ['SG Citizen', 'SG PR', 'Foreigner']})
    
    # Calculate averages for annotation
    avg_by_nat = df_filtered.groupby('NATIONALITY_STATUS')['SELF-STUDY HRS'].mean()
    
    layout_config = CHART_TEMPLATE['layout'].copy()
    layout_config.update({
        'title': {
            'text': '<b>Study Effort by Nationality Status</b><br><sub>Comparing self-study hours across student backgrounds</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': COLORS['text_primary']}
        },
        'xaxis_title': 'Nationality Status',
        'yaxis_title': 'Weekly Self-Study Hours',
        'showlegend': False,
        'height': 400
    })
    
    fig.update_layout(**layout_config)
    
    # Add average line annotations
    for nat, avg in avg_by_nat.items():
        fig.add_annotation(
            x=nat,
            y=avg,
            text=f"Avg: {avg:.1f}h",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=COLORS['primary'],
            ax=40,
            ay=-30,
            font=dict(size=10, color=COLORS['primary'])
        )
    
    return fig


def create_support_factors_impact(df, selected_factor='all'):
    """
    CHART 2: Support Factors - ALIGNED VERSION
    Fixes: Height=450 (matches Chart 3), Stretched to right, Filter aligned.
    """
    fig = go.Figure()
    
    # Support factors configuration
    support_factors = {
        'TEACHING SUPPORT': {'name': 'Teaching Support', 'color': '#3b82f6'},
        'COMPANY SUPPORT': {'name': 'Company Support', 'color': '#8b5cf6'},
        'FAMILY SUPPORT': {'name': 'Family Support', 'color': '#ec4899'},
        'COURSE RELEVANCE': {'name': 'Course Relevance', 'color': '#06b6d4'}
    }
    
    trace_keys = []
    
    # 1. Create Traces
    for col, info in support_factors.items():
        if col not in df.columns:
            continue
            
        support_impact = df.groupby(col)['GPA'].mean().reset_index().sort_values(col)
        trace_keys.append(col)
        
        fig.add_trace(go.Scatter(
            x=support_impact[col],
            y=support_impact['GPA'],
            mode='lines+markers',
            name=info['name'],
            line=dict(color=info['color'], width=3),
            marker=dict(size=12, symbol='circle'),
            visible=(selected_factor == 'all' or col == selected_factor),
            hovertemplate=f'<b>{info["name"]}</b><br>Level: %{{x}}<br>GPA: %{{y:.2f}}<extra></extra>'
        ))

    # 2. Dropdown Logic
    dropdown_buttons = [
        dict(
            label='📊 All Factors (Overlay)',
            method='update',
            args=[
                {'visible': [True] * len(trace_keys)},
                {'title.text': '<b>Support Factors Impact on GPA</b><br><sub>Comparing all environmental support systems</sub>'}
            ]
        )
    ]
    
    for i, (col, info) in enumerate(support_factors.items()):
        if col not in df.columns: continue
        
        visibility_list = [False] * len(trace_keys)
        if i < len(visibility_list):
            visibility_list[i] = True
            
        dropdown_buttons.append(dict(
            label=f"{info['name']}",
            method='update',
            args=[
                {'visible': visibility_list},
                {'title.text': f"<b>{info['name']} Impact on GPA</b><br><sub>Analysis of specific support factor</sub>"}
            ]
        ))

    # 3. Layout Configuration (UPDATED)
    layout_config = CHART_TEMPLATE['layout'].copy()
    layout_config.update({
        'title': {
            'text': '<b>Support Factors Impact on GPA</b><br><sub>Comparing all environmental support systems</sub>',
        },
        'xaxis_title': 'Support Level (1=Low, 5=High)',
        'yaxis_title': 'Average GPA',
        'yaxis_range': [1.5, 4.0],
        
        # --- ALIGNMENT FIXES ---
        'height': 450,                          # Increased to match Chart 3
        'margin': {'l': 60, 'r': 20, 't': 80, 'b': 60}, # r=20 stretches graph to right
        # -----------------------
        
        'updatemenus': [
            dict(
                buttons=dropdown_buttons,
                direction="down",
                pad={"r": 0, "t": 0},
                showactive=True,
                x=1.0,              # Top Right
                xanchor="right",
                y=1.15,
                yanchor="top",
                bgcolor=COLORS['card'],
                bordercolor=COLORS['primary'],
                borderwidth=1,
                font=dict(color=COLORS['text_primary'], size=11)
            )
        ]
    })
    
    fig.update_layout(**layout_config)
    
    # Add passing threshold line
    fig.add_hline(y=2.0, line_dash="dash", line_color=COLORS['danger'],
                  annotation_text="Passing Threshold", annotation_position="right")
    
    return fig

def create_attendance_study_compensation_heatmap(df, view_mode='pass_fail'):
    """
    CHART 3: Attendance vs Study Hours Matrix - ENHANCED
    Fixes: Stretched layout, numbers on all charts, clearer grid lines.
    """
    
    # Create bins for attendance and study hours
    df_analysis = df.copy()
    df_analysis['Att_Bin'] = pd.cut(df_analysis['ATTENDANCE'],
                                     bins=[0, 60, 75, 85, 100],
                                     labels=['<60%', '60-75%', '75-85%', '85%+'])
    df_analysis['Study_Bin'] = pd.cut(df_analysis['SELF-STUDY HRS'],
                                      bins=[0, 5, 10, 15, 100],
                                      labels=['0-5h', '5-10h', '10-15h', '15h+'])
    
    fig = go.Figure()
    
    # Common font settings for the numbers inside the boxes
    text_style = {"family": "Inter, sans-serif", "size": 14, "color": "white"}

    # --- View 1: Pass/Fail Zones ---
    passfail_matrix = df_analysis.groupby(['Study_Bin', 'Att_Bin']).apply(
        lambda x: (x['Pass_Status'] == 'Pass').sum() / len(x) * 100 if len(x) > 0 else 0
    ).reset_index()
    passfail_pivot = passfail_matrix.pivot(index='Study_Bin', columns='Att_Bin', values=0).fillna(0)
    
    hover_text_1 = []
    for i, study in enumerate(passfail_pivot.index):
        row = []
        for j, att in enumerate(passfail_pivot.columns):
            value = passfail_pivot.iloc[i, j]
            count = len(df_analysis[(df_analysis['Study_Bin'] == study) & (df_analysis['Att_Bin'] == att)])
            row.append(f"<b>Study: {study}</b><br>Attendance: {att}<br>Pass Rate: {value:.1f}%<br>Students: {count}")
        hover_text_1.append(row)
    
    fig.add_trace(go.Heatmap(
        z=passfail_pivot.values,
        x=passfail_pivot.columns.tolist(),
        y=passfail_pivot.index.tolist(),
        colorscale=[[0, COLORS['danger']], [0.5, COLORS['warning']], [1, COLORS['success']]],
        text=hover_text_1,
        texttemplate="%{z:.0f}%", # Adds numbers (e.g. 85%)
        textfont=text_style,
        xgap=2, ygap=2,           # Adds gaps for clarity
        hovertemplate='%{text}<extra></extra>',
        visible=(view_mode == 'pass_fail')
    ))
    
    # --- View 2: GPA Gradient ---
    gpa_matrix = df_analysis.groupby(['Study_Bin', 'Att_Bin'])['GPA'].mean().reset_index()
    gpa_pivot = gpa_matrix.pivot(index='Study_Bin', columns='Att_Bin', values='GPA').fillna(0)
    
    hover_text_2 = []
    for i, study in enumerate(gpa_pivot.index):
        row = []
        for j, att in enumerate(gpa_pivot.columns):
            value = gpa_pivot.iloc[i, j]
            count = len(df_analysis[(df_analysis['Study_Bin'] == study) & (df_analysis['Att_Bin'] == att)])
            row.append(f"<b>Study: {study}</b><br>Attendance: {att}<br>Avg GPA: {value:.2f}<br>Students: {count}")
        hover_text_2.append(row)
    
    fig.add_trace(go.Heatmap(
        z=gpa_pivot.values,
        x=gpa_pivot.columns.tolist(),
        y=gpa_pivot.index.tolist(),
        colorscale=[[0, '#1e293b'], [0.33, COLORS['danger']], [0.66, COLORS['warning']], [1, COLORS['success']]],
        text=hover_text_2,
        texttemplate="%{z:.2f}",  # Adds numbers (e.g. 3.42)
        textfont=text_style,
        xgap=2, ygap=2,
        hovertemplate='%{text}<extra></extra>',
        visible=(view_mode == 'gpa_gradient')
    ))
    
    # --- View 3: Student Count ---
    count_matrix = df_analysis.groupby(['Study_Bin', 'Att_Bin']).size().reset_index()
    count_pivot = count_matrix.pivot(index='Study_Bin', columns='Att_Bin', values=0).fillna(0)
    
    hover_text_3 = []
    for i, study in enumerate(count_pivot.index):
        row = []
        for j, att in enumerate(count_pivot.columns):
            value = int(count_pivot.iloc[i, j])
            row.append(f"<b>Study: {study}</b><br>Attendance: {att}<br>Students: {value}")
        hover_text_3.append(row)
    
    fig.add_trace(go.Heatmap(
        z=count_pivot.values,
        x=count_pivot.columns.tolist(),
        y=count_pivot.index.tolist(),
        colorscale=[[0, COLORS['surface']], [0.5, COLORS['info']], [1, COLORS['primary']]],
        text=hover_text_3,
        texttemplate="%{z:.0f}",  # Adds numbers (e.g. 120)
        textfont=text_style,
        xgap=2, ygap=2,
        hovertemplate='%{text}<extra></extra>',
        visible=(view_mode == 'student_count')
    ))
    
    # --- Radio Buttons (Fixed Location & Title Bug) ---
    radio_buttons = [
        dict(
            label='✅ Pass/Fail Zones',
            method='update',
            args=[{'visible': [True, False, False]},
                  {'title.text': '<b>Compensation Matrix: Pass/Fail</b><br><sub>Can high study hours fix low attendance?</sub>'}]
        ),
        dict(
            label='📊 GPA Gradient',
            method='update',
            args=[{'visible': [False, True, False]},
                  {'title.text': '<b>Performance Matrix: Avg GPA</b><br><sub>Average GPA across effort combinations</sub>'}]
        ),
        dict(
            label='👥 Student Distribution',
            method='update',
            args=[{'visible': [False, False, True]},
                  {'title.text': '<b>Population Matrix: Student Count</b><br><sub>Number of students in each effort category</sub>'}]
        )
    ]
    
    layout_config = CHART_TEMPLATE['layout'].copy()
    layout_config.update({
        'title': {
            'text': '<b>Compensation Matrix: Pass/Fail</b><br><sub>Can high study hours fix low attendance?</sub>',
        },
        'xaxis_title': 'Attendance Level',
        'yaxis_title': 'Weekly Study Hours',
        'height': 450,
        # Reduced margins to let the chart expand
        'margin': {'l': 60, 'r': 20, 't': 80, 'b': 60}, 
        'updatemenus': [
            dict(
                buttons=radio_buttons,
                direction="down",
                pad={"r": 0, "t": 0},
                showactive=True,
                x=1.0,              # Moves dropdown to inside-right (Fixed)
                xanchor="right",
                y=1.15,
                yanchor="top",
                bgcolor=COLORS['card'],
                bordercolor=COLORS['primary'],
                borderwidth=1,
                font=dict(color=COLORS['text_primary'], size=11)
            )
        ]
    })
    
    fig.update_layout(**layout_config)
    
    return fig

def create_age_attendance_discipline_slider(df, threshold=75):
    """
    CHART 4: Risk Composition - HORIZONTAL Stacked Bar
    Title Updated: More descriptive about the simulation aspect.
    """
    
    fig = go.Figure()
    
    age_groups = ['46+', '36-45', '26-35', '18-25']
    
    risk_pcts = []
    safe_pcts = []
    risk_counts = []
    safe_counts = []
    risk_text = []
    safe_text = []
    
    for age_grp in age_groups:
        age_data = df[df['Age_Group'] == age_grp]
        total = len(age_data)
        
        at_risk = (age_data['ATTENDANCE'] < threshold).sum()
        safe = total - at_risk
        
        r_pct = (at_risk / total * 100) if total > 0 else 0
        s_pct = (safe / total * 100) if total > 0 else 0
        
        risk_pcts.append(r_pct)
        safe_pcts.append(s_pct)
        risk_counts.append(at_risk)
        safe_counts.append(safe)
        
        if r_pct > 5:
            risk_text.append(f"<b>{r_pct:.1f}%</b><br>({at_risk})")
        else:
            risk_text.append("")
            
        safe_text.append(f"<b>{s_pct:.1f}%</b>")
    
    # Trace 1: AT RISK
    fig.add_trace(go.Bar(
        name=f'Fails Requirement (< {threshold}%)', # Legend explains the logic
        y=age_groups,
        x=risk_pcts,
        orientation='h',
        marker_color=COLORS['danger'],
        text=risk_text,
        textposition='auto',
        hovertemplate='<b>Age: %{y}</b><br>Fail Rate: %{x:.1f}%<br>Students: %{customdata}<extra></extra>',
        customdata=risk_counts
    ))
    
    # Trace 2: SAFE
    fig.add_trace(go.Bar(
        name=f'Meets Requirement (≥ {threshold}%)', # Legend explains the logic
        y=age_groups,
        x=safe_pcts,
        orientation='h',
        marker_color=COLORS['success'],
        text=safe_text,
        textposition='auto',
        hovertemplate='<b>Age: %{y}</b><br>Pass Rate: %{x:.1f}%<br>Students: %{customdata}<extra></extra>',
        customdata=safe_counts
    ))

    layout_config = CHART_TEMPLATE['layout'].copy()
    layout_config.update({
        # --- NEW CLEARER TITLE ---
        'title': f'<b>Simulate Risk Scenarios: Who fails the attendance rule?</b><br><sub>Showing % of students below the {threshold}% threshold (Adjust slider below)</sub>',
        'xaxis_title': 'Percentage of Group',
        'yaxis_title': 'Age Group',
        'barmode': 'stack',
        'showlegend': True,
        'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1},
        'height': 350,
        'xaxis': {'range': [0, 100], 'gridcolor': COLORS['grid']},
        'margin': {'l': 80, 'r': 40, 't': 80, 'b': 40} 
    })
    
    fig.update_layout(**layout_config)
    
    # --- Stats return remains same ---
    summary_data = []
    normal_order = ['18-25', '26-35', '36-45', '46+']
    for age in normal_order:
         age_data = df[df['Age_Group'] == age]
         total = len(age_data)
         at_risk = (age_data['ATTENDANCE'] < threshold).sum()
         summary_data.append({
            'Age_Group': age, 
            'At_Risk_Pct': (at_risk / total * 100) if total > 0 else 0,
            'At_Risk_Count': at_risk,
            'Total': total
         })
         
    return fig, pd.DataFrame(summary_data)

# ============================================================================
# KPI CALCULATION
# ============================================================================

def calculate_kpis(df, selected_period=None, selected_course=None):
    """Calculate KPIs based on filters"""
    
    filtered_df = df.copy()
    
    if selected_period and selected_period != 'all':
        filtered_df = filtered_df[filtered_df['PERIOD'] == selected_period]
    
    if selected_course and selected_course != 'all':
        filtered_df = filtered_df[filtered_df['Course_Code'] == selected_course]
    
    # Calculate metrics
    total_students = filtered_df['STUDENT ID'].nunique()
    
    # Support seeking percentage (students with any support > 3)
    if all(col in filtered_df.columns for col in ['TEACHING SUPPORT', 'COMPANY SUPPORT', 'FAMILY SUPPORT']):
        high_support = filtered_df[
            (filtered_df['TEACHING SUPPORT'] >= 4) |
            (filtered_df['COMPANY SUPPORT'] >= 4) |
            (filtered_df['FAMILY SUPPORT'] >= 4)
        ]['STUDENT ID'].nunique()
        support_seeking_pct = (high_support / total_students * 100) if total_students > 0 else 0
    else:
        support_seeking_pct = 0
    
    # Average support rating
    if all(col in filtered_df.columns for col in ['TEACHING SUPPORT', 'COMPANY SUPPORT', 'FAMILY SUPPORT']):
        avg_support = filtered_df[['TEACHING SUPPORT', 'COMPANY SUPPORT', 'FAMILY SUPPORT']].mean().mean()
    else:
        avg_support = 0
    
    # High risk percentage
    high_risk_count = filtered_df[filtered_df['GPA'] < 2.5]['STUDENT ID'].nunique()
    high_risk_pct = (high_risk_count / total_students * 100) if total_students > 0 else 0
    
    return {
        'total_students': total_students,
        'avg_support': avg_support,
        'high_risk_pct': high_risk_pct,
        'support_seeking_pct': support_seeking_pct
    }


def create_kpi_card(title, value, subtitle, icon, color='primary'):
    """Create a Bootstrap KPI card component"""
    
    color_map = {
        'primary': COLORS['primary'],
        'danger': COLORS['danger'],
        'success': COLORS['success'],
        'info': COLORS['info']
    }
    
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(icon, style={
                    'fontSize': '2rem',
                    'color': color_map[color],
                    'marginRight': '10px'
                }),
                html.Div([
                    html.H6(title, style={
                        'color': COLORS['text_secondary'],
                        'fontSize': '0.875rem',
                        'fontWeight': '500',
                        'marginBottom': '0.25rem'
                    }),
                    html.H3(value, style={
                        'color': COLORS['text_primary'],
                        'fontSize': '1.875rem',
                        'fontWeight': '700',
                        'marginBottom': '0.25rem'
                    }),
                    html.P(subtitle, style={
                        'color': COLORS['text_secondary'],
                        'fontSize': '0.75rem',
                        'marginBottom': '0'
                    })
                ])
            ], style={'display': 'flex', 'alignItems': 'center'})
        ])
    ], style={
        'backgroundColor': COLORS['card'],
        'border': f'1px solid {COLORS["border"]}',
        'borderRadius': '8px',
        'marginBottom': '1rem'
    })


# ============================================================================
# DASH APP INITIALIZATION
# ============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Load data
df = load_and_prepare_data()

# ============================================================================
# LAYOUT
# ============================================================================

app.layout = dbc.Container([
    
    # Header Section
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H2([
                    html.Span("🤝 ", style={'marginRight': '10px'}),
                    "Student Support Ecosystem Dashboard"
                ], style={
                    'color': COLORS['text_primary'],
                    'fontWeight': '700',
                    'marginBottom': '0.5rem'
                }),
                html.P("Analyzing environmental factors and support systems that drive student success", 
                       style={'color': COLORS['text_secondary'], 'fontSize': '0.95rem'})
            ], style={'padding': '1.5rem 0'})
        ])
    ]),
    
    # Dashboard Switcher
    dbc.Row([
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button(
                    "🎯 Thomas - Risk Monitor",
                    id='btn-thomas',
                    color='secondary',
                    outline=True,
                    style={
                        'borderColor': COLORS['border'],
                        'color': COLORS['text_primary']
                    },
                    href='http://127.0.0.1:8050',
                    external_link=True
                ),
                dbc.Button(
                    "🤝 Lingger - Support Systems",
                    id='btn-lingger',
                    color='info',
                    className='active',
                    style={
                        'backgroundColor': COLORS['primary'],
                        'borderColor': COLORS['primary'],
                        'color': COLORS['background'],
                        'fontWeight': '600'
                    }
                )
            ], style={'marginBottom': '1rem'})
        ], width=12)
    ]),
    
    # Global Filters Row
    dbc.Row([
        # Filter 1: Semester
        dbc.Col([
            html.Label("Select Semester", style={'color': COLORS['text_secondary'], 'fontSize': '0.875rem'}),
            dcc.Dropdown(
                id='semester-filter',
                options=[{'label': 'All Semesters', 'value': 'all'}] + 
                        [{'label': period, 'value': period} for period in sorted([p for p in df['PERIOD'].unique() if pd.notna(p)])],
                value='all',
                clearable=False
            )
        ], width=3),
        
        # Filter 2: Course
        dbc.Col([
            html.Label("Select Course", style={'color': COLORS['text_secondary'], 'fontSize': '0.875rem'}),
            dcc.Dropdown(
                id='course-filter',
                options=[{'label': 'All Courses', 'value': 'all'}] +
                        [{'label': f'Course {code}', 'value': code} for code in sorted([c for c in df['Course_Code'].unique() if pd.notna(c)])],
                value='all',
                clearable=False
            )
        ], width=3),
        
        # Filter 3: Nationality (IGNORED BY CHART 1)
        dbc.Col([
            html.Label("Nationality Filter", style={'color': COLORS['text_secondary'], 'fontSize': '0.875rem'}),
            dcc.Dropdown(
                id='nationality-filter',
                options=[
                    {'label': 'All Nationalities', 'value': 'all'},
                    {'label': '🇸🇬 SG Citizen', 'value': 'SG Citizen'},
                    {'label': '🏠 SG PR', 'value': 'SG PR'},
                    {'label': '🌏 Foreigner', 'value': 'Foreigner'}
                ],
                value='all',
                clearable=False
            )
        ], width=3),
        
        # Filter 4: Age Group (NEW! REPLACES SLIDER) (IGNORED BY CHART 4)
        dbc.Col([
            html.Label("Age Group Filter", style={'color': COLORS['text_secondary'], 'fontSize': '0.875rem'}),
            dcc.Dropdown(
                id='age-filter',
                options=[{'label': 'All Ages', 'value': 'all'}] + 
                        [{'label': age, 'value': age} for age in sorted([a for a in df['Age_Group'].unique() if pd.notna(a)])],
                value='all',
                clearable=False
            )
        ], width=3)
    ], style={'marginBottom': '2rem'}),
    
    # KPI Cards Row
    dbc.Row([
        dbc.Col(html.Div(id='kpi-total-students'), width=3),
        dbc.Col(html.Div(id='kpi-avg-support'), width=3),
        dbc.Col(html.Div(id='kpi-high-risk'), width=3),
        dbc.Col(html.Div(id='kpi-support-seeking'), width=3)
    ], style={'marginBottom': '2rem'}),
    
    # Chart 1: Nationality Study Effort (Full Width)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='chart-nationality-study', config={'displayModeBar': False})
                ])
            ], style={
                'backgroundColor': COLORS['card'],
                'border': f'1px solid {COLORS["border"]}',
                'borderRadius': '8px'
            })
        ], width=12)
    ], style={'marginBottom': '1.5rem'}),
    
    # Charts 2 & 3: Support Factors + Compensation Matrix (Side by Side)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='chart-support-factors', config={'displayModeBar': False})
                ])
            ], style={
                'backgroundColor': COLORS['card'],
                'border': f'1px solid {COLORS["border"]}',
                'borderRadius': '8px'
            })
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='chart-compensation-matrix', config={'displayModeBar': False})
                ])
            ], style={
                'backgroundColor': COLORS['card'],
                'border': f'1px solid {COLORS["border"]}',
                'borderRadius': '8px'
            })
        ], width=6)
    ], style={'marginBottom': '1.5rem'}),
    
    # Chart 4: Age Attendance Discipline (Full Width)
    dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        # Graph
                        dcc.Graph(id='chart-age-attendance', config={'displayModeBar': False}),
                        
                        # Slider Control Section (New Location)
                        html.Div([
                            html.Label("🎚️ Adjust Attendance Passing Threshold:", 
                                    style={'color': COLORS['primary'], 'fontWeight': 'bold', 'marginBottom': '10px'}),
                            dcc.Slider(
                                id='global-attendance-slider', # ID stays the same so callback works
                                min=50,
                                max=100,
                                step=5,
                                value=75,
                                marks={i: {'label': f'{i}%', 'style': {'color': COLORS['text_secondary']}} for i in range(50, 101, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'padding': '0px 20px 20px 20px'}), # Padding to separate from graph
                        
                        html.Div(id='attendance-discipline-insights', style={'marginTop': '1rem'})
                    ])
                ], style={
                    'backgroundColor': COLORS['card'],
                    'border': f'1px solid {COLORS["border"]}',
                    'borderRadius': '8px'
                })
            ], width=12)
        ], style={'marginBottom': '2rem'}),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(style={'borderColor': COLORS['border']}),
            html.P([
                "CA2 Data Visualization Assignment • ST1502 • ",
                html.Span("Lingger's Dashboard", style={'color': COLORS['primary'], 'fontWeight': '600'}),
                " • AY2526 Sem 2"
            ], style={
                'textAlign': 'center',
                'color': COLORS['text_secondary'],
                'fontSize': '0.875rem',
                'marginTop': '1rem'
            })
        ])
    ])
    
], fluid=True, style={
    'backgroundColor': COLORS['background'],
    'minHeight': '100vh',
    'padding': '2rem'
})


# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [Output('kpi-total-students', 'children'),
     Output('kpi-avg-support', 'children'),
     Output('kpi-high-risk', 'children'),
     Output('kpi-support-seeking', 'children'),
     Output('chart-nationality-study', 'figure'),
     Output('chart-support-factors', 'figure'),
     Output('chart-compensation-matrix', 'figure'),
     Output('chart-age-attendance', 'figure'),
     Output('attendance-discipline-insights', 'children')],
    [Input('semester-filter', 'value'),
     Input('course-filter', 'value'),
     Input('nationality-filter', 'value'),
     Input('age-filter', 'value'),
     Input('global-attendance-slider', 'value')]
)

def update_dashboard(semester, course, nationality, age_group, attendance_threshold):
    # 1. Base Filter (Semester & Course affect everything)
    df_base = df.copy()
    if semester != 'all':
        df_base = df_base[df_base['PERIOD'] == semester]
    if course != 'all':
        df_base = df_base[df_base['Course_Code'] == course]

    # 2. Fully Filtered Data (For KPIs, Chart 2, Chart 3)
    filtered_df = df_base.copy()
    if nationality != 'all':
        filtered_df = filtered_df[filtered_df['NATIONALITY_STATUS'] == nationality]
    if age_group != 'all':
        filtered_df = filtered_df[filtered_df['Age_Group'] == age_group]

    # 3. Data for Chart 1 (Nationality Box Plot)
    # IGNORES nationality filter, but applies Age filter
    df_chart1 = df_base.copy()
    if age_group != 'all':
        df_chart1 = df_chart1[df_chart1['Age_Group'] == age_group]

    # 4. Data for Chart 4 (Age Risk Bar)
    # IGNORES age filter, but applies Nationality filter
    df_chart4 = df_base.copy()
    if nationality != 'all':
        df_chart4 = df_chart4[df_chart4['NATIONALITY_STATUS'] == nationality]
    
    # --- KPI CALCULATION (FIXED ARGUMENTS) ---
    # We pass semester and course explicitly to match your original function signature
    kpis = calculate_kpis(filtered_df, semester if semester != 'all' else None,
                          course if course != 'all' else None)
    
    kpi1 = create_kpi_card("Total Students", f"{kpis['total_students']:,}", "Unique students analyzed", "👥", 'info')
    kpi2 = create_kpi_card("Avg Support Rating", f"{kpis['avg_support']:.1f}/5", "Across all support factors", "🤝", 'primary')
    kpi3 = create_kpi_card("High Risk %", f"{kpis['high_risk_pct']:.1f}%", "Students with GPA < 2.5", "⚠️", 'danger')
    kpi4 = create_kpi_card("High Support %", f"{kpis['support_seeking_pct']:.1f}%", "Students with support ≥ 4", "🌟", 'success')
    
    # --- GENERATE CHARTS ---
    
    # Chart 1: Uses df_chart1 (All Nationalities)
    chart1 = create_nationality_study_effort(
        df_chart1,
        selected_nationality=None 
    )
    
    # Chart 2: Uses filtered_df (Fully Filtered)
    chart2 = create_support_factors_impact(filtered_df)
    
    # Chart 3: Uses filtered_df (Fully Filtered)
    chart3 = create_attendance_study_compensation_heatmap(filtered_df)
    
    # Chart 4: Uses df_chart4 (All Ages)
    chart4, discipline_stats = create_age_attendance_discipline_slider(df_chart4, attendance_threshold)
    
    # --- INSIGHTS (WITH CRASH PROTECTION) ---
    if not discipline_stats.empty and discipline_stats['Total'].sum() > 0:
        worst_age = discipline_stats.loc[discipline_stats['At_Risk_Pct'].idxmax()]
        best_age = discipline_stats.loc[discipline_stats['At_Risk_Pct'].idxmin()]
        
        insights = dbc.Alert([
            html.H6(f"💡 Attendance Discipline Insights (Threshold: {attendance_threshold}%):", style={'marginBottom': '0.5rem'}),
            html.Ul([
                html.Li(f"🔴 Highest Risk: {worst_age['Age_Group']} ({worst_age['At_Risk_Pct']:.1f}% fail rate)"),
                html.Li(f"🟢 Lowest Risk: {best_age['Age_Group']} ({best_age['At_Risk_Pct']:.1f}% fail rate)"),
                html.Li(f"📊 Overall Impact: {discipline_stats['At_Risk_Count'].sum():.0f} students fail attendance requirements")
            ], style={'marginBottom': 0})
        ], color='info', style={'backgroundColor': COLORS['surface'], 'borderColor': COLORS['info'], 'color': COLORS['text_primary']})
    else:
        # Fallback if filter results in no students
        insights = dbc.Alert("⚠️ No student data available for this filter combination.", color='warning')

    return kpi1, kpi2, kpi3, kpi4, chart1, chart2, chart3, chart4, insights

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == '__main__':
    app.run(debug=True, port=8051)