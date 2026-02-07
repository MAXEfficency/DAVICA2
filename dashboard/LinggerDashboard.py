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
    CHART 2: Support Factors Impact Analysis - Graph Objects with DROPDOWN
    Shows how different support levels affect GPA
    """
    
    fig = go.Figure()
    
    # Support factors to analyze
    support_factors = {
        'TEACHING SUPPORT': {'name': 'Teaching Support', 'color': '#3b82f6'},
        'COMPANY SUPPORT': {'name': 'Company Support', 'color': '#8b5cf6'},
        'FAMILY SUPPORT': {'name': 'Family Support', 'color': '#ec4899'},
        'COURSE RELEVANCE': {'name': 'Course Relevance', 'color': '#06b6d4'}
    }
    
    traces = {}
    
    # Create traces for each support factor
    for col, info in support_factors.items():
        if col not in df.columns:
            continue
            
        # Group by support level and calculate average GPA
        support_impact = df.groupby(col)['GPA'].mean().reset_index()
        support_impact = support_impact.sort_values(col)
        
        trace_key = col.lower().replace(' ', '_')
        traces[trace_key] = go.Scatter(
            x=support_impact[col],
            y=support_impact['GPA'],
            mode='lines+markers',
            name=info['name'],
            line=dict(color=info['color'], width=3),
            marker=dict(size=12, symbol='circle'),
            visible=(selected_factor == 'all' or col == selected_factor),
            hovertemplate='<b>Support Level: %{x}</b><br>Avg GPA: %{y:.2f}<extra></extra>'
        )
    
    # Create "All Factors" overlay view
    all_factors_data = []
    for col, info in support_factors.items():
        if col in df.columns:
            support_impact = df.groupby(col)['GPA'].mean().reset_index()
            for _, row in support_impact.iterrows():
                all_factors_data.append({
                    'Support_Level': row[col],
                    'GPA': row['GPA'],
                    'Factor': info['name']
                })
    
    all_df = pd.DataFrame(all_factors_data)
    
    for factor_name, info in support_factors.items():
        factor_label = info['name']
        factor_data = all_df[all_df['Factor'] == factor_label]
        
        traces[f'all_{factor_label.lower().replace(" ", "_")}'] = go.Scatter(
            x=factor_data['Support_Level'],
            y=factor_data['GPA'],
            mode='lines+markers',
            name=factor_label,
            line=dict(color=info['color'], width=2),
            marker=dict(size=8),
            visible=(selected_factor == 'all'),
            hovertemplate=f'<b>{factor_label}</b><br>Level: %{{x}}<br>GPA: %{{y:.2f}}<extra></extra>'
        )
    
    # Add all traces
    for trace in traces.values():
        fig.add_trace(trace)
    
    # Create dropdown menu
    dropdown_buttons = [
        dict(
            label='📊 All Factors (Overlay)',
            method='update',
            args=[{'visible': ['all_' in k for k in traces.keys()]},
                  {'title': '<b>Support Factors Impact on GPA</b><br><sub>Comparing all environmental support systems</sub>'}]
        ),
        dict(
            label='👨‍🏫 Teaching Support',
            method='update',
            args=[{'visible': [k == 'teaching_support' for k in traces.keys()]},
                  {'title': '<b>Teaching Support Impact on GPA</b><br><sub>Effect of instructional support on performance</sub>'}]
        ),
        dict(
            label='🏢 Company Support',
            method='update',
            args=[{'visible': [k == 'company_support' for k in traces.keys()]},
                  {'title': '<b>Company Support Impact on GPA</b><br><sub>Effect of workplace support on performance</sub>'}]
        ),
        dict(
            label='👨‍👩‍👧 Family Support',
            method='update',
            args=[{'visible': [k == 'family_support' for k in traces.keys()]},
                  {'title': '<b>Family Support Impact on GPA</b><br><sub>Effect of family support on performance</sub>'}]
        ),
        dict(
            label='🎯 Course Relevance',
            method='update',
            args=[{'visible': [k == 'course_relevance' for k in traces.keys()]},
                  {'title': '<b>Course Relevance Impact on GPA</b><br><sub>Effect of perceived relevance on performance</sub>'}]
        )
    ]
    
    layout_config = CHART_TEMPLATE['layout'].copy()
    layout_config.update({
        'title': '<b>Support Factors Impact on GPA</b><br><sub>Comparing all environmental support systems</sub>',
        'xaxis_title': 'Support Level (1=Low, 5=High)',
        'yaxis_title': 'Average GPA',
        'yaxis_range': [1.5, 4.0],
        'height': 400,
        'updatemenus': [
            dict(
                buttons=dropdown_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.02,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor=COLORS['card'],
                bordercolor=COLORS['primary'],
                borderwidth=2,
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
    CHART 3: Attendance vs Study Hours Compensation Matrix - Graph Objects Heatmap with RADIO BUTTONS
    Shows if low attendance can be compensated by high study hours
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
    
    # View Mode 1: Pass/Fail Zones
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
            text = f"<b>Study: {study}</b><br>Attendance: {att}<br>Pass Rate: {value:.1f}%<br>Students: {count}"
            row.append(text)
        hover_text_1.append(row)
    
    fig.add_trace(go.Heatmap(
        z=passfail_pivot.values,
        x=passfail_pivot.columns.tolist(),
        y=passfail_pivot.index.tolist(),
        colorscale=[[0, COLORS['danger']], [0.5, COLORS['warning']], [1, COLORS['success']]],
        text=hover_text_1,
        hovertemplate='%{text}<extra></extra>',
        showscale=True,
        colorbar=dict(
            title="Pass Rate %",
            tickfont=dict(color=COLORS['text_secondary'])
        ),
        visible=(view_mode == 'pass_fail')
    ))
    
    # View Mode 2: GPA Gradient
    gpa_matrix = df_analysis.groupby(['Study_Bin', 'Att_Bin'])['GPA'].mean().reset_index()
    gpa_pivot = gpa_matrix.pivot(index='Study_Bin', columns='Att_Bin', values='GPA').fillna(0)
    
    hover_text_2 = []
    for i, study in enumerate(gpa_pivot.index):
        row = []
        for j, att in enumerate(gpa_pivot.columns):
            value = gpa_pivot.iloc[i, j]
            count = len(df_analysis[(df_analysis['Study_Bin'] == study) & (df_analysis['Att_Bin'] == att)])
            text = f"<b>Study: {study}</b><br>Attendance: {att}<br>Avg GPA: {value:.2f}<br>Students: {count}"
            row.append(text)
        hover_text_2.append(row)
    
    fig.add_trace(go.Heatmap(
        z=gpa_pivot.values,
        x=gpa_pivot.columns.tolist(),
        y=gpa_pivot.index.tolist(),
        colorscale=[[0, '#1e293b'], [0.33, COLORS['danger']], [0.66, COLORS['warning']], [1, COLORS['success']]],
        text=hover_text_2,
        hovertemplate='%{text}<extra></extra>',
        showscale=True,
        colorbar=dict(
            title="Avg GPA",
            tickfont=dict(color=COLORS['text_secondary'])
        ),
        visible=(view_mode == 'gpa_gradient')
    ))
    
    # View Mode 3: Student Count
    count_matrix = df_analysis.groupby(['Study_Bin', 'Att_Bin']).size().reset_index()
    count_pivot = count_matrix.pivot(index='Study_Bin', columns='Att_Bin', values=0).fillna(0)
    
    hover_text_3 = []
    for i, study in enumerate(count_pivot.index):
        row = []
        for j, att in enumerate(count_pivot.columns):
            value = int(count_pivot.iloc[i, j])
            text = f"<b>Study: {study}</b><br>Attendance: {att}<br>Students: {value}"
            row.append(text)
        hover_text_3.append(row)
    
    fig.add_trace(go.Heatmap(
        z=count_pivot.values,
        x=count_pivot.columns.tolist(),
        y=count_pivot.index.tolist(),
        colorscale=[[0, COLORS['surface']], [0.5, COLORS['info']], [1, COLORS['primary']]],
        text=hover_text_3,
        hovertemplate='%{text}<extra></extra>',
        showscale=True,
        colorbar=dict(
            title="# Students",
            tickfont=dict(color=COLORS['text_secondary'])
        ),
        visible=(view_mode == 'student_count')
    ))
    
    # Create radio buttons
    radio_buttons = [
        dict(
            label='✅ Pass/Fail Zones',
            method='update',
            args=[{'visible': [True, False, False]},
                  {'title': '<b>Compensation Matrix: Can Study Make Up for Attendance?</b><br><sub>Pass rate by attendance and study hours (Green=Safe, Red=Danger)</sub>'}]
        ),
        dict(
            label='📊 GPA Gradient',
            method='update',
            args=[{'visible': [False, True, False]},
                  {'title': '<b>Performance Matrix: GPA by Attendance & Study</b><br><sub>Average GPA across effort combinations</sub>'}]
        ),
        dict(
            label='👥 Student Distribution',
            method='update',
            args=[{'visible': [False, False, True]},
                  {'title': '<b>Population Matrix: Where Are Students?</b><br><sub>Number of students in each effort category</sub>'}]
        )
    ]
    
    layout_config = CHART_TEMPLATE['layout'].copy()
    layout_config.update({
        'title': '<b>Compensation Matrix: Can Study Make Up for Attendance?</b><br><sub>Pass rate by attendance and study hours (Green=Safe, Red=Danger)</sub>',
        'xaxis_title': 'Attendance Level',
        'yaxis_title': 'Weekly Study Hours',
        'height': 450,
        'updatemenus': [
            dict(
                buttons=radio_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.02,
                xanchor="left",
                y=1.12,
                yanchor="top",
                bgcolor=COLORS['card'],
                bordercolor=COLORS['primary'],
                borderwidth=2,
                font=dict(color=COLORS['text_primary'], size=11)
            )
        ]
    })
    
    fig.update_layout(**layout_config)
    
    return fig


def create_age_attendance_discipline_slider(df, threshold=75):
    """
    CHART 4: Age-Based Attendance Discipline Analysis - Graph Objects Box Plot with SLIDER
    Shows which age groups struggle with attendance, with adjustable threshold
    """
    
    fig = go.Figure()
    
    # Create box plots for each age group
    age_groups = ['18-25', '26-35', '36-45', '46+']
    colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f97316']
    
    for i, age_grp in enumerate(age_groups):
        age_data = df[df['Age_Group'] == age_grp]
        
        # Calculate risk status
        below_threshold = (age_data['ATTENDANCE'] < threshold).sum()
        total = len(age_data)
        risk_pct = (below_threshold / total * 100) if total > 0 else 0
        
        fig.add_trace(go.Box(
            y=age_data['ATTENDANCE'],
            name=f'{age_grp}<br>({risk_pct:.0f}% at-risk)',
            marker_color=COLORS['danger'] if risk_pct > 30 else (COLORS['warning'] if risk_pct > 15 else COLORS['success']),
            boxmean='sd',
            hovertemplate='<b>Age: ' + age_grp + '</b><br>Attendance: %{y:.1f}%<extra></extra>'
        ))
    
    # Add threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=COLORS['primary'],
        line_width=2,
        annotation_text=f"Threshold: {threshold}%",
        annotation_position="left"
    )
    
    layout_config = CHART_TEMPLATE['layout'].copy()
    layout_config.update({
        'title': '<b>Attendance Discipline by Age Group</b><br><sub>Which demographics struggle most with attendance? (Use slider to adjust threshold)</sub>',
        'xaxis_title': 'Age Group',
        'yaxis_title': 'Attendance Rate (%)',
        'yaxis_range': [0, 105],
        'showlegend': False,
        'height': 450
    })
    
    fig.update_layout(**layout_config)
    
    # Calculate summary statistics
    summary_stats = []
    for age_grp in age_groups:
        age_data = df[df['Age_Group'] == age_grp]
        below = (age_data['ATTENDANCE'] < threshold).sum()
        total = len(age_data)
        avg_att = age_data['ATTENDANCE'].mean()
        
        summary_stats.append({
            'Age_Group': age_grp,
            'At_Risk_Count': below,
            'Total': total,
            'At_Risk_Pct': (below / total * 100) if total > 0 else 0,
            'Avg_Attendance': avg_att
        })
    
    return fig, pd.DataFrame(summary_stats)


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
        
        dbc.Col([
            html.Label("Attendance Threshold", style={'color': COLORS['text_secondary'], 'fontSize': '0.875rem'}),
            dcc.Slider(
                id='global-attendance-slider',
                min=50,
                max=100,
                step=5,
                value=75,
                marks={i: f'{i}%' for i in range(50, 101, 10)},
                tooltip={"placement": "bottom", "always_visible": True}
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
                    dcc.Graph(id='chart-age-attendance', config={'displayModeBar': False}),
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
     Input('global-attendance-slider', 'value')]
)
def update_dashboard(semester, course, nationality, attendance_threshold):
    """Main callback to update all dashboard components"""
    
    # Filter data
    filtered_df = df.copy()
    
    if semester != 'all':
        filtered_df = filtered_df[filtered_df['PERIOD'] == semester]
    
    if course != 'all':
        filtered_df = filtered_df[filtered_df['Course_Code'] == course]
    
    if nationality != 'all':
        filtered_df = filtered_df[filtered_df['NATIONALITY_STATUS'] == nationality]
    
    # Calculate KPIs
    kpis = calculate_kpis(filtered_df, semester if semester != 'all' else None,
                         course if course != 'all' else None)
    
    # Create KPI cards
    kpi1 = create_kpi_card(
        "Total Students",
        f"{kpis['total_students']:,}",
        "Unique students analyzed",
        "👥",
        'info'
    )
    
    kpi2 = create_kpi_card(
        "Avg Support Rating",
        f"{kpis['avg_support']:.1f}/5",
        "Across all support factors",
        "🤝",
        'primary'
    )
    
    kpi3 = create_kpi_card(
        "High Risk %",
        f"{kpis['high_risk_pct']:.1f}%",
        "Students with GPA < 2.5",
        "⚠️",
        'danger'
    )
    
    kpi4 = create_kpi_card(
        "High Support %",
        f"{kpis['support_seeking_pct']:.1f}%",
        "Students with support ≥ 4",
        "🌟",
        'success'
    )
    
    # Generate charts
    chart1 = create_nationality_study_effort(
        filtered_df,
        selected_nationality=nationality if nationality != 'all' else None
    )
    
    chart2 = create_support_factors_impact(filtered_df)
    
    chart3 = create_attendance_study_compensation_heatmap(filtered_df)
    
    chart4, discipline_stats = create_age_attendance_discipline_slider(filtered_df, attendance_threshold)
    
    # Create discipline insights
    worst_age = discipline_stats.loc[discipline_stats['At_Risk_Pct'].idxmax()]
    best_age = discipline_stats.loc[discipline_stats['At_Risk_Pct'].idxmin()]
    
    insights = dbc.Alert([
        html.H6("💡 Attendance Discipline Insights:", style={'marginBottom': '0.5rem'}),
        html.Ul([
            html.Li(f"🔴 Highest Risk: {worst_age['Age_Group']} age group with {worst_age['At_Risk_Pct']:.1f}% below {attendance_threshold}% threshold ({worst_age['At_Risk_Count']:.0f} students)"),
            html.Li(f"🟢 Lowest Risk: {best_age['Age_Group']} age group with {best_age['At_Risk_Pct']:.1f}% below threshold ({best_age['At_Risk_Count']:.0f} students)"),
            html.Li(f"📊 Overall: {discipline_stats['At_Risk_Count'].sum():.0f} out of {discipline_stats['Total'].sum():.0f} students fall below {attendance_threshold}% attendance"),
            html.Li(f"💡 Recommendation: Focus attendance interventions on {worst_age['Age_Group']} age group")
        ], style={'marginBottom': 0})
    ], color='info', style={
        'backgroundColor': COLORS['surface'],
        'borderColor': COLORS['info'],
        'color': COLORS['text_primary']
    })
    
    return kpi1, kpi2, kpi3, kpi4, chart1, chart2, chart3, chart4, insights


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == '__main__':
    app.run(debug=True, port=8051)