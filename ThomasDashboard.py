"""
THOMAS'S DASHBOARD: Student Risk & Performance Monitor
CA2 Data Visualization Assignment - ST1502
Theme: Dark professional with yellow/orange accents (matching reference design)
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

# Color scheme matching the reference image
COLORS = {
    'background': '#0a1929',          # Dark navy blue
    'surface': '#132f4c',              # Slightly lighter navy
    'card': '#1e3a5f',                 # Card background
    'primary': '#fbbf24',              # Yellow/amber accent
    'secondary': '#f59e0b',            # Orange accent  
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
        'title': {'font': {'size': 16, 'color': COLORS['text_primary']}},
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
    
    df = pd.read_csv('cleaned_data/master_dataset.csv')
    
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
    
    # Risk classification based on Sem 1 GPA
    df['Initial_Risk'] = pd.cut(df['GPA'].where(df['PERIOD'] == 'Sem 1'),
                                 bins=[0, 2.5, 3.0, 4.0],
                                 labels=['High Risk', 'Medium Risk', 'Low Risk'])
    df['Initial_Risk'] = df.groupby('STUDENT ID')['Initial_Risk'].transform('first')
    
    # Pass/Fail status
    df['Pass_Status'] = df['GPA'].apply(lambda x: 'Pass' if x >= 2.0 else 'Fail')
    
    # Attendance categories (fill NaN with 0 before categorizing)
    df['ATTENDANCE'] = df['ATTENDANCE'].fillna(0)
    df['Attendance_Category'] = pd.cut(df['ATTENDANCE'],
                                        bins=[0, 60, 75, 85, 100],
                                        labels=['Critical (<60%)', 'Low (60-75%)', 
                                               'Good (75-85%)', 'Excellent (85%+)'])
    
    # Study hours categories (fill NaN with 0 before categorizing)
    df['SELF-STUDY HRS'] = df['SELF-STUDY HRS'].fillna(0)
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

def create_course_difficulty_bubble(df, selected_courses=None):
    """
    CHART 1: Course Difficulty Matrix (Bubble Chart) - Plotly Express
    Shows GPA vs Failure Rate with enrollment as bubble size
    """
    
    # Calculate course-level statistics
    course_stats = df.groupby('Course_Code').agg({
        'GPA': 'mean',
        'Pass_Status': lambda x: (x == 'Fail').sum() / len(x) * 100,
        'STUDENT ID': 'nunique'
    }).reset_index()
    
    course_stats.columns = ['Course_Code', 'Avg_GPA', 'Failure_Rate', 'Enrollment']
    
    # Filter if specific courses selected
    if selected_courses:
        course_stats = course_stats[course_stats['Course_Code'].isin(selected_courses)]
    
    # Create bubble chart
    fig = px.scatter(course_stats,
                     x='Avg_GPA',
                     y='Failure_Rate',
                     size='Enrollment',
                     color='Failure_Rate',
                     hover_name='Course_Code',
                     hover_data={
                         'Avg_GPA': ':.2f',
                         'Failure_Rate': ':.1f',
                         'Enrollment': ':,',
                         'Course_Code': False
                     },
                     size_max=60,
                     color_continuous_scale=['#10b981', '#fbbf24', '#ef4444'],
                     range_color=[0, 50])
    
    # Create a modified template without title
    layout_config = CHART_TEMPLATE['layout'].copy()
    layout_config['title'] = {
        'text': '<b>Course Difficulty Matrix</b><br><sub>Size = Enrollment | Color = Failure Rate</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 16, 'color': COLORS['text_primary']}
    }
    layout_config['xaxis_title'] = 'Average GPA'
    layout_config['yaxis_title'] = 'Failure Rate (%)'
    layout_config['showlegend'] = False
    layout_config['height'] = 400
    
    fig.update_layout(**layout_config)
    
    # Add quadrant lines
    fig.add_hline(y=25, line_dash="dash", line_color=COLORS['border'], opacity=0.5)
    fig.add_vline(x=3.0, line_dash="dash", line_color=COLORS['border'], opacity=0.5)
    
    # Add annotations for quadrants
    fig.add_annotation(x=3.5, y=45, text="Low Risk", showarrow=False, 
                      font=dict(color=COLORS['success'], size=10))
    fig.add_annotation(x=2.5, y=45, text="⚠️ High Risk", showarrow=False,
                      font=dict(color=COLORS['danger'], size=10, weight='bold'))
    
    return fig


def create_gpa_trajectory_with_dropdown(df, view_mode='risk_level'):
    """
    CHART 2: GPA Trajectory (Line Chart) - Graph Objects with DROPDOWN
    Multiple view modes: Risk Level, Age Group, Course Type, All Students
    """
    
    # Filter to retained students only (completed all semesters)
    semester_counts = df.groupby('STUDENT ID')['PERIOD'].nunique()
    retained_students = semester_counts[semester_counts >= 2].index
    df_retained = df[df['STUDENT ID'].isin(retained_students)].copy()
    
    fig = go.Figure()
    
    # Create traces for all view modes
    traces = {}
    
    # View Mode 1: By Risk Level
    for risk in ['High Risk', 'Medium Risk', 'Low Risk']:
        risk_data = df_retained[df_retained['Initial_Risk'] == risk]
        trajectory = risk_data.groupby('Period_Clean')['GPA'].mean().reset_index()
        
        color_map = {'High Risk': COLORS['danger'], 
                     'Medium Risk': COLORS['warning'], 
                     'Low Risk': COLORS['success']}
        
        traces[f'risk_{risk}'] = go.Scatter(
            x=trajectory['Period_Clean'],
            y=trajectory['GPA'],
            mode='lines+markers',
            name=risk,
            line=dict(color=color_map[risk], width=3),
            marker=dict(size=10, symbol='circle'),
            visible=(view_mode == 'risk_level'),
            hovertemplate='<b>%{x}</b><br>Avg GPA: %{y:.2f}<extra></extra>'
        )
    
    # View Mode 2: By Age Group
    for age_grp in ['18-25', '26-35', '36-45', '46+']:
        age_data = df_retained[df_retained['Age_Group'] == age_grp]
        trajectory = age_data.groupby('Period_Clean')['GPA'].mean().reset_index()
        
        color_map = {'18-25': '#3b82f6', '26-35': '#8b5cf6', 
                     '36-45': '#ec4899', '46+': '#f97316'}
        
        traces[f'age_{age_grp}'] = go.Scatter(
            x=trajectory['Period_Clean'],
            y=trajectory['GPA'],
            mode='lines+markers',
            name=f'Age {age_grp}',
            line=dict(color=color_map[age_grp], width=3),
            marker=dict(size=10),
            visible=(view_mode == 'age_group'),
            hovertemplate='<b>%{x}</b><br>Avg GPA: %{y:.2f}<extra></extra>'
        )
    
    # View Mode 3: By Course Type (based on course code patterns)
    df_retained['Course_Type'] = df_retained['Course_Code'].apply(
        lambda x: 'Certificate' if int(x) < 2000 else ('Diploma' if int(x) < 3000 else 'Specialist')
    )
    
    for course_type in ['Certificate', 'Diploma', 'Specialist']:
        type_data = df_retained[df_retained['Course_Type'] == course_type]
        trajectory = type_data.groupby('Period_Clean')['GPA'].mean().reset_index()
        
        color_map = {'Certificate': '#06b6d4', 'Diploma': '#8b5cf6', 
                     'Specialist': '#f59e0b'}
        
        traces[f'course_{course_type}'] = go.Scatter(
            x=trajectory['Period_Clean'],
            y=trajectory['GPA'],
            mode='lines+markers',
            name=course_type,
            line=dict(color=color_map[course_type], width=3),
            marker=dict(size=10),
            visible=(view_mode == 'course_type'),
            hovertemplate='<b>%{x}</b><br>Avg GPA: %{y:.2f}<extra></extra>'
        )
    
    # View Mode 4: All Students
    all_trajectory = df_retained.groupby('Period_Clean')['GPA'].mean().reset_index()
    traces['all_students'] = go.Scatter(
        x=all_trajectory['Period_Clean'],
        y=all_trajectory['GPA'],
        mode='lines+markers',
        name='All Students',
        line=dict(color=COLORS['primary'], width=4),
        marker=dict(size=12, symbol='diamond'),
        visible=(view_mode == 'all'),
        hovertemplate='<b>%{x}</b><br>Avg GPA: %{y:.2f}<extra></extra>'
    )
    
    # Add all traces to figure
    for trace in traces.values():
        fig.add_trace(trace)
    
    # Create dropdown menu
    dropdown_buttons = [
        dict(label='📊 By Initial Risk Level',
             method='update',
             args=[{'visible': [k.startswith('risk_') for k in traces.keys()]},
                   {'title': '<b>GPA Trajectory by Initial Risk Level</b><br><sub>Students who completed all semesters</sub>'}]),
        dict(label='👥 By Age Group',
             method='update',
             args=[{'visible': [k.startswith('age_') for k in traces.keys()]},
                   {'title': '<b>GPA Trajectory by Age Group</b><br><sub>Performance across different age demographics</sub>'}]),
        dict(label='🎓 By Course Type',
             method='update',
             args=[{'visible': [k.startswith('course_') for k in traces.keys()]},
                   {'title': '<b>GPA Trajectory by Course Type</b><br><sub>Certificate vs Diploma vs Specialist</sub>'}]),
        dict(label='🌐 All Students',
             method='update',
             args=[{'visible': [k == 'all_students' for k in traces.keys()]},
                   {'title': '<b>Overall GPA Trajectory</b><br><sub>Average performance across all students</sub>'}])
    ]
    
    # Create layout configuration
    layout_config = CHART_TEMPLATE['layout'].copy()
    layout_config.update({
        'title': '<b>GPA Trajectory by Initial Risk Level</b><br><sub>Students who completed all semesters</sub>',
        'xaxis_title': 'Semester',
        'yaxis_title': 'Average GPA',
        'yaxis_range': [1.5, 4.0],
        'hovermode': 'x unified',
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
                  annotation_text="Passing Threshold (2.0)", 
                  annotation_position="right")
    
    return fig


def create_risk_hotspot_heatmap(df, metric='failure_rate'):
    """
    CHART 3: Risk Hotspot Heatmap - Graph Objects with RADIO BUTTONS
    Shows Age Group vs Course with toggleable metrics
    """
    
    fig = go.Figure()
    
    # Prepare data for all three metrics
    metrics_data = {}
    
    # Metric 1: Failure Rate
    failure_pivot = df.groupby(['Age_Group', 'Course_Code']).apply(
        lambda x: (x['Pass_Status'] == 'Fail').sum() / len(x) * 100
    ).reset_index()
    failure_pivot.columns = ['Age_Group', 'Course_Code', 'Value']
    failure_matrix = failure_pivot.pivot(index='Age_Group', columns='Course_Code', values='Value').fillna(0)
    
    metrics_data['failure_rate'] = {
        'z': failure_matrix.values,
        'x': failure_matrix.columns.tolist(),
        'y': failure_matrix.index.tolist(),
        'colorscale': [[0, COLORS['success']], [0.3, COLORS['primary']], [1, COLORS['danger']]],
        'text_suffix': '%',
        'title': '<b>Risk Hotspot: Failure Rate by Age & Course</b><br><sub>% of students failing in each segment</sub>'
    }
    
    # Metric 2: Average GPA
    gpa_pivot = df.groupby(['Age_Group', 'Course_Code'])['GPA'].mean().reset_index()
    gpa_pivot.columns = ['Age_Group', 'Course_Code', 'Value']
    gpa_matrix = gpa_pivot.pivot(index='Age_Group', columns='Course_Code', values='Value').fillna(0)
    
    metrics_data['avg_gpa'] = {
        'z': gpa_matrix.values,
        'x': gpa_matrix.columns.tolist(),
        'y': gpa_matrix.index.tolist(),
        'colorscale': [[0, COLORS['danger']], [0.5, COLORS['primary']], [1, COLORS['success']]],
        'text_suffix': '',
        'title': '<b>Risk Hotspot: Average GPA by Age & Course</b><br><sub>Performance levels across segments</sub>'
    }
    
    # Metric 3: At-Risk Count
    risk_pivot = df.groupby(['Age_Group', 'Course_Code']).apply(
        lambda x: (x['GPA'] < 2.5).sum()
    ).reset_index()
    risk_pivot.columns = ['Age_Group', 'Course_Code', 'Value']
    risk_matrix = risk_pivot.pivot(index='Age_Group', columns='Course_Code', values='Value').fillna(0)
    
    metrics_data['risk_count'] = {
        'z': risk_matrix.values,
        'x': risk_matrix.columns.tolist(),
        'y': risk_matrix.index.tolist(),
        'colorscale': [[0, COLORS['surface']], [0.5, COLORS['warning']], [1, COLORS['danger']]],
        'text_suffix': ' students',
        'title': '<b>Risk Hotspot: At-Risk Student Count</b><br><sub>Number of students with GPA < 2.5</sub>'
    }
    
    # Create heatmap traces (only one visible at a time)
    for metric_key, metric_data in metrics_data.items():
        
        # Create custom hover text
        hover_text = []
        for i, age in enumerate(metric_data['y']):
            row = []
            for j, course in enumerate(metric_data['x']):
                value = metric_data['z'][i][j]
                text = f"<b>Age: {age}</b><br>Course: {course}<br>Value: {value:.1f}{metric_data['text_suffix']}"
                row.append(text)
            hover_text.append(row)
        
        fig.add_trace(go.Heatmap(
            z=metric_data['z'],
            x=metric_data['x'],
            y=metric_data['y'],
            colorscale=metric_data['colorscale'],
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            showscale=True,
            colorbar=dict(
                title=dict(text="", side="right"),
                tickfont=dict(color=COLORS['text_secondary']),
                outlinecolor=COLORS['border']
            ),
            visible=(metric == metric_key)
        ))
    
    # Create radio button menu
    radio_buttons = [
        dict(label='🔴 Failure Rate %',
             method='update',
             args=[{'visible': [True, False, False]},
                   {'title': metrics_data['failure_rate']['title']}]),
        dict(label='📊 Average GPA',
             method='update',
             args=[{'visible': [False, True, False]},
                   {'title': metrics_data['avg_gpa']['title']}]),
        dict(label='⚠️ At-Risk Count',
             method='update',
             args=[{'visible': [False, False, True]},
                   {'title': metrics_data['risk_count']['title']}])
    ]
    
    # Create layout configuration
    layout_config = CHART_TEMPLATE['layout'].copy()
    layout_config.update({
        'title': metrics_data[metric]['title'],
        'xaxis_title': 'Course Code',
        'yaxis_title': 'Age Group',
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


def create_attendance_threshold_slider(df, threshold=75):
    """
    CHART 4: Attendance Threshold Analysis - Graph Objects with SLIDER
    Combo chart (bar + line) showing pass rate and GPA by attendance
    """
    
    # Create attendance bands
    attendance_bands = list(range(50, 101, 5))
    
    threshold_stats = []
    for band in attendance_bands:
        band_data = df[df['ATTENDANCE'] >= band]
        if len(band_data) > 0:
            pass_rate = (band_data['Pass_Status'] == 'Pass').sum() / len(band_data) * 100
            avg_gpa = band_data['GPA'].mean()
            student_count = len(band_data)
            at_risk = (band_data['GPA'] < 2.5).sum()
        else:
            pass_rate = avg_gpa = student_count = at_risk = 0
        
        threshold_stats.append({
            'Threshold': f'{band}%+',
            'Band': band,
            'Pass_Rate': pass_rate,
            'Avg_GPA': avg_gpa,
            'Student_Count': student_count,
            'At_Risk': at_risk
        })
    
    stats_df = pd.DataFrame(threshold_stats)
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add pass rate bars
    fig.add_trace(
        go.Bar(
            x=stats_df['Threshold'],
            y=stats_df['Pass_Rate'],
            name='Pass Rate %',
            marker_color=COLORS['primary'],
            text=stats_df['Pass_Rate'].round(1),
            texttemplate='%{text}%',
            textposition='outside',
            hovertemplate='<b>Attendance: %{x}</b><br>Pass Rate: %{y:.1f}%<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add GPA line
    fig.add_trace(
        go.Scatter(
            x=stats_df['Threshold'],
            y=stats_df['Avg_GPA'],
            name='Average GPA',
            mode='lines+markers',
            line=dict(color=COLORS['danger'], width=3),
            marker=dict(size=10, symbol='diamond'),
            yaxis='y2',
            hovertemplate='<b>Attendance: %{x}</b><br>Avg GPA: %{y:.2f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Add threshold marker
    threshold_idx = (threshold - 50) // 5
    if threshold_idx < len(stats_df):
        fig.add_vline(
            x=threshold_idx,
            line_dash="dash",
            line_color=COLORS['info'],
            line_width=2,
            annotation_text=f"Current Threshold: {threshold}%",
            annotation_position="top"
        )
    
    # Create layout configuration
    layout_config = CHART_TEMPLATE['layout'].copy()
    layout_config.update({
        'title': '<b>Attendance Threshold Impact Analysis</b><br><sub>Pass rate & GPA by minimum attendance requirement</sub>',
        'xaxis_title': 'Minimum Attendance Requirement',
        'height': 400,
        'hovermode': 'x unified',
        'legend': dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            bgcolor=COLORS['card'],
            bordercolor=COLORS['border'],
            borderwidth=1
        )
    })
    
    fig.update_layout(**layout_config)
    
    # Update y-axes
    fig.update_yaxes(title_text="Pass Rate (%)", secondary_y=False, range=[0, 100])
    fig.update_yaxes(title_text="Average GPA", secondary_y=True, range=[0, 4])
    
    return fig, stats_df


# ============================================================================
# KPI CARD FUNCTIONS
# ============================================================================

def calculate_kpis(df, selected_period=None, selected_course=None):
    """Calculate KPIs based on filters"""
    
    # Apply filters
    filtered_df = df.copy()
    if selected_period:
        filtered_df = filtered_df[filtered_df['PERIOD'] == selected_period]
    if selected_course:
        filtered_df = filtered_df[filtered_df['Course_Code'] == selected_course]
    
    # Calculate metrics
    total_students = filtered_df['STUDENT ID'].nunique()
    at_risk_count = filtered_df[filtered_df['GPA'] < 2.5]['STUDENT ID'].nunique()
    at_risk_pct = (at_risk_count / total_students * 100) if total_students > 0 else 0
    avg_gpa = filtered_df['GPA'].mean()
    
    # Trend calculation (compare to previous period if applicable)
    if selected_period and selected_period != 'Sem 1':
        prev_period = f'Sem {int(selected_period.split()[1]) - 1}'
        prev_gpa = df[df['PERIOD'] == prev_period]['GPA'].mean()
        gpa_trend = 'up' if avg_gpa > prev_gpa else 'down'
        gpa_change = abs(avg_gpa - prev_gpa)
    else:
        gpa_trend = 'neutral'
        gpa_change = 0
    
    pass_rate = (filtered_df['Pass_Status'] == 'Pass').sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
    
    return {
        'total_students': total_students,
        'at_risk_count': at_risk_count,
        'at_risk_pct': at_risk_pct,
        'avg_gpa': avg_gpa,
        'gpa_trend': gpa_trend,
        'gpa_change': gpa_change,
        'pass_rate': pass_rate
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

# Initialize the Dash app with Bootstrap theme
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
                    html.Span("📊 ", style={'marginRight': '10px'}),
                    "Student Risk & Performance Monitor"
                ], style={
                    'color': COLORS['text_primary'],
                    'fontWeight': '700',
                    'marginBottom': '0.5rem'
                }),
                html.P("Real-time analytics dashboard for identifying at-risk students and course intervention needs", 
                       style={'color': COLORS['text_secondary'], 'fontSize': '0.95rem'})
            ], style={'padding': '1.5rem 0'})
        ])
    ]),
    
    # Dashboard Switcher + Global Filters
    dbc.Row([
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button(
                    "🎯 Thomas - Risk Monitor",
                    id='btn-thomas',
                    color='warning',
                    className='active',
                    style={
                        'backgroundColor': COLORS['primary'],
                        'borderColor': COLORS['primary'],
                        'color': COLORS['background'],
                        'fontWeight': '600'
                    }
                ),
                dbc.Button(
                    "🤝 Lingger - Support Systems",
                    id='btn-lingger',
                    color='secondary',
                    outline=True,
                    style={
                        'borderColor': COLORS['border'],
                        'color': COLORS['text_primary']
                    },
                    href='http://127.0.0.1:8051',
                    external_link=True
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
                clearable=False,
                style={
                    'backgroundColor': COLORS['surface'],
                    'color': COLORS['text_primary']
                }
            )
        ], width=3),
        
        dbc.Col([
            html.Label("Select Course", style={'color': COLORS['text_secondary'], 'fontSize': '0.875rem'}),
            dcc.Dropdown(
                id='course-filter',
                options=[{'label': 'All Courses', 'value': 'all'}] +
                        [{'label': f'Course {code}', 'value': code} for code in sorted([c for c in df['Course_Code'].unique() if pd.notna(c)])],
                value='all',
                clearable=False,
                style={
                    'backgroundColor': COLORS['surface'],
                    'color': COLORS['text_primary']
                }
            )
        ], width=3),
        
        dbc.Col([
            html.Label("Risk Level Filter", style={'color': COLORS['text_secondary'], 'fontSize': '0.875rem'}),
            dcc.Dropdown(
                id='risk-filter',
                options=[
                    {'label': 'All Risk Levels', 'value': 'all'},
                    {'label': '🔴 High Risk Only', 'value': 'High Risk'},
                    {'label': '🟡 Medium Risk Only', 'value': 'Medium Risk'},
                    {'label': '🟢 Low Risk Only', 'value': 'Low Risk'}
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
        dbc.Col(html.Div(id='kpi-at-risk'), width=3),
        dbc.Col(html.Div(id='kpi-avg-gpa'), width=3),
        dbc.Col(html.Div(id='kpi-pass-rate'), width=3)
    ], style={'marginBottom': '2rem'}),
    
    # Chart 1: Course Difficulty Bubble (Full Width)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='chart-course-difficulty', config={'displayModeBar': False})
                ])
            ], style={
                'backgroundColor': COLORS['card'],
                'border': f'1px solid {COLORS["border"]}',
                'borderRadius': '8px'
            })
        ], width=12)
    ], style={'marginBottom': '1.5rem'}),
    
    # Charts 2 & 3: GPA Trajectory + Risk Hotspot (Side by Side)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='chart-gpa-trajectory', config={'displayModeBar': False})
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
                    dcc.Graph(id='chart-risk-hotspot', config={'displayModeBar': False})
                ])
            ], style={
                'backgroundColor': COLORS['card'],
                'border': f'1px solid {COLORS["border"]}',
                'borderRadius': '8px'
            })
        ], width=6)
    ], style={'marginBottom': '1.5rem'}),
    
    # Chart 4: Attendance Threshold (Full Width)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='chart-attendance-threshold', config={'displayModeBar': False}),
                    html.Div(id='attendance-insights', style={'marginTop': '1rem'})
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
                html.Span("Thomas's Dashboard", style={'color': COLORS['primary'], 'fontWeight': '600'}),
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
     Output('kpi-at-risk', 'children'),
     Output('kpi-avg-gpa', 'children'),
     Output('kpi-pass-rate', 'children'),
     Output('chart-course-difficulty', 'figure'),
     Output('chart-gpa-trajectory', 'figure'),
     Output('chart-risk-hotspot', 'figure'),
     Output('chart-attendance-threshold', 'figure'),
     Output('attendance-insights', 'children')],
    [Input('semester-filter', 'value'),
     Input('course-filter', 'value'),
     Input('risk-filter', 'value'),
     Input('global-attendance-slider', 'value')]
)
def update_dashboard(semester, course, risk_level, attendance_threshold):
    """Main callback to update all dashboard components"""
    
    # Filter data
    filtered_df = df.copy()
    
    if semester != 'all':
        filtered_df = filtered_df[filtered_df['PERIOD'] == semester]
    
    if course != 'all':
        filtered_df = filtered_df[filtered_df['Course_Code'] == course]
    
    if risk_level != 'all':
        filtered_df = filtered_df[filtered_df['Initial_Risk'] == risk_level]
    
    # Calculate KPIs
    kpis = calculate_kpis(filtered_df, semester if semester != 'all' else None, 
                         course if course != 'all' else None)
    
    # Create KPI cards
    kpi1 = create_kpi_card(
        "Total Students",
        f"{kpis['total_students']:,}",
        "Unique students in dataset",
        "👥",
        'info'
    )
    
    kpi2 = create_kpi_card(
        "At-Risk Students",
        f"{kpis['at_risk_count']:,}",
        f"{kpis['at_risk_pct']:.1f}% of total (GPA < 2.5)",
        "🔴",
        'danger'
    )
    
    trend_icon = '📈' if kpis['gpa_trend'] == 'up' else ('📉' if kpis['gpa_trend'] == 'down' else '➡️')
    trend_text = f"{trend_icon} {'+' if kpis['gpa_trend'] == 'up' else ''}{kpis['gpa_change']:.2f}" if kpis['gpa_trend'] != 'neutral' else "Current semester"
    
    kpi3 = create_kpi_card(
        "Average GPA",
        f"{kpis['avg_gpa']:.2f}",
        trend_text,
        "📊",
        'success' if kpis['avg_gpa'] >= 3.0 else 'warning'
    )
    
    kpi4 = create_kpi_card(
        "Pass Rate",
        f"{kpis['pass_rate']:.1f}%",
        "Students with GPA ≥ 2.0",
        "✅",
        'success' if kpis['pass_rate'] >= 80 else 'danger'
    )
    
    # Generate charts
    chart1 = create_course_difficulty_bubble(
        filtered_df, 
        selected_courses=[course] if course != 'all' else None
    )
    
    chart2 = create_gpa_trajectory_with_dropdown(filtered_df)
    
    chart3 = create_risk_hotspot_heatmap(filtered_df)
    
    chart4, threshold_stats = create_attendance_threshold_slider(filtered_df, attendance_threshold)
    
    # Create attendance insights
    current_stats = threshold_stats[threshold_stats['Band'] == attendance_threshold].iloc[0]
    insights = dbc.Alert([
        html.H6("💡 Insights at Current Threshold:", style={'marginBottom': '0.5rem'}),
        html.Ul([
            html.Li(f"{current_stats['Student_Count']:,} students meet {attendance_threshold}% attendance requirement"),
            html.Li(f"{current_stats['Pass_Rate']:.1f}% pass rate among students at this threshold"),
            html.Li(f"Average GPA of {current_stats['Avg_GPA']:.2f} for students meeting requirement"),
            html.Li(f"{current_stats['At_Risk']:,} at-risk students (GPA < 2.5) in this group")
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
    app.run(debug=True, port=8050)