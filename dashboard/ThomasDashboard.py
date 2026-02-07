"""
THOMAS'S DASHBOARD: Student Risk & Performance Monitor (UPGRADED VERSION)
CA2 Data Visualization Assignment - ST1502
Theme: Dark professional with yellow/orange accents

STRATEGIC UPGRADES IMPLEMENTED:
1. Cross-filtering: Click Chart 1 to filter all other charts
2. Smart Trajectory: Auto-switches between line chart (diploma) and gauge (certificate)
3. Multi-metric Heatmap: Toggle between 3 different metrics with radio buttons
4. KPI Sparklines: Mini trend charts inside KPI cards
5. Enhanced interactivity throughout
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

COLORS = {
    'background': '#0a1929',
    'surface': '#132f4c',
    'card': '#1e3a5f',
    'primary': '#fbbf24',
    'secondary': '#f59e0b',
    'success': '#10b981',
    'danger': '#ef4444',
    'warning': '#f59e0b',
    'info': '#3b82f6',
    'text_primary': '#f1f5f9',
    'text_secondary': '#94a3b8',
    'border': '#334155',
    'grid': '#1e293b'
}

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
# DATA LOADING
# ============================================================================

def load_and_prepare_data():
    """Load and prepare the master dataset"""
    
    df = pd.read_csv('../cleaned_data/master_dataset.csv')
    df = df.dropna(subset=['PERIOD', 'GPA', 'STUDENT ID'])
    
    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
    df['COMMENCEMENT DATE'] = pd.to_datetime(df['COMMENCEMENT DATE'], errors='coerce')
    df['COMPLETION DATE'] = pd.to_datetime(df['COMPLETION DATE'], errors='coerce')
    
    df['Age_Group'] = pd.cut(df['AGE'], bins=[0, 25, 35, 45, 100], 
                              labels=['18-25', '26-35', '36-45', '46+'])
    
    df['Initial_Risk'] = pd.cut(df['GPA'].where(df['PERIOD'] == 'Sem 1'),
                                 bins=[0, 2.5, 3.0, 4.0],
                                 labels=['High Risk', 'Medium Risk', 'Low Risk'])
    df['Initial_Risk'] = df.groupby('STUDENT ID')['Initial_Risk'].transform('first')
    
    df['Pass_Status'] = df['GPA'].apply(lambda x: 'Pass' if x >= 2.0 else 'Fail')
    df['ATTENDANCE'] = df['ATTENDANCE'].fillna(0)
    df['SELF-STUDY HRS'] = df['SELF-STUDY HRS'].fillna(0)
    
    df['Period_Clean'] = df['PERIOD'].str.replace('Sem ', 'Semester ')
    df['Course_Code'] = df['STUDENT ID'].str.extract(r'^(\d{4})-')[0]
    
    # Determine course type
    df['Course_Type'] = df['Course_Code'].apply(
        lambda x: 'Certificate' if int(x) < 2000 else ('Diploma' if int(x) < 3000 else 'Specialist')
    )
    
    return df

# ============================================================================
# UPGRADED CHART 1: COURSE DIFFICULTY BUBBLE with CROSS-FILTERING
# ============================================================================

def create_interactive_course_bubble(df, selected_courses=None):
    """
    UPGRADED: Interactive bubble chart showing course difficulty
    Click any bubble to filter the entire dashboard by that course
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
                     text='Course_Code',
                     hover_data={
                         'Avg_GPA': ':.2f',
                         'Failure_Rate': ':.1f',
                         'Enrollment': ':,',
                         'Course_Code': False
                     },
                     size_max=60,
                     color_continuous_scale=[
                         [0, COLORS['success']],
                         [0.3, COLORS['primary']],
                         [0.6, COLORS['warning']],
                         [1, COLORS['danger']]
                     ],
                     range_color=[0, 50])
    
    # Update text position and styling
    fig.update_traces(
        textposition='middle center',
        textfont=dict(size=10, color=COLORS['text_primary'], family='monospace'),
        marker=dict(
            line=dict(width=2, color=COLORS['border']),
            opacity=0.8
        )
    )
    
    layout_config = CHART_TEMPLATE['layout'].copy()
    layout_config.update({
        'title': {
            'text': '<b>📊 Course Difficulty Matrix</b><br><sub>Click any bubble to filter dashboard | Size=Enrollment | Color=Failure Rate</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': COLORS['text_primary']}
        },
        'xaxis_title': 'Average GPA',
        'yaxis_title': 'Failure Rate (%)',
        'showlegend': False,
        'height': 450
    })
    
    fig.update_layout(**layout_config)
    
    # Add quadrant lines for interpretation
    fig.add_hline(y=25, line_dash="dash", line_color=COLORS['border'], 
                  opacity=0.5, line_width=1)
    fig.add_vline(x=3.0, line_dash="dash", line_color=COLORS['border'], 
                  opacity=0.5, line_width=1)
    
    # Add quadrant annotations
    fig.add_annotation(x=3.5, y=45, text="✅ Low Risk<br>High GPA, Low Failure", 
                      showarrow=False, font=dict(color=COLORS['success'], size=11),
                      bgcolor=COLORS['surface'], bordercolor=COLORS['success'], 
                      borderwidth=1, borderpad=4, opacity=0.8)
    
    fig.add_annotation(x=2.5, y=45, text="🔴 HIGH RISK<br>Low GPA, High Failure", 
                      showarrow=False, font=dict(color=COLORS['danger'], size=11, weight='bold'),
                      bgcolor=COLORS['surface'], bordercolor=COLORS['danger'], 
                      borderwidth=2, borderpad=4, opacity=0.9)
    
    fig.add_annotation(x=3.5, y=5, text="⚠️ Moderate<br>Good GPA, Some Failures", 
                      showarrow=False, font=dict(color=COLORS['warning'], size=10),
                      bgcolor=COLORS['surface'], bordercolor=COLORS['warning'], 
                      borderwidth=1, borderpad=4, opacity=0.8)
    
    fig.add_annotation(x=2.5, y=5, text="📉 Needs Support<br>Low GPA, Variable Outcomes", 
                      showarrow=False, font=dict(color=COLORS['info'], size=10),
                      bgcolor=COLORS['surface'], bordercolor=COLORS['info'], 
                      borderwidth=1, borderpad=4, opacity=0.8)
    
    # Make bubbles clickable
    fig.update_traces(
        customdata=course_stats[['Course_Code']],
        hovertemplate='<b>Course: %{customdata[0]}</b><br>' +
                      'Avg GPA: %{x:.2f}<br>' +
                      'Failure Rate: %{y:.1f}%<br>' +
                      'Enrollment: %{marker.size} students<br>' +
                      '<i>Click to filter dashboard</i><extra></extra>'
    )
    
    return fig

# ============================================================================
# UPGRADED CHART 2: SMART TRAJECTORY (LINE or GAUGE)
# ============================================================================

def create_smart_trajectory(df, view_mode='risk_level', selected_filter=None):
    """
    UPGRADED: Intelligently switches between LINE CHART (diploma) and GAUGE (certificate)
    Handles edge cases gracefully
    """
    
    # Apply cross-filter if exists
    if selected_filter:
        df = df[df['Course_Code'].isin(selected_filter)]
    
    # Detect if filtered data contains only certificates (1 semester courses)
    semester_counts = df.groupby('STUDENT ID')['PERIOD'].nunique()
    
    # If average semesters < 2, it's mostly certificates → use GAUGE
    if semester_counts.mean() < 1.5:
        return create_gauge_chart(df)
    else:
        return create_line_trajectory(df, view_mode)


def create_line_trajectory(df, view_mode):
    """Traditional line chart for multi-semester courses"""
    
    # Filter to students with at least 2 semesters
    retained_students = df.groupby('STUDENT ID')['PERIOD'].nunique()
    retained_students = retained_students[retained_students >= 2].index
    df_retained = df[df['STUDENT ID'].isin(retained_students)].copy()
    
    # IMPORTANT: Remove Semester 4 (only 1 student) to avoid misleading data
    df_retained = df_retained[df_retained['PERIOD'] != 'Sem 4']
    
    # Also filter out age groups with very few students (< 5)
    age_counts = df_retained.groupby('Age_Group').size()
    valid_ages = age_counts[age_counts >= 5].index
    df_retained = df_retained[df_retained['Age_Group'].isin(valid_ages)]
    
    fig = go.Figure()
    traces = {}
    
    # By Risk Level
    for risk in ['High Risk', 'Medium Risk', 'Low Risk']:
        risk_data = df_retained[df_retained['Initial_Risk'] == risk]
        if len(risk_data) == 0:
            continue
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
    
    # By Age Group (only include groups with sufficient data)
    for age_grp in valid_ages:
        age_data = df_retained[df_retained['Age_Group'] == age_grp]
        if len(age_data) == 0:
            continue
        trajectory = age_data.groupby('Period_Clean')['GPA'].mean().reset_index()
        
        color_map = {'18-25': '#3b82f6', '26-35': '#8b5cf6', 
                     '36-45': '#ec4899', '46+': '#f97316'}
        
        traces[f'age_{age_grp}'] = go.Scatter(
            x=trajectory['Period_Clean'],
            y=trajectory['GPA'],
            mode='lines+markers',
            name=f'Age {age_grp}',
            line=dict(color=color_map.get(age_grp, '#3b82f6'), width=3),
            marker=dict(size=10),
            visible=(view_mode == 'age_group'),
            hovertemplate='<b>%{x}</b><br>Avg GPA: %{y:.2f}<extra></extra>'
        )
    
    # All Students
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
    
    for trace in traces.values():
        fig.add_trace(trace)
    
    dropdown_buttons = [
        dict(label='📊 By Initial Risk Level',
             method='update',
             args=[{'visible': [k.startswith('risk_') for k in traces.keys()]},
                   {'title': '<b>GPA Trajectory by Risk Level</b><br><sub>Students who completed 2-3 semesters (Sem 4 excluded)</sub>'}]),
        dict(label='👥 By Age Group',
             method='update',
             args=[{'visible': [k.startswith('age_') for k in traces.keys()]},
                   {'title': '<b>GPA Trajectory by Age</b><br><sub>Age groups with sufficient data only</sub>'}]),
        dict(label='🌐 All Students',
             method='update',
             args=[{'visible': [k == 'all_students' for k in traces.keys()]},
                   {'title': '<b>Overall GPA Trajectory</b><br><sub>Average across all students (Sem 1-3)</sub>'}])
    ]
    
    layout_config = CHART_TEMPLATE['layout'].copy()
    layout_config.update({
        'title': '<b>GPA Trajectory by Risk Level</b><br><sub>Students who completed 2-3 semesters (Sem 4 excluded)</sub>',
        'xaxis_title': 'Semester',
        'yaxis_title': 'Average GPA',
        'yaxis_range': [1.5, 4.0],
        'hovermode': 'x unified',
        'height': 450,  # Increased height
        'updatemenus': [{
            'buttons': dropdown_buttons,
            'direction': "down",
            'pad': {"r": 10, "t": 10},
            'showactive': True,
            'x': 0.02,
            'xanchor': "left",
            'y': 1.22,  # Moved higher to avoid title overlap
            'yanchor': "top",
            'bgcolor': COLORS['card'],
            'bordercolor': COLORS['primary'],
            'borderwidth': 2,
            'font': dict(color=COLORS['text_primary'], size=11)
        }],
        'margin': {'l': 60, 'r': 40, 't': 100, 'b': 60}  # Increased top margin
    })
    
    fig.update_layout(**layout_config)
    fig.add_hline(y=2.0, line_dash="dash", line_color=COLORS['danger'],
                  annotation_text="Passing Threshold (2.0)", annotation_position="right")
    
    return fig


def create_gauge_chart(df):
    """GAUGE chart for single-semester (Certificate) courses"""
    
    avg_gpa = df['GPA'].mean()
    pass_rate = (df['Pass_Status'] == 'Pass').sum() / len(df) * 100
    
    fig = go.Figure()
    
    # Main GPA Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=avg_gpa,
        domain={'x': [0, 0.48], 'y': [0.2, 0.8]},
        title={'text': "<b>Average GPA</b>", 'font': {'size': 16, 'color': COLORS['text_primary']}},
        delta={'reference': 2.5, 'increasing': {'color': COLORS['success']}, 'decreasing': {'color': COLORS['danger']}},
        number={'font': {'size': 40, 'color': COLORS['primary']}},
        gauge={
            'axis': {'range': [0, 4], 'tickwidth': 2, 'tickcolor': COLORS['text_secondary']},
            'bar': {'color': COLORS['primary']},
            'bgcolor': COLORS['surface'],
            'borderwidth': 2,
            'bordercolor': COLORS['border'],
            'steps': [
                {'range': [0, 2.0], 'color': COLORS['danger']},
                {'range': [2.0, 2.5], 'color': COLORS['warning']},
                {'range': [2.5, 4.0], 'color': COLORS['success']}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': avg_gpa
            }
        }
    ))
    
    # Pass Rate Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=pass_rate,
        domain={'x': [0.52, 1], 'y': [0.2, 0.8]},
        title={'text': "<b>Pass Rate</b>", 'font': {'size': 16, 'color': COLORS['text_primary']}},
        number={'suffix': "%", 'font': {'size': 40, 'color': COLORS['success']}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': COLORS['text_secondary']},
            'bar': {'color': COLORS['success']},
            'bgcolor': COLORS['surface'],
            'borderwidth': 2,
            'bordercolor': COLORS['border'],
            'steps': [
                {'range': [0, 60], 'color': COLORS['danger']},
                {'range': [60, 80], 'color': COLORS['warning']},
                {'range': [80, 100], 'color': COLORS['success']}
            ]
        }
    ))
    
    layout_config = CHART_TEMPLATE['layout'].copy()
    layout_config.update({
        'title': {
            'text': '<b>📊 Certificate Course Performance Snapshot</b><br><sub>Single semester - showing gauges instead of trajectory</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': COLORS['text_primary']}
        },
        'height': 400
    })
    
    fig.update_layout(**layout_config)
    
    return fig

# ============================================================================
# UPGRADED CHART 3: MULTI-METRIC HEATMAP with RADIO BUTTONS
# ============================================================================

def create_multi_metric_heatmap(df, metric='failure_rate', gpa_threshold=2.0, selected_filter=None):
    """
    UPGRADED: Toggle between 3 metrics with DYNAMIC GPA THRESHOLD
    Now accepts a gpa_threshold parameter for flexible "failure" definition
    """
    
    if selected_filter:
        df = df[df['Course_Code'].isin(selected_filter)]
    
    # Filter out age groups with very few students
    age_counts = df.groupby('Age_Group').size()
    valid_ages = age_counts[age_counts >= 5].index
    df = df[df['Age_Group'].isin(valid_ages)]
    
    fig = go.Figure()
    metrics_data = {}
    
    # Metric 1: Failure Rate % (now uses dynamic threshold)
    failure_pivot = df.groupby(['Age_Group', 'Course_Code']).apply(
        lambda x: (x['GPA'] < gpa_threshold).sum() / len(x) * 100 if len(x) > 0 else 0
    ).reset_index()
    failure_pivot.columns = ['Age_Group', 'Course_Code', 'Value']
    failure_matrix = failure_pivot.pivot(index='Age_Group', columns='Course_Code', values='Value').fillna(0)
    
    hover_text_1 = []
    for i, age in enumerate(failure_matrix.index):
        row = []
        for j, course in enumerate(failure_matrix.columns):
            value = failure_matrix.iloc[i, j]
            count = len(df[(df['Age_Group'] == age) & (df['Course_Code'] == course)])
            text = f"<b>Age: {age}</b><br>Course: {course}<br>Below {gpa_threshold} GPA: {value:.1f}%<br>Students: {count}"
            row.append(text)
        hover_text_1.append(row)
    
    metrics_data['failure_rate'] = {
        'z': failure_matrix.values,
        'x': failure_matrix.columns.tolist(),
        'y': failure_matrix.index.tolist(),
        'colorscale': [[0, COLORS['success']], [0.3, COLORS['warning']], [1, COLORS['danger']]],
        'text': hover_text_1,
        'title': f'<b>🔴 Risk Hotspot: % Below {gpa_threshold} GPA</b><br><sub>Percentage of students below threshold by age & course</sub>',
        'colorbar_title': f'% < {gpa_threshold}'
    }
    
    # Metric 2: Average GPA
    gpa_pivot = df.groupby(['Age_Group', 'Course_Code'])['GPA'].mean().reset_index()
    gpa_pivot.columns = ['Age_Group', 'Course_Code', 'Value']
    gpa_matrix = gpa_pivot.pivot(index='Age_Group', columns='Course_Code', values='Value').fillna(0)
    
    hover_text_2 = []
    for i, age in enumerate(gpa_matrix.index):
        row = []
        for j, course in enumerate(gpa_matrix.columns):
            value = gpa_matrix.iloc[i, j]
            count = len(df[(df['Age_Group'] == age) & (df['Course_Code'] == course)])
            text = f"<b>Age: {age}</b><br>Course: {course}<br>Avg GPA: {value:.2f}<br>Students: {count}"
            row.append(text)
        hover_text_2.append(row)
    
    metrics_data['avg_gpa'] = {
        'z': gpa_matrix.values,
        'x': gpa_matrix.columns.tolist(),
        'y': gpa_matrix.index.tolist(),
        'colorscale': [[0, COLORS['danger']], [0.5, COLORS['warning']], [1, COLORS['success']]],
        'text': hover_text_2,
        'title': '<b>📊 Performance Heatmap: Average GPA</b><br><sub>Performance levels by age and course</sub>',
        'colorbar_title': 'Avg GPA'
    }
    
    # Metric 3: Average Attendance Rate
    attendance_pivot = df.groupby(['Age_Group', 'Course_Code'])['ATTENDANCE'].mean().reset_index()
    attendance_pivot.columns = ['Age_Group', 'Course_Code', 'Value']
    attendance_matrix = attendance_pivot.pivot(index='Age_Group', columns='Course_Code', values='Value').fillna(0)
    
    hover_text_3 = []
    for i, age in enumerate(attendance_matrix.index):
        row = []
        for j, course in enumerate(attendance_matrix.columns):
            value = attendance_matrix.iloc[i, j]
            count = len(df[(df['Age_Group'] == age) & (df['Course_Code'] == course)])
            text = f"<b>Age: {age}</b><br>Course: {course}<br>Avg Attendance: {value:.1f}%<br>Students: {count}"
            row.append(text)
        hover_text_3.append(row)
    
    metrics_data['attendance'] = {
        'z': attendance_matrix.values,
        'x': attendance_matrix.columns.tolist(),
        'y': attendance_matrix.index.tolist(),
        'colorscale': [[0, COLORS['danger']], [0.5, COLORS['warning']], [1, COLORS['success']]],
        'text': hover_text_3,
        'title': '<b>📅 Discipline Heatmap: Attendance Rates</b><br><sub>Average attendance by age and course</sub>',
        'colorbar_title': 'Attendance %'
    }
    
    # Create heatmap traces
    for metric_key, metric_info in metrics_data.items():
        fig.add_trace(go.Heatmap(
            z=metric_info['z'],
            x=metric_info['x'],
            y=metric_info['y'],
            colorscale=metric_info['colorscale'],
            text=metric_info['text'],
            hovertemplate='%{text}<extra></extra>',
            showscale=True,
            colorbar=dict(
                title=metric_info['colorbar_title'],
                tickfont=dict(color=COLORS['text_secondary']),
                outlinecolor=COLORS['border']
            ),
            visible=(metric == metric_key)
        ))
    
    # Radio buttons
    radio_buttons = [
        dict(label=f'🔴 Below {gpa_threshold} GPA %',
             method='update',
             args=[{'visible': [True, False, False]},
                   {'title': metrics_data['failure_rate']['title']}]),
        dict(label='📊 Average GPA',
             method='update',
             args=[{'visible': [False, True, False]},
                   {'title': metrics_data['avg_gpa']['title']}]),
        dict(label='📅 Attendance Rate',
             method='update',
             args=[{'visible': [False, False, True]},
                   {'title': metrics_data['attendance']['title']}])
    ]
    
    layout_config = CHART_TEMPLATE['layout'].copy()
    layout_config.update({
        'title': metrics_data[metric]['title'],
        'xaxis_title': 'Course Code',
        'yaxis_title': 'Age Group',
        'height': 500,  # Increased height
        'updatemenus': [{
            'buttons': radio_buttons,
            'direction': "down",
            'pad': {"r": 10, "t": 10},
            'showactive': True,
            'x': 0.02,
            'xanchor': "left",
            'y': 1.18,  # Moved higher to avoid overlap
            'yanchor': "top",
            'bgcolor': COLORS['card'],
            'bordercolor': COLORS['primary'],
            'borderwidth': 2,
            'font': dict(color=COLORS['text_primary'], size=11)
        }],
        'margin': {'l': 60, 'r': 40, 't': 100, 'b': 60}  # Increased top margin
    })
    
    fig.update_layout(**layout_config)
    
    return fig

# ============================================================================
# CHART 4: ATTENDANCE THRESHOLD (Enhanced)
# ============================================================================

def create_attendance_threshold_slider(df, threshold=75, selected_filter=None):
    """Enhanced attendance threshold analysis with proper spacing"""
    
    if selected_filter:
        df = df[df['Course_Code'].isin(selected_filter)]
    
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
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=stats_df['Threshold'],
            y=stats_df['Pass_Rate'],
            name='Pass Rate %',
            marker_color=COLORS['primary'],
            text=stats_df['Pass_Rate'].round(1),
            texttemplate='%{text}%',
            textposition='outside',
            textfont=dict(size=10),  # Smaller text to prevent overlap
            hovertemplate='<b>Attendance: %{x}</b><br>Pass Rate: %{y:.1f}%<extra></extra>'
        ),
        secondary_y=False
    )
    
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
    
    threshold_idx = (threshold - 50) // 5
    if threshold_idx < len(stats_df):
        fig.add_vline(
            x=threshold_idx,
            line_dash="dash",
            line_color=COLORS['info'],
            line_width=2,
            annotation_text=f"Current: {threshold}%",
            annotation_position="top"
        )
    
    layout_config = CHART_TEMPLATE['layout'].copy()
    layout_config.update({
        'title': '<b>Attendance Threshold Impact</b><br><sub>Pass rate & GPA by minimum attendance</sub>',
        'xaxis_title': 'Minimum Attendance Requirement',
        'height': 500,  # Significantly increased height
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
        ),
        'margin': {'l': 60, 'r': 40, 't': 80, 'b': 80}  # More breathing room
    })
    
    fig.update_layout(**layout_config)
    fig.update_yaxes(title_text="Pass Rate (%)", secondary_y=False, range=[0, 110])  # Extended range for text labels
    fig.update_yaxes(title_text="Average GPA", secondary_y=True, range=[0, 4])
    
    return fig, stats_df

# ============================================================================
# UPGRADED KPI CARDS with SPARKLINES
# ============================================================================

def create_kpi_card_with_sparkline(title, value, subtitle, icon, sparkline_data=None, color='primary'):
    """
    UPGRADED KPI Card with mini sparkline trend chart
    """
    
    color_map = {
        'primary': COLORS['primary'],
        'danger': COLORS['danger'],
        'success': COLORS['success'],
        'info': COLORS['info']
    }
    
    # Create sparkline if data provided
    sparkline_fig = None
    if sparkline_data is not None and len(sparkline_data) > 1:
        sparkline_fig = go.Figure()
        sparkline_fig.add_trace(go.Scatter(
            y=sparkline_data,
            mode='lines',
            line=dict(color=color_map[color], width=2),
            fill='tozeroy',
            fillcolor=f'rgba({int(color_map[color][1:3], 16)}, {int(color_map[color][3:5], 16)}, {int(color_map[color][5:7], 16)}, 0.2)',
            hovertemplate='Value: %{y:.1f}<extra></extra>'
        ))
        sparkline_fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            height=60,
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        sparkline_fig.update_xaxes(showgrid=False, zeroline=False)
        sparkline_fig.update_yaxes(showgrid=False, zeroline=False)
    
    card_content = [
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
    ]
    
    # Add sparkline if available
    if sparkline_fig:
        card_content.append(
            dcc.Graph(
                figure=sparkline_fig,
                config={'displayModeBar': False},
                style={'marginTop': '0.5rem'}
            )
        )
    
    return dbc.Card([
        dbc.CardBody(card_content)
    ], style={
        'backgroundColor': COLORS['card'],
        'border': f'1px solid {COLORS["border"]}',
        'borderRadius': '8px',
        'marginBottom': '1rem'
    })

# ============================================================================
# KPI CALCULATION
# ============================================================================

def calculate_kpis(df, selected_period=None, selected_course=None, selected_filter=None):
    """Calculate KPIs with historical data for sparklines"""
    
    filtered_df = df.copy()
    
    if selected_filter:
        filtered_df = filtered_df[filtered_df['Course_Code'].isin(selected_filter)]
    
    if selected_period and selected_period != 'all':
        filtered_df = filtered_df[filtered_df['PERIOD'] == selected_period]
    
    if selected_course and selected_course != 'all':
        filtered_df = filtered_df[filtered_df['Course_Code'] == selected_course]
    
    total_students = filtered_df['STUDENT ID'].nunique()
    at_risk_count = filtered_df[filtered_df['GPA'] < 2.5]['STUDENT ID'].nunique()
    at_risk_pct = (at_risk_count / total_students * 100) if total_students > 0 else 0
    avg_gpa = filtered_df['GPA'].mean()
    pass_rate = (filtered_df['Pass_Status'] == 'Pass').sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
    
    # Calculate sparkline data (GPA trend across semesters)
    gpa_sparkline = df.groupby('PERIOD')['GPA'].mean().values.tolist()
    
    # Calculate trend
    if selected_period and selected_period != 'all' and selected_period != 'Sem 1':
        prev_period = f'Sem {int(selected_period.split()[1]) - 1}'
        prev_gpa = df[df['PERIOD'] == prev_period]['GPA'].mean()
        gpa_trend = 'up' if avg_gpa > prev_gpa else 'down'
        gpa_change = abs(avg_gpa - prev_gpa)
    else:
        gpa_trend = 'neutral'
        gpa_change = 0
    
    return {
        'total_students': total_students,
        'at_risk_count': at_risk_count,
        'at_risk_pct': at_risk_pct,
        'avg_gpa': avg_gpa,
        'gpa_trend': gpa_trend,
        'gpa_change': gpa_change,
        'pass_rate': pass_rate,
        'gpa_sparkline': gpa_sparkline
    }

# ============================================================================
# DASH APP
# ============================================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
df = load_and_prepare_data()

# ============================================================================
# LAYOUT
# ============================================================================

app.layout = dbc.Container([
    
    # Store for cross-filter state
    dcc.Store(id='selected-courses-store', data=[]),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H2([
                    html.Span("📊 ", style={'marginRight': '10px'}),
                    "Student Risk & Performance Monitor",
                    html.Span(" [UPGRADED]", style={'fontSize': '0.6em', 'color': COLORS['primary'], 'marginLeft': '10px'})
                ], style={'color': COLORS['text_primary'], 'fontWeight': '700', 'marginBottom': '0.5rem'}),
                html.P("Real-time analytics with cross-filtering, smart charts & enhanced interactivity", 
                       style={'color': COLORS['text_secondary'], 'fontSize': '0.95rem'})
            ], style={'padding': '1.5rem 0'})
        ])
    ]),
    
    # Dashboard Switcher
    dbc.Row([
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button("🎯 Thomas - Risk Monitor", id='btn-thomas', color='warning', className='active',
                          style={'backgroundColor': COLORS['primary'], 'borderColor': COLORS['primary'], 
                                'color': COLORS['background'], 'fontWeight': '600'}),
                dbc.Button("🤝 Lingger - Support Systems", id='btn-lingger', color='secondary', outline=True,
                          style={'borderColor': COLORS['border'], 'color': COLORS['text_primary']},
                          href='http://127.0.0.1:8051', external_link=True)
            ], style={'marginBottom': '1rem'})
        ], width=12)
    ]),
    
    # Global Filters
    dbc.Row([
        dbc.Col([
            html.Label("Select Semester", style={'color': COLORS['text_secondary'], 'fontSize': '0.875rem'}),
            dcc.Dropdown(id='semester-filter',
                        options=[{'label': 'All Semesters', 'value': 'all'}] + 
                                [{'label': period, 'value': period} for period in sorted([p for p in df['PERIOD'].unique() if pd.notna(p)])],
                        value='all', clearable=False)
        ], width=2),
        
        dbc.Col([
            html.Label("Select Course", style={'color': COLORS['text_secondary'], 'fontSize': '0.875rem'}),
            dcc.Dropdown(id='course-filter',
                        options=[{'label': 'All Courses', 'value': 'all'}] +
                                [{'label': f'Course {code}', 'value': code} for code in sorted([c for c in df['Course_Code'].unique() if pd.notna(c)])],
                        value='all', clearable=False)
        ], width=2),
        
        dbc.Col([
            html.Label("Risk Level Filter", style={'color': COLORS['text_secondary'], 'fontSize': '0.875rem'}),
            dcc.Dropdown(id='risk-filter',
                        options=[{'label': 'All Risk Levels', 'value': 'all'},
                                {'label': '🔴 High Risk Only', 'value': 'High Risk'},
                                {'label': '🟡 Medium Risk Only', 'value': 'Medium Risk'},
                                {'label': '🟢 Low Risk Only', 'value': 'Low Risk'}],
                        value='all', clearable=False)
        ], width=2),
        
        dbc.Col([
            html.Label("GPA Threshold", style={'color': COLORS['text_secondary'], 'fontSize': '0.875rem'}),
            dcc.Slider(id='gpa-threshold-slider', min=1.5, max=3.5, step=0.1, value=2.5,
                      marks={1.5: '1.5', 2.0: '2.0', 2.5: '2.5', 3.0: '3.0', 3.5: '3.5'},
                      tooltip={"placement": "bottom", "always_visible": True})
        ], width=3),
        
        dbc.Col([
            html.Label("Attendance Threshold", style={'color': COLORS['text_secondary'], 'fontSize': '0.875rem'}),
            dcc.Slider(id='global-attendance-slider', min=50, max=100, step=5, value=75,
                      marks={i: f'{i}%' for i in range(50, 101, 10)},
                      tooltip={"placement": "bottom", "always_visible": True})
        ], width=3)
    ], style={'marginBottom': '2rem'}),
    
    # Clear Filter Button
    dbc.Row([
        dbc.Col([
            dbc.Button("🔄 Clear Course Filter", id='clear-filter-btn', color='secondary', size='sm',
                      style={'backgroundColor': COLORS['card'], 'borderColor': COLORS['border'], 
                            'color': COLORS['text_primary']})
        ], width=12)
    ], style={'marginBottom': '1rem'}),
    
    # KPI Cards
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
                    dcc.Graph(id='chart-bubble', config={'displayModeBar': False})
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': f'1px solid {COLORS["border"]}', 'borderRadius': '8px'})
        ], width=12)
    ], style={'marginBottom': '1.5rem'}),
    
    # Charts 2 & 3 (Side by Side)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='chart-trajectory', config={'displayModeBar': False})
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': f'1px solid {COLORS["border"]}', 'borderRadius': '8px'})
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='chart-heatmap', config={'displayModeBar': False})
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': f'1px solid {COLORS["border"]}', 'borderRadius': '8px'})
        ], width=6)
    ], style={'marginBottom': '1.5rem'}),
    
    # Chart 4 (Full Width)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='chart-attendance', config={'displayModeBar': False}),
                    html.Div(id='attendance-insights', style={'marginTop': '1rem'})
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': f'1px solid {COLORS["border"]}', 'borderRadius': '8px'})
        ], width=12)
    ], style={'marginBottom': '2rem'}),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(style={'borderColor': COLORS['border']}),
            html.P([
                "CA2 Data Visualization Assignment • ST1502 • ",
                html.Span("Thomas's Dashboard [UPGRADED]", style={'color': COLORS['primary'], 'fontWeight': '600'}),
                " • AY2526 Sem 2 • ",
                html.Span("✨ With Cross-Filtering, Smart Charts & Sparklines", style={'color': COLORS['success'], 'fontSize': '0.8rem'})
            ], style={'textAlign': 'center', 'color': COLORS['text_secondary'], 'fontSize': '0.875rem', 'marginTop': '1rem'})
        ])
    ])
    
], fluid=True, style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'padding': '2rem'})

# ============================================================================
# CALLBACKS
# ============================================================================

# Callback for bubble chart click (cross-filtering)
@app.callback(
    Output('selected-courses-store', 'data'),
    Input('chart-bubble', 'clickData'),
    Input('clear-filter-btn', 'n_clicks'),
    prevent_initial_call=True
)
def update_cross_filter(clickData, clear_clicks):
    """Handle bubble chart clicks for cross-filtering"""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return []
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'clear-filter-btn':
        return []
    
    if clickData and 'points' in clickData:
        # Get the course code from customdata
        course_code = clickData['points'][0]['customdata'][0]
        return [course_code]
    
    return []


# Main dashboard update callback
@app.callback(
    [Output('kpi-total-students', 'children'),
     Output('kpi-at-risk', 'children'),
     Output('kpi-avg-gpa', 'children'),
     Output('kpi-pass-rate', 'children'),
     Output('chart-bubble', 'figure'),
     Output('chart-trajectory', 'figure'),
     Output('chart-heatmap', 'figure'),
     Output('chart-attendance', 'figure'),
     Output('attendance-insights', 'children')],
    [Input('semester-filter', 'value'),
     Input('course-filter', 'value'),
     Input('risk-filter', 'value'),
     Input('gpa-threshold-slider', 'value'),
     Input('global-attendance-slider', 'value'),
     Input('selected-courses-store', 'data')]
)
def update_dashboard(semester, course, risk_level, gpa_threshold, attendance_threshold, selected_courses):
    """Main callback with cross-filtering support and dynamic GPA threshold"""
    
    # Filter data
    filtered_df = df.copy()
    
    if semester != 'all':
        filtered_df = filtered_df[filtered_df['PERIOD'] == semester]
    
    if course != 'all':
        filtered_df = filtered_df[filtered_df['Course_Code'] == course]
    
    if risk_level != 'all':
        filtered_df = filtered_df[filtered_df['Initial_Risk'] == risk_level]
    
    # Calculate KPIs (using the dynamic GPA threshold)
    total_students = filtered_df['STUDENT ID'].nunique()
    at_risk_count = filtered_df[filtered_df['GPA'] < gpa_threshold]['STUDENT ID'].nunique()
    at_risk_pct = (at_risk_count / total_students * 100) if total_students > 0 else 0
    avg_gpa = filtered_df['GPA'].mean()
    pass_rate = (filtered_df['GPA'] >= 2.0).sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
    
    # Create KPI cards WITHOUT sparklines
    kpi1 = create_kpi_card_with_sparkline(
        "Total Students", f"{total_students:,}", "Unique students in dataset", "👥",
        sparkline_data=None, color='info'
    )
    
    kpi2 = create_kpi_card_with_sparkline(
        "At-Risk Students", f"{at_risk_count:,}",
        f"{at_risk_pct:.1f}% below {gpa_threshold} GPA threshold", "🔴",
        sparkline_data=None, color='danger'
    )
    
    kpi3 = create_kpi_card_with_sparkline(
        "Average GPA", f"{avg_gpa:.2f}", "Current selection", "📊",
        sparkline_data=None,
        color='success' if avg_gpa >= 3.0 else 'warning'
    )
    
    kpi4 = create_kpi_card_with_sparkline(
        "Pass Rate", f"{pass_rate:.1f}%", "Students with GPA ≥ 2.0", "✅",
        sparkline_data=None, color='success' if pass_rate >= 80 else 'danger'
    )
    
    # Generate charts
    chart1 = create_interactive_course_bubble(filtered_df, selected_courses if selected_courses else None)
    chart2 = create_smart_trajectory(filtered_df, selected_filter=selected_courses if selected_courses else None)
    chart3 = create_multi_metric_heatmap(filtered_df, gpa_threshold=gpa_threshold, 
                                         selected_filter=selected_courses if selected_courses else None)
    chart4, threshold_stats = create_attendance_threshold_slider(filtered_df, attendance_threshold,
                                                                  selected_filter=selected_courses if selected_courses else None)
    
    # Create attendance insights
    current_stats = threshold_stats[threshold_stats['Band'] == attendance_threshold].iloc[0]
    insights = dbc.Alert([
        html.H6("💡 Insights at Current Threshold:", style={'marginBottom': '0.5rem'}),
        html.Ul([
            html.Li(f"{current_stats['Student_Count']:,} students meet {attendance_threshold}% attendance requirement"),
            html.Li(f"{current_stats['Pass_Rate']:.1f}% pass rate among students at this threshold"),
            html.Li(f"Average GPA of {current_stats['Avg_GPA']:.2f} for students meeting requirement"),
            html.Li(f"{current_stats['At_Risk']:,} at-risk students (GPA < {gpa_threshold}) in this group")
        ], style={'marginBottom': 0})
    ], color='info', style={
        'backgroundColor': COLORS['surface'],
        'borderColor': COLORS['info'],
        'color': COLORS['text_primary']
    })
    
    return kpi1, kpi2, kpi3, kpi4, chart1, chart2, chart3, chart4, insights


if __name__ == '__main__':
    app.run(debug=True, port=8050)