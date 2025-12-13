"""
VISUALIZATION 2: Heart Disease Rate by Two Categorical Variables
Shows heart disease rate as a grouped bar chart for two categorical variables
"""
import matplotlib.pyplot as plt
import pandas as pd


def plot_vis2(df, choice1='Age_Category', choice2='Sex'):
    """
    Create a grouped bar plot showing heart disease rate across
    two categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The cardiovascular disease dataset
    choice1 : str
        First categorical variable for grouping
    choice2 : str
        Second categorical variable for grouping
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Ensure Heart_Disease is numeric (convert Yes/No to 1/0 if needed)
    df_copy = df.copy()
    if df_copy['Heart_Disease'].dtype == 'object':
        df_copy['Heart_Disease'] = (df_copy['Heart_Disease'] == 'Yes').astype(int)
    
    # Calculate heart disease rate
    pivot = df_copy.groupby([choice1, choice2])['Heart_Disease'].mean().unstack()
    
    # Create grouped bar plot
    pivot.plot(kind='bar', ax=ax, colormap='Set2')
    
    ax.set_ylabel('Heart Disease Rate')
    ax.set_title(f'Heart Disease Rate by {choice1} and {choice2}')
    ax.set_xlabel(choice1)
    ax.legend(title=choice2)
    ax.set_ylim(0, max(0.5, pivot.max().max() * 1.1))
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig
