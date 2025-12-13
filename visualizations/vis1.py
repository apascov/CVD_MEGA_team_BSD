"""
VISUALIZATION 1: Categorical Variable Distribution by Heart Disease Status
Shows count plots for categorical variables split by Heart Disease presence
"""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_vis1(df, column='General_Health'):
    """
    Create a count plot showing distribution of a categorical variable
    split by Heart Disease status with percentage annotations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The cardiovascular disease dataset
    column : str
        The categorical column to visualize
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    trg = 'Heart_Disease'
    
    fig, ax = plt.subplots(figsize=(16, 8))
    color = 'Set2'
    
    # Create count plot
    sns.countplot(
        x=column, 
        data=df, 
        hue=trg, 
        palette=color, 
        order=df[column].value_counts().index, 
        ax=ax
    )
    
    ax.set_ylabel('Count')
    ax.set_title(f'{column} Distribution by Heart Disease Status')
    offset = df[column].value_counts().max() * 0.005
    
    # Add percentage annotations
    bars = ax.patches
    bars_pos = 0
    counts = df.groupby([column, trg])[column].agg('count').unstack().fillna(0).values
    
    for i in range(df[trg].nunique()):
        for j in range(df[column].nunique()):
            vals = counts[j]
            total = vals.sum()
            count = vals[i]
            pct = count / total if total != 0 else 0
            if pct != 0:
                x = bars[bars_pos].get_x() + bars[bars_pos].get_width() / 2
                y = bars[bars_pos].get_height() + offset
                ax.annotate(f'{pct*100:.1f}%', (x, y), ha='center', fontsize=9)
                bars_pos += 1
    
    plt.tight_layout()
    return fig
