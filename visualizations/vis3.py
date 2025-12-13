"""
VISUALIZATION 3: Numerical Variable Distribution by Heart Disease Status
Shows histogram and boxplot for numerical variables split by Heart Disease
"""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_vis3(df, column='BMI'):
    """
    Create side-by-side histogram and boxplot showing distribution
    of a numerical variable by Heart Disease status.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The cardiovascular disease dataset
    column : str
        The numerical column to visualize
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    trg = 'Heart_Disease'
    color = 'Set2'
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Histogram with KDE
    sns.histplot(
        data=df, 
        x=column, 
        hue=trg, 
        bins=50, 
        kde=True, 
        palette=color, 
        ax=axes[0]
    )
    axes[0].set_title(f'{column} Distribution by {trg}')
    axes[0].grid(False)
    
    # Horizontal boxplot
    sns.boxplot(
        data=df, 
        x=column, 
        y=trg,
        hue=trg, 
        palette=color,
        ax=axes[1], 
        orient='h'
    )
    axes[1].set_title(f'{column} Boxplot by {trg}')
    
    plt.tight_layout()
    
    return fig
