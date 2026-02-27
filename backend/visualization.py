"""
Visualization utilities for generating charts and graphs.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import base64
from typing import Optional, Tuple, Any
import logging
from dataclasses import dataclass

from backend.config import settings

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


@dataclass
class ChartConfig:
    """Configuration for chart generation."""
    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 100
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    legend_fontsize: int = 10


class VisualizationEngine:
    """Engine for generating various visualizations."""
    
    def __init__(self, config: Optional[ChartConfig] = None):
        """Initialize the visualization engine."""
        self.config = config or ChartConfig()
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style settings."""
        plt.rcParams.update({
            'figure.figsize': self.config.figsize,
            'figure.dpi': self.config.dpi,
            'axes.titlesize': self.config.title_fontsize,
            'axes.labelsize': self.config.label_fontsize,
            'xtick.labelsize': self.config.tick_fontsize,
            'ytick.labelsize': self.config.tick_fontsize,
            'legend.fontsize': self.config.legend_fontsize,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'white'
        })
    
    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=self.config.dpi)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        return image_base64
    
    def create_histogram(
        self, 
        df: pd.DataFrame, 
        column: str, 
        title: Optional[str] = None,
        bins: int = 30,
        hue: Optional[str] = None,
        kde: bool = True
    ) -> Tuple[str, str]:
        """Create a histogram for a numeric column."""
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        if hue and hue in df.columns:
            for category in df[hue].dropna().unique():
                subset = df[df[hue] == category][column].dropna()
                ax.hist(subset, bins=bins, alpha=0.6, label=str(category), edgecolor='white')
            ax.legend(title=hue.capitalize())
        else:
            ax.hist(df[column].dropna(), bins=bins, color='steelblue', edgecolor='white', alpha=0.7)
            if kde:
                ax2 = ax.twinx()
                df[column].dropna().plot(kind='kde', ax=ax2, color='darkred', linewidth=2)
                ax2.set_ylabel('Density')
        
        title = title or f'Distribution of {column.capitalize()}'
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel(column.capitalize())
        ax.set_ylabel('Frequency')
        
        description = f"Histogram showing the distribution of {column}. "
        if hue:
            description += f"Color-coded by {hue}. "
        description += f"Data range: {df[column].min():.2f} to {df[column].max():.2f}, Mean: {df[column].mean():.2f}"
        
        return self._fig_to_base64(fig), description
    
    def create_bar_chart(
        self, 
        df: pd.DataFrame, 
        x: str, 
        y: Optional[str] = None,
        title: Optional[str] = None,
        hue: Optional[str] = None,
        orientation: str = 'vertical',
        aggregate: str = 'count'
    ) -> Tuple[str, str]:
        """Create a bar chart."""
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        if y is None:
            # Count plot
            if hue:
                data = df.groupby([x, hue]).size().unstack(fill_value=0)
                data.plot(kind='bar', ax=ax, width=0.8)
            else:
                counts = df[x].value_counts()
                if orientation == 'horizontal':
                    counts.plot(kind='barh', ax=ax, color='steelblue')
                else:
                    counts.plot(kind='bar', ax=ax, color='steelblue')
        else:
            # Aggregated bar chart
            if aggregate == 'mean':
                data = df.groupby(x)[y].mean()
            elif aggregate == 'sum':
                data = df.groupby(x)[y].sum()
            else:
                data = df.groupby(x)[y].count()
            
            if orientation == 'horizontal':
                data.plot(kind='barh', ax=ax, color='steelblue')
            else:
                data.plot(kind='bar', ax=ax, color='steelblue')
        
        title = title or f'{x.capitalize()} Distribution'
        ax.set_title(title, fontweight='bold', pad=20)
        
        if orientation == 'horizontal':
            ax.set_xlabel('Count' if y is None else y.capitalize())
            ax.set_ylabel(x.capitalize())
        else:
            ax.set_ylabel('Count' if y is None else y.capitalize())
            ax.set_xlabel(x.capitalize())
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        description = f"Bar chart showing {x} distribution"
        if hue:
            description += f", grouped by {hue}"
        
        return self._fig_to_base64(fig), description
    
    def create_survival_comparison(
        self, 
        df: pd.DataFrame, 
        groupby: str,
        title: Optional[str] = None
    ) -> Tuple[str, str]:
        """Create a survival rate comparison chart."""
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        survival_rates = df.groupby(groupby)['survived'].agg(['mean', 'count'])
        survival_rates.columns = ['Survival Rate', 'Count']
        survival_rates['Survival Rate'] *= 100
        
        colors = plt.cm.RdYlGn(survival_rates['Survival Rate'] / 100)
        
        bars = ax.bar(
            range(len(survival_rates)), 
            survival_rates['Survival Rate'],
            color=colors,
            edgecolor='white',
            linewidth=1.5
        )
        
        # Add value labels on bars
        for bar, rate, count in zip(bars, survival_rates['Survival Rate'], survival_rates['Count']):
            height = bar.get_height()
            ax.annotate(
                f'{rate:.1f}%\n(n={count})',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10, fontweight='bold'
            )
        
        ax.set_xticks(range(len(survival_rates)))
        ax.set_xticklabels(survival_rates.index, rotation=45, ha='right')
        
        title = title or f'Survival Rate by {groupby.capitalize()}'
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_ylabel('Survival Rate (%)')
        ax.set_xlabel(groupby.capitalize())
        ax.set_ylim(0, 100)
        
        # Add reference line for overall survival rate
        overall_rate = df['survived'].mean() * 100
        ax.axhline(y=overall_rate, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.annotate(
            f'Overall: {overall_rate:.1f}%', 
            xy=(len(survival_rates) - 0.5, overall_rate),
            fontsize=10, color='red', fontweight='bold'
        )
        
        plt.tight_layout()
        
        description = f"Survival rate comparison by {groupby}. "
        best_group = survival_rates['Survival Rate'].idxmax()
        worst_group = survival_rates['Survival Rate'].idxmin()
        description += f"Highest survival: {best_group} ({survival_rates.loc[best_group, 'Survival Rate']:.1f}%). "
        description += f"Lowest survival: {worst_group} ({survival_rates.loc[worst_group, 'Survival Rate']:.1f}%)."
        
        return self._fig_to_base64(fig), description
    
    def create_pie_chart(
        self, 
        df: pd.DataFrame, 
        column: str,
        title: Optional[str] = None
    ) -> Tuple[str, str]:
        """Create a pie chart."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        counts = df[column].value_counts()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
        
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index,
            autopct='%1.1f%%',
            colors=colors,
            explode=[0.02] * len(counts),
            shadow=True,
            startangle=90
        )
        
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        
        title = title or f'{column.capitalize()} Distribution'
        ax.set_title(title, fontweight='bold', pad=20)
        
        description = f"Pie chart showing {column} distribution. "
        for idx, val in counts.items():
            pct = val / counts.sum() * 100
            description += f"{idx}: {pct:.1f}%. "
        
        return self._fig_to_base64(fig), description
    
    def create_box_plot(
        self, 
        df: pd.DataFrame, 
        x: str, 
        y: str,
        title: Optional[str] = None,
        hue: Optional[str] = None
    ) -> Tuple[str, str]:
        """Create a box plot."""
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax, palette='Set2')
        
        title = title or f'{y.capitalize()} by {x.capitalize()}'
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel(x.capitalize())
        ax.set_ylabel(y.capitalize())
        
        if hue:
            ax.legend(title=hue.capitalize())
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        description = f"Box plot showing {y} distribution across {x} categories"
        if hue:
            description += f", grouped by {hue}"
        
        return self._fig_to_base64(fig), description
    
    def create_heatmap(
        self, 
        df: pd.DataFrame, 
        columns: Optional[list] = None,
        title: Optional[str] = None
    ) -> Tuple[str, str]:
        """Create a correlation heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if columns:
            numeric_df = df[columns].select_dtypes(include=[np.number])
        else:
            numeric_df = df.select_dtypes(include=[np.number])
        
        corr_matrix = numeric_df.corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
            vmin=-1,
            vmax=1
        )
        
        title = title or 'Correlation Heatmap'
        ax.set_title(title, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        description = "Correlation heatmap showing relationships between numeric variables. "
        description += "Values range from -1 (negative correlation) to 1 (positive correlation)."
        
        return self._fig_to_base64(fig), description
    
    def create_scatter_plot(
        self, 
        df: pd.DataFrame, 
        x: str, 
        y: str,
        title: Optional[str] = None,
        hue: Optional[str] = None,
        size: Optional[str] = None
    ) -> Tuple[str, str]:
        """Create a scatter plot."""
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        scatter_kwargs = {'data': df, 'x': x, 'y': y, 'ax': ax, 'alpha': 0.6}
        
        if hue:
            scatter_kwargs['hue'] = hue
            scatter_kwargs['palette'] = 'viridis'
        
        if size:
            scatter_kwargs['size'] = size
        
        sns.scatterplot(**scatter_kwargs)
        
        title = title or f'{y.capitalize()} vs {x.capitalize()}'
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel(x.capitalize())
        ax.set_ylabel(y.capitalize())
        
        plt.tight_layout()
        
        description = f"Scatter plot showing relationship between {x} and {y}"
        if hue:
            description += f", colored by {hue}"
        
        return self._fig_to_base64(fig), description
    
    def create_multi_panel_survival(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Create a multi-panel survival analysis visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Panel 1: Survival by Class
        survival_class = df.groupby('pclass')['survived'].mean() * 100
        colors1 = plt.cm.RdYlGn(survival_class.values / 100)
        axes[0, 0].bar(survival_class.index.astype(str), survival_class.values, color=colors1)
        axes[0, 0].set_title('Survival Rate by Class', fontweight='bold')
        axes[0, 0].set_xlabel('Passenger Class')
        axes[0, 0].set_ylabel('Survival Rate (%)')
        axes[0, 0].set_ylim(0, 100)
        for i, v in enumerate(survival_class.values):
            axes[0, 0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # Panel 2: Survival by Gender
        survival_sex = df.groupby('sex')['survived'].mean() * 100
        colors2 = plt.cm.RdYlGn(survival_sex.values / 100)
        axes[0, 1].bar(survival_sex.index, survival_sex.values, color=colors2)
        axes[0, 1].set_title('Survival Rate by Gender', fontweight='bold')
        axes[0, 1].set_xlabel('Gender')
        axes[0, 1].set_ylabel('Survival Rate (%)')
        axes[0, 1].set_ylim(0, 100)
        for i, v in enumerate(survival_sex.values):
            axes[0, 1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # Panel 3: Age distribution by survival
        df_survived = df[df['survived'] == 1]['age'].dropna()
        df_not_survived = df[df['survived'] == 0]['age'].dropna()
        axes[1, 0].hist(df_not_survived, bins=20, alpha=0.6, label='Did not survive', color='salmon')
        axes[1, 0].hist(df_survived, bins=20, alpha=0.6, label='Survived', color='seagreen')
        axes[1, 0].set_title('Age Distribution by Survival', fontweight='bold')
        axes[1, 0].set_xlabel('Age')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
        
        # Panel 4: Survival by Embarked
        if 'embarked' in df.columns:
            survival_embarked = df.groupby('embarked')['survived'].mean() * 100
            colors4 = plt.cm.RdYlGn(survival_embarked.values / 100)
            axes[1, 1].bar(survival_embarked.index, survival_embarked.values, color=colors4)
            axes[1, 1].set_title('Survival Rate by Embarkation', fontweight='bold')
            axes[1, 1].set_xlabel('Port')
            axes[1, 1].set_ylabel('Survival Rate (%)')
            axes[1, 1].set_ylim(0, 100)
            for i, v in enumerate(survival_embarked.values):
                axes[1, 1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        plt.suptitle('Titanic Survival Analysis Overview', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        description = "Comprehensive survival analysis showing survival rates by class, gender, age distribution, and embarkation port."
        
        return self._fig_to_base64(fig), description


# Singleton instance
viz_engine = VisualizationEngine()