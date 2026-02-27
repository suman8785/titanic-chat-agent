"""
Custom LangChain tools for the Titanic Chat Agent.
"""

import pandas as pd
import numpy as np
from typing import Optional, Type, Any, List
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field
import json
import logging

from backend.data_loader import get_data_loader
from backend.visualization import viz_engine

logger = logging.getLogger(__name__)


# ============================================
# Tool Input Schemas
# ============================================

class DataQueryInput(BaseModel):
    """Input schema for data query tool."""
    query: str = Field(
        description="A pandas query string or natural language description of the data query"
    )
    columns: Optional[List[str]] = Field(
        default=None,
        description="Specific columns to return in the result"
    )
    limit: int = Field(
        default=10,
        description="Maximum number of rows to return"
    )


class StatisticalAnalysisInput(BaseModel):
    """Input schema for statistical analysis tool."""
    analysis_type: str = Field(
        description="Type of analysis: 'describe', 'correlation', 'value_counts', 'groupby', 'survival_rate'"
    )
    column: Optional[str] = Field(
        default=None,
        description="Column to analyze (required for value_counts and some operations)"
    )
    groupby_column: Optional[str] = Field(
        default=None,
        description="Column to group by (for groupby and survival_rate operations)"
    )
    target_column: Optional[str] = Field(
        default=None,
        description="Target column for aggregation in groupby operations"
    )
    aggregation: str = Field(
        default="mean",
        description="Aggregation function for groupby: 'mean', 'sum', 'count', 'median'"
    )


class VisualizationInput(BaseModel):
    """Input schema for visualization tool."""
    chart_type: str = Field(
        description="Type of chart: 'histogram', 'bar_chart', 'survival_comparison', 'pie_chart', 'box_plot', 'heatmap', 'scatter', 'multi_panel'"
    )
    column: Optional[str] = Field(
        default=None,
        description="Primary column for the visualization"
    )
    groupby: Optional[str] = Field(
        default=None,
        description="Column to group or color by"
    )
    y_column: Optional[str] = Field(
        default=None,
        description="Secondary column (for scatter plots, box plots)"
    )
    title: Optional[str] = Field(
        default=None,
        description="Custom title for the chart"
    )


class InsightGenerationInput(BaseModel):
    """Input schema for insight generation tool."""
    focus_area: str = Field(
        description="Area to generate insights for: 'survival', 'demographics', 'fare', 'family', 'overall'"
    )


# ============================================
# Tools
# ============================================

class DataQueryTool(BaseTool):
    """Tool for querying and filtering the Titanic dataset."""
    
    name: str = "titanic_data_query"
    description: str = """
    Query and filter the Titanic dataset. Use this tool to:
    - Get specific rows based on conditions (e.g., "survived == 1", "age > 30", "sex == 'female'")
    - Retrieve data for specific columns
    - Count passengers matching criteria
    - Find specific passenger information
    
    The dataset has columns: survived, pclass, sex, age, sibsp, parch, fare, embarked, 
    class, who, adult_male, deck, embark_town, alive, alone, family_size, is_alone, 
    age_group, fare_category.
    
    Examples of valid queries:
    - "survived == 1 and pclass == 1"
    - "age > 50 and sex == 'female'"
    - "fare > 100"
    """
    args_schema: Type[BaseModel] = DataQueryInput
    
    def _run(
        self, 
        query: str,
        columns: Optional[List[str]] = None,
        limit: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the data query."""
        try:
            loader = get_data_loader()
            df = loader.dataframe
            
            # Try to parse as a pandas query
            if any(op in query for op in ['==', '>', '<', '>=', '<=', '!=']):
                result = df.query(query)
            else:
                # Return full dataset if no filter
                result = df
            
            # Select columns if specified
            if columns:
                valid_columns = [c for c in columns if c in result.columns]
                if valid_columns:
                    result = result[valid_columns]
            
            # Limit results
            result = result.head(limit)
            
            # Format output
            output = {
                "total_matching": len(df.query(query)) if any(op in query for op in ['==', '>', '<', '>=', '<=', '!=']) else len(df),
                "showing": len(result),
                "data": result.to_dict(orient='records')
            }
            
            return json.dumps(output, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Data query error: {e}")
            return json.dumps({"error": str(e), "hint": "Check your query syntax. Use pandas query format."})


class StatisticalAnalysisTool(BaseTool):
    """Tool for performing statistical analysis on the Titanic dataset."""
    
    name: str = "titanic_statistical_analysis"
    description: str = """
    Perform statistical analysis on the Titanic dataset. Use this tool to:
    - Get descriptive statistics (describe)
    - Calculate correlations between numeric columns (correlation)
    - Get value counts for categorical columns (value_counts)
    - Perform group-by aggregations (groupby)
    - Calculate survival rates by different categories (survival_rate)
    
    Analysis types:
    - 'describe': Get summary statistics for all numeric columns
    - 'correlation': Get correlation matrix for numeric columns
    - 'value_counts': Count unique values in a column
    - 'groupby': Group by a column and aggregate another
    - 'survival_rate': Calculate survival rate grouped by a column
    """
    args_schema: Type[BaseModel] = StatisticalAnalysisInput
    
    def _run(
        self,
        analysis_type: str,
        column: Optional[str] = None,
        groupby_column: Optional[str] = None,
        target_column: Optional[str] = None,
        aggregation: str = "mean",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the statistical analysis."""
        try:
            loader = get_data_loader()
            df = loader.dataframe
            
            if analysis_type == "describe":
                result = df.describe().to_dict()
                
            elif analysis_type == "correlation":
                numeric_df = df.select_dtypes(include=[np.number])
                result = numeric_df.corr().round(3).to_dict()
                
            elif analysis_type == "value_counts":
                if not column:
                    return json.dumps({"error": "Column name required for value_counts"})
                if column not in df.columns:
                    return json.dumps({"error": f"Column '{column}' not found"})
                counts = df[column].value_counts()
                result = {
                    "column": column,
                    "counts": counts.to_dict(),
                    "percentages": (counts / len(df) * 100).round(2).to_dict()
                }
                
            elif analysis_type == "groupby":
                if not groupby_column:
                    return json.dumps({"error": "groupby_column required"})
                target = target_column or "survived"
                
                agg_funcs = {
                    "mean": "mean",
                    "sum": "sum", 
                    "count": "count",
                    "median": "median"
                }
                agg_func = agg_funcs.get(aggregation, "mean")
                
                grouped = df.groupby(groupby_column)[target].agg(agg_func)
                result = {
                    "groupby": groupby_column,
                    "target": target,
                    "aggregation": aggregation,
                    "results": grouped.round(4).to_dict()
                }
                
            elif analysis_type == "survival_rate":
                groupby = groupby_column or "pclass"
                survival = df.groupby(groupby)['survived'].agg(['mean', 'sum', 'count'])
                survival.columns = ['survival_rate', 'survived_count', 'total_count']
                survival['survival_rate'] = (survival['survival_rate'] * 100).round(2)
                result = {
                    "groupby": groupby,
                    "overall_survival_rate": round(df['survived'].mean() * 100, 2),
                    "by_group": survival.to_dict(orient='index')
                }
                
            else:
                return json.dumps({"error": f"Unknown analysis type: {analysis_type}"})
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Statistical analysis error: {e}")
            return json.dumps({"error": str(e)})


class VisualizationTool(BaseTool):
    """Tool for creating visualizations of the Titanic dataset."""
    
    name: str = "titanic_visualization"
    description: str = """
    Create visualizations of the Titanic dataset. Use this tool when the user asks for a chart, 
    graph, plot, or visual representation. Also use it when showing data visually would help 
    explain patterns or comparisons.
    
    Chart types:
    - 'histogram': Distribution of a numeric column (age, fare)
    - 'bar_chart': Counts or values for categorical columns (pclass, sex, embarked)
    - 'survival_comparison': Compare survival rates across a category (RECOMMENDED for survival questions)
    - 'pie_chart': Proportions of categories
    - 'box_plot': Distribution comparison across categories
    - 'heatmap': Correlation heatmap of numeric variables
    - 'scatter': Scatter plot between two numeric variables
    - 'multi_panel': Comprehensive survival analysis dashboard
    
    For survival-related questions, prefer 'survival_comparison' or 'multi_panel'.
    For distribution questions, use 'histogram' or 'box_plot'.
    For counting questions, use 'bar_chart' or 'pie_chart'.
    """
    args_schema: Type[BaseModel] = VisualizationInput
    return_direct: bool = False
    
    def _run(
        self,
        chart_type: str,
        column: Optional[str] = None,
        groupby: Optional[str] = None,
        y_column: Optional[str] = None,
        title: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Generate the visualization."""
        try:
            loader = get_data_loader()
            df = loader.dataframe
            
            if chart_type == "histogram":
                col = column or "age"
                image_b64, description = viz_engine.create_histogram(df, col, title=title, hue=groupby)
                
            elif chart_type == "bar_chart":
                col = column or "pclass"
                image_b64, description = viz_engine.create_bar_chart(df, col, title=title, hue=groupby)
                
            elif chart_type == "survival_comparison":
                grp = groupby or column or "pclass"
                image_b64, description = viz_engine.create_survival_comparison(df, grp, title=title)
                
            elif chart_type == "pie_chart":
                col = column or "pclass"
                image_b64, description = viz_engine.create_pie_chart(df, col, title=title)
                
            elif chart_type == "box_plot":
                x = column or "pclass"
                y = y_column or "age"
                image_b64, description = viz_engine.create_box_plot(df, x, y, title=title, hue=groupby)
                
            elif chart_type == "heatmap":
                image_b64, description = viz_engine.create_heatmap(df, title=title)
                
            elif chart_type == "scatter":
                x = column or "age"
                y = y_column or "fare"
                image_b64, description = viz_engine.create_scatter_plot(df, x, y, title=title, hue=groupby)
                
            elif chart_type == "multi_panel":
                image_b64, description = viz_engine.create_multi_panel_survival(df)
                
            else:
                return json.dumps({"error": f"Unknown chart type: {chart_type}"})
            
            result = {
                "chart_type": chart_type,
                "image_base64": image_b64,
                "description": description,
                "title": title or f"{chart_type.replace('_', ' ').title()}"
            }
            
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return json.dumps({"error": str(e)})


class InsightGenerationTool(BaseTool):
    """Tool for generating automatic insights about the Titanic dataset."""
    
    name: str = "titanic_insights"
    description: str = """
    Generate automatic insights and interesting findings about the Titanic dataset.
    Use this when the user asks for insights, patterns, interesting facts, or a summary.
    
    Focus areas:
    - 'survival': Key survival patterns and factors
    - 'demographics': Age, gender, class distributions
    - 'fare': Fare analysis and patterns
    - 'family': Family size and traveling companion patterns
    - 'overall': Comprehensive overview of all patterns
    """
    args_schema: Type[BaseModel] = InsightGenerationInput
    
    def _run(
        self,
        focus_area: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Generate insights for the specified area."""
        try:
            loader = get_data_loader()
            df = loader.dataframe
            
            insights = []
            
            if focus_area in ['survival', 'overall']:
                # Survival insights
                overall_survival = df['survived'].mean() * 100
                insights.append(f"📊 Overall survival rate: {overall_survival:.1f}%")
                
                # By gender
                female_survival = df[df['sex'] == 'female']['survived'].mean() * 100
                male_survival = df[df['sex'] == 'male']['survived'].mean() * 100
                insights.append(f"👩 Women survival rate: {female_survival:.1f}% vs 👨 Men: {male_survival:.1f}%")
                
                # By class
                for pclass in [1, 2, 3]:
                    class_survival = df[df['pclass'] == pclass]['survived'].mean() * 100
                    insights.append(f"🎫 Class {pclass} survival rate: {class_survival:.1f}%")
                
                # Children
                children_survival = df[df['age'] < 18]['survived'].mean() * 100
                adult_survival = df[df['age'] >= 18]['survived'].mean() * 100
                insights.append(f"👶 Children (<18) survival: {children_survival:.1f}% vs Adults: {adult_survival:.1f}%")
            
            if focus_area in ['demographics', 'overall']:
                # Demographics insights
                avg_age = df['age'].mean()
                median_age = df['age'].median()
                insights.append(f"📅 Average passenger age: {avg_age:.1f} years (median: {median_age:.1f})")
                
                male_pct = (df['sex'] == 'male').mean() * 100
                insights.append(f"⚥ Gender distribution: {male_pct:.1f}% male, {100-male_pct:.1f}% female")
                
                for pclass in [1, 2, 3]:
                    class_pct = (df['pclass'] == pclass).mean() * 100
                    insights.append(f"🎟️ Class {pclass}: {class_pct:.1f}% of passengers")
            
            if focus_area in ['fare', 'overall']:
                # Fare insights
                avg_fare = df['fare'].mean()
                median_fare = df['fare'].median()
                max_fare = df['fare'].max()
                insights.append(f"💰 Average fare: ${avg_fare:.2f} (median: ${median_fare:.2f}, max: ${max_fare:.2f})")
                
                # Fare by class
                for pclass in [1, 2, 3]:
                    class_fare = df[df['pclass'] == pclass]['fare'].mean()
                    insights.append(f"💵 Average Class {pclass} fare: ${class_fare:.2f}")
            
            if focus_area in ['family', 'overall']:
                # Family insights
                avg_family = df['family_size'].mean()
                alone_pct = df['is_alone'].mean() * 100
                insights.append(f"👨‍👩‍👧‍👦 Average family size: {avg_family:.1f}")
                insights.append(f"🧍 Passengers traveling alone: {alone_pct:.1f}%")
                
                alone_survival = df[df['is_alone'] == 1]['survived'].mean() * 100
                family_survival = df[df['is_alone'] == 0]['survived'].mean() * 100
                insights.append(f"🏠 Survival - Alone: {alone_survival:.1f}% vs With family: {family_survival:.1f}%")
            
            result = {
                "focus_area": focus_area,
                "insights": insights,
                "total_passengers": len(df),
                "total_survivors": int(df['survived'].sum())
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Insight generation error: {e}")
            return json.dumps({"error": str(e)})


class DatasetInfoTool(BaseTool):
    """Tool for getting information about the dataset structure."""
    
    name: str = "titanic_dataset_info"
    description: str = """
    Get information about the Titanic dataset structure, columns, and data types.
    Use this when the user asks what data is available, what columns exist, or how to query the data.
    """
    
    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Get dataset information."""
        try:
            loader = get_data_loader()
            stats = loader.get_statistics()
            column_info = loader.get_column_info()
            
            result = {
                "total_passengers": stats['total_passengers'],
                "columns": stats['columns'],
                "column_details": {
                    col: {
                        "dtype": info['dtype'],
                        "non_null": info['non_null_count'],
                        "unique_values": info.get('unique_values', info.get('unique_count'))
                    }
                    for col, info in column_info.items()
                },
                "useful_columns_for_analysis": [
                    "survived (0/1 - target variable)",
                    "pclass (1/2/3 - passenger class)",
                    "sex (male/female)",
                    "age (numeric)",
                    "fare (ticket price)",
                    "embarked (S/C/Q - port)",
                    "family_size (derived)",
                    "is_alone (derived)"
                ]
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Dataset info error: {e}")
            return json.dumps({"error": str(e)})


def get_all_tools() -> List[BaseTool]:
    """Get all available tools for the agent."""
    return [
        DataQueryTool(),
        StatisticalAnalysisTool(),
        VisualizationTool(),
        InsightGenerationTool(),
        DatasetInfoTool()
    ]