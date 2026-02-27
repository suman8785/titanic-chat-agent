"""
Data loading and preprocessing utilities for the Titanic dataset.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class TitanicDataLoader:
    """Handles loading and preprocessing of the Titanic dataset."""
    
    _instance: Optional['TitanicDataLoader'] = None
    _df: Optional[pd.DataFrame] = None
    
    def __new__(cls) -> 'TitanicDataLoader':
        """Singleton pattern to ensure single dataset instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the data loader."""
        if self._df is None:
            self._df = self._load_and_preprocess()
    
    def _load_and_preprocess(self) -> pd.DataFrame:
        """Load and preprocess the Titanic dataset."""
        logger.info("Loading Titanic dataset...")
        
        try:
            # Try loading from seaborn first
            import seaborn as sns
            df = sns.load_dataset('titanic')
            logger.info("Loaded dataset from seaborn")
        except Exception as e:
            logger.warning(f"Could not load from seaborn: {e}. Trying alternative source...")
            # Fallback to direct URL
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            df = pd.read_csv(url)
            df = self._standardize_columns(df)
        
        # Preprocess the data
        df = self._preprocess(df)
        
        logger.info(f"Dataset loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for consistency."""
        column_mapping = {
            'PassengerId': 'passenger_id',
            'Survived': 'survived',
            'Pclass': 'pclass',
            'Name': 'name',
            'Sex': 'sex',
            'Age': 'age',
            'SibSp': 'sibsp',
            'Parch': 'parch',
            'Ticket': 'ticket',
            'Fare': 'fare',
            'Cabin': 'cabin',
            'Embarked': 'embarked'
        }
        
        df = df.rename(columns=column_mapping)
        return df
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing transformations."""
        df = df.copy()
        
        # Create derived features
        if 'sibsp' in df.columns and 'parch' in df.columns:
            df['family_size'] = df['sibsp'] + df['parch'] + 1
            df['is_alone'] = (df['family_size'] == 1).astype(int)
        
        # Create age groups
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'], 
                bins=[0, 12, 18, 35, 50, 65, 100],
                labels=['Child', 'Teenager', 'Young Adult', 'Adult', 'Middle Aged', 'Senior'],
                include_lowest=True
            )
        
        # Create fare categories
        if 'fare' in df.columns:
            df['fare_category'] = pd.qcut(
                df['fare'].fillna(df['fare'].median()), 
                q=4, 
                labels=['Low', 'Medium', 'High', 'Very High']
            )
        
        # Map embarkation ports to full names
        if 'embarked' in df.columns:
            embark_mapping = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
            df['embark_town'] = df['embarked'].map(embark_mapping)
        
        # Create class labels
        if 'pclass' in df.columns:
            class_mapping = {1: 'First', 2: 'Second', 3: 'Third'}
            df['class'] = df['pclass'].map(class_mapping)
        
        # Create survival labels
        if 'survived' in df.columns:
            df['survived_label'] = df['survived'].map({0: 'Did not survive', 1: 'Survived'})
        
        return df
    
    @property
    def dataframe(self) -> pd.DataFrame:
        """Get the preprocessed dataframe."""
        return self._df.copy()
    
    def get_column_info(self) -> dict:
        """Get information about all columns."""
        info = {}
        for col in self._df.columns:
            col_info = {
                'dtype': str(self._df[col].dtype),
                'non_null_count': int(self._df[col].notna().sum()),
                'null_count': int(self._df[col].isna().sum()),
                'unique_count': int(self._df[col].nunique())
            }
            
            if self._df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'min': float(self._df[col].min()) if pd.notna(self._df[col].min()) else None,
                    'max': float(self._df[col].max()) if pd.notna(self._df[col].max()) else None,
                    'mean': float(self._df[col].mean()) if pd.notna(self._df[col].mean()) else None,
                    'std': float(self._df[col].std()) if pd.notna(self._df[col].std()) else None
                })
            elif self._df[col].dtype == 'object' or self._df[col].dtype.name == 'category':
                col_info['unique_values'] = self._df[col].dropna().unique().tolist()[:10]
            
            info[col] = col_info
        
        return info
    
    def get_statistics(self) -> dict:
        """Get comprehensive dataset statistics."""
        df = self._df
        
        return {
            'total_passengers': len(df),
            'total_survived': int(df['survived'].sum()),
            'survival_rate': round(df['survived'].mean() * 100, 2),
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'class_distribution': df['pclass'].value_counts().to_dict() if 'pclass' in df.columns else {},
            'gender_distribution': df['sex'].value_counts().to_dict() if 'sex' in df.columns else {},
            'age_stats': {
                'mean': round(df['age'].mean(), 2) if 'age' in df.columns else None,
                'median': round(df['age'].median(), 2) if 'age' in df.columns else None,
                'min': round(df['age'].min(), 2) if 'age' in df.columns else None,
                'max': round(df['age'].max(), 2) if 'age' in df.columns else None
            },
            'fare_stats': {
                'mean': round(df['fare'].mean(), 2) if 'fare' in df.columns else None,
                'median': round(df['fare'].median(), 2) if 'fare' in df.columns else None,
                'min': round(df['fare'].min(), 2) if 'fare' in df.columns else None,
                'max': round(df['fare'].max(), 2) if 'fare' in df.columns else None
            }
        }
    
    def query(self, query_str: str) -> pd.DataFrame:
        """Execute a pandas query on the dataset."""
        try:
            return self._df.query(query_str)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise ValueError(f"Invalid query: {e}")


@lru_cache()
def get_data_loader() -> TitanicDataLoader:
    """Get cached data loader instance."""
    return TitanicDataLoader()