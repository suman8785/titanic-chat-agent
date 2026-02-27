"""
Tests for the custom LangChain tools.
"""

import pytest
import json
from backend.tools import (
    DataQueryTool,
    StatisticalAnalysisTool,
    VisualizationTool,
    InsightGenerationTool,
    DatasetInfoTool
)


class TestDataQueryTool:
    """Tests for DataQueryTool."""
    
    @pytest.fixture
    def tool(self):
        return DataQueryTool()
    
    def test_basic_query(self, tool):
        """Test basic data query."""
        result = tool._run(query="survived == 1", limit=5)
        data = json.loads(result)
        
        assert "total_matching" in data
        assert "data" in data
        assert len(data["data"]) <= 5
    
    def test_query_with_columns(self, tool):
        """Test query with specific columns."""
        result = tool._run(
            query="pclass == 1",
            columns=["name", "age", "survived"],
            limit=3
        )
        data = json.loads(result)
        
        assert "data" in data
        # Check that only requested columns are present
        if data["data"]:
            assert set(data["data"][0].keys()).issubset({"name", "age", "survived"})
    
    def test_invalid_query(self, tool):
        """Test handling of invalid query."""
        result = tool._run(query="invalid_column == 'test'")
        data = json.loads(result)
        
        assert "error" in data


class TestStatisticalAnalysisTool:
    """Tests for StatisticalAnalysisTool."""
    
    @pytest.fixture
    def tool(self):
        return StatisticalAnalysisTool()
    
    def test_describe(self, tool):
        """Test describe analysis."""
        result = tool._run(analysis_type="describe")
        data = json.loads(result)
        
        assert "survived" in data or "age" in data
    
    def test_value_counts(self, tool):
        """Test value counts analysis."""
        result = tool._run(analysis_type="value_counts", column="sex")
        data = json.loads(result)
        
        assert "counts" in data
        assert "percentages" in data
    
    def test_survival_rate(self, tool):
        """Test survival rate calculation."""
        result = tool._run(analysis_type="survival_rate", groupby_column="pclass")
        data = json.loads(result)
        
        assert "overall_survival_rate" in data
        assert "by_group" in data
    
    def test_missing_column_error(self, tool):
        """Test error handling for missing column."""
        result = tool._run(analysis_type="value_counts")
        data = json.loads(result)
        
        assert "error" in data


class TestVisualizationTool:
    """Tests for VisualizationTool."""
    
    @pytest.fixture
    def tool(self):
        return VisualizationTool()
    
    def test_histogram(self, tool):
        """Test histogram generation."""
        result = tool._run(chart_type="histogram", column="age")
        data = json.loads(result)
        
        assert "image_base64" in data
        assert "description" in data
        assert data["chart_type"] == "histogram"
    
    def test_survival_comparison(self, tool):
        """Test survival comparison chart."""
        result = tool._run(chart_type="survival_comparison", groupby="sex")
        data = json.loads(result)
        
        assert "image_base64" in data
        assert "description" in data
    
    def test_invalid_chart_type(self, tool):
        """Test error handling for invalid chart type."""
        result = tool._run(chart_type="invalid_type")
        data = json.loads(result)
        
        assert "error" in data


class TestInsightGenerationTool:
    """Tests for InsightGenerationTool."""
    
    @pytest.fixture
    def tool(self):
        return InsightGenerationTool()
    
    def test_survival_insights(self, tool):
        """Test survival insights generation."""
        result = tool._run(focus_area="survival")
        data = json.loads(result)
        
        assert "insights" in data
        assert len(data["insights"]) > 0
    
    def test_overall_insights(self, tool):
        """Test overall insights generation."""
        result = tool._run(focus_area="overall")
        data = json.loads(result)
        
        assert "insights" in data
        assert "total_passengers" in data


class TestDatasetInfoTool:
    """Tests for DatasetInfoTool."""
    
    @pytest.fixture
    def tool(self):
        return DatasetInfoTool()
    
    def test_get_info(self, tool):
        """Test dataset info retrieval."""
        result = tool._run()
        data = json.loads(result)
        
        assert "total_passengers" in data
        assert "columns" in data
        assert "column_details" in data