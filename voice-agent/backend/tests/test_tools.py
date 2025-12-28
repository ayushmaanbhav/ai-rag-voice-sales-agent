"""
Tests for tool system: base tools, gold loan tools, executor.
"""
import pytest
from tools.base_tool import ToolParameter, ToolResult
from tools.gold_loan_tools import (
    CalculateSavingsTool,
    CalculateEMITool,
    CalculateLoanAmountTool,
    CreateSupportTicketTool,
    GetBranchInfoTool,
)
from tools.executor import ToolExecutor, get_tool_executor


class TestCalculateSavingsTool:
    """Test savings calculator tool."""

    @pytest.fixture
    def tool(self):
        return CalculateSavingsTool()

    def test_calculate_savings(self, tool):
        """Test savings calculation."""
        result = tool.execute(
            loan_amount=500000,
            current_rate=18.0,
            kotak_rate=10.0
        )
        assert result.success
        assert "40,000" in result.display_text  # 8% of 5L = 40K
        assert result.data["annual_savings"] == 40000

    def test_calculate_savings_default_kotak_rate(self, tool):
        """Test with default Kotak rate."""
        result = tool.execute(
            loan_amount=100000,
            current_rate=20.0
        )
        assert result.success
        assert result.data["kotak_rate"] == 10.0

    def test_invalid_rates(self, tool):
        """Test with invalid rate (current < kotak)."""
        result = tool.execute(
            loan_amount=100000,
            current_rate=8.0,
            kotak_rate=10.0
        )
        assert result.success
        assert result.data["annual_savings"] < 0  # Negative savings


class TestCalculateEMITool:
    """Test EMI calculator tool."""

    @pytest.fixture
    def tool(self):
        return CalculateEMITool()

    def test_calculate_emi(self, tool):
        """Test EMI calculation."""
        result = tool.execute(
            loan_amount=100000,  # Correct param name
            interest_rate=12.0,  # Correct param name
            tenure_months=12
        )
        assert result.success
        assert "monthly_emi" in result.data
        assert result.data["monthly_emi"] > 0
        assert result.data["total_interest"] > 0

    def test_calculate_emi_different_tenures(self, tool):
        """Test EMI varies with tenure."""
        result_12 = tool.execute(loan_amount=100000, interest_rate=12.0, tenure_months=12)
        result_24 = tool.execute(loan_amount=100000, interest_rate=12.0, tenure_months=24)

        # Longer tenure = lower EMI
        assert result_24.data["monthly_emi"] < result_12.data["monthly_emi"]
        # But more total interest
        assert result_24.data["total_interest"] > result_12.data["total_interest"]


class TestCalculateLoanAmountTool:
    """Test loan amount estimator tool."""

    @pytest.fixture
    def tool(self):
        return CalculateLoanAmountTool()

    def test_calculate_loan_amount(self, tool):
        """Test loan amount estimation."""
        result = tool.execute(
            gold_weight_grams=100,
            gold_purity=22
        )
        assert result.success
        assert result.data["eligible_loan_amount"] > 0  # Correct key name
        assert result.data["ltv_percent"] == 75  # Correct key name

    def test_different_purities(self, tool):
        """Test different gold purities."""
        result_22k = tool.execute(gold_weight_grams=100, gold_purity=22)
        result_24k = tool.execute(gold_weight_grams=100, gold_purity=24)

        # 24K gold = higher loan amount
        assert result_24k.data["eligible_loan_amount"] > result_22k.data["eligible_loan_amount"]


class TestToolExecutor:
    """Test tool executor."""

    @pytest.fixture
    def executor(self):
        return get_tool_executor()

    def test_list_tools(self, executor):
        """Test listing available tools."""
        tools = list(executor.tools.keys())  # Use tools dict
        assert len(tools) >= 5
        assert "calculate_savings" in tools
        assert "calculate_emi" in tools

    def test_execute_tool(self, executor):
        """Test executing a tool by name."""
        result = executor.execute_tool(
            "calculate_savings",
            {"loan_amount": 500000, "current_rate": 18.0}
        )
        assert result.success
        assert "savings" in result.display_text.lower()

    def test_execute_unknown_tool(self, executor):
        """Test executing unknown tool."""
        result = executor.execute_tool("unknown_tool", {})
        assert not result.success
        assert "unknown" in result.error.lower()  # Match actual error message

    def test_parse_tool_calls(self, executor):
        """Test parsing XML tool calls from LLM output."""
        llm_output = """
        I'll calculate your savings.

        <tool_call>
          <name>calculate_savings</name>
          <parameters>
            <loan_amount>500000</loan_amount>
            <current_rate>18</current_rate>
          </parameters>
        </tool_call>

        This will help you understand the benefits.
        """
        tool_calls = executor.parse_tool_calls(llm_output)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "calculate_savings"
        assert tool_calls[0]["parameters"]["loan_amount"] == "500000"

    def test_process_llm_output(self, executor):
        """Test processing LLM output with tool calls."""
        llm_output = """
        Let me calculate your savings.

        <tool_call>
          <name>calculate_savings</name>
          <parameters>
            <loan_amount>500000</loan_amount>
            <current_rate>18</current_rate>
            <kotak_rate>10</kotak_rate>
          </parameters>
        </tool_call>
        """
        cleaned, results = executor.process_llm_output(llm_output)
        assert "Let me calculate" in cleaned
        assert "<tool_call>" not in cleaned
        assert len(results) == 1
        assert results[0].success

    def test_get_tools_prompt(self, executor):
        """Test getting tools description for prompt."""
        prompt = executor.get_tools_prompt()
        assert "calculate_savings" in prompt
        assert "loan_amount" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
