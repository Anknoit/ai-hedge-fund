from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
    get_company_news,
    get_prices,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import json
import statistics
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm
from src.utils.api_key import get_api_key_from_state


class RayDalioSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(description="Confidence level 0-100")
    reasoning: str = Field(description="Dalio-style macro and fundamental reasoning")


def ray_dalio_agent(state: AgentState, agent_id: str = "ray_dalio_agent"):
    """
    Analyzes stocks using Ray Dalio's investment principles:
      - Debt cycle analysis: sustainability of leverage relative to income generation
      - Productivity & real growth: revenue and earnings trends across economic cycles
      - Balance sheet resilience: cash buffers and ability to survive deleveraging
      - All-weather characteristics: pricing power, FCF consistency, inflation sensitivity
      - Macro cycle positioning: valuation and cyclicality context

    Returns a bullish/bearish/neutral signal with confidence and reasoning.
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    analysis_data = {}
    dalio_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5, api_key=api_key)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "net_income",
                "operating_income",
                "gross_profit",
                "earnings_per_share",
                "free_cash_flow",
                "capital_expenditure",
                "cash_and_equivalents",
                "total_debt",
                "total_assets",
                "total_liabilities",
                "shareholders_equity",
                "interest_expense",
                "ebit",
                "ebitda",
                "outstanding_shares",
            ],
            end_date,
            period="annual",
            limit=5,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching recent price data")
        prices = get_prices(ticker, start_date=start_date, end_date=end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching company news")
        company_news = get_company_news(ticker, end_date, limit=30, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing debt cycle position")
        debt_analysis = analyze_debt_cycle(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing productivity and real growth")
        productivity_analysis = analyze_productivity_and_growth(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing balance sheet resilience")
        balance_sheet_analysis = analyze_balance_sheet_resilience(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing all-weather characteristics")
        all_weather_analysis = analyze_all_weather_characteristics(financial_line_items, prices)

        progress.update_status(agent_id, ticker, "Analyzing macro cycle positioning")
        macro_analysis = analyze_macro_cycle_positioning(financial_line_items, market_cap, company_news)

        # Dalio weighting: debt sustainability matters most, then productivity, balance sheet, all-weather, macro
        total_score = (
            debt_analysis["score"] * 0.25
            + productivity_analysis["score"] * 0.25
            + balance_sheet_analysis["score"] * 0.20
            + all_weather_analysis["score"] * 0.15
            + macro_analysis["score"] * 0.15
        )

        if total_score >= 7.0:
            signal = "bullish"
        elif total_score <= 4.0:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": 10,
            "debt_cycle_analysis": debt_analysis,
            "productivity_growth_analysis": productivity_analysis,
            "balance_sheet_resilience": balance_sheet_analysis,
            "all_weather_characteristics": all_weather_analysis,
            "macro_cycle_positioning": macro_analysis,
        }

        progress.update_status(agent_id, ticker, "Generating Ray Dalio analysis")
        dalio_output = generate_dalio_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        dalio_analysis[ticker] = {
            "signal": dalio_output.signal,
            "confidence": dalio_output.confidence,
            "reasoning": dalio_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=dalio_output.reasoning)

    message = HumanMessage(content=json.dumps(dalio_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(dalio_analysis, "Ray Dalio Agent")

    state["data"]["analyst_signals"][agent_id] = dalio_analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


def analyze_debt_cycle(financial_line_items: list) -> dict:
    """
    Dalio's core framework: assess whether debt is sustainable relative to income.
    Checks debt-to-equity trend, interest coverage, and debt growth vs. income growth.
    Max 10 raw points scaled to 0-10.
    """
    if not financial_line_items or len(financial_line_items) < 2:
        return {"score": 0, "details": "Insufficient data for debt cycle analysis"}

    details = []
    raw_score = 0  # max 10 raw points

    # 1. Debt-to-Equity (up to 3 pts)
    debt_values = [fi.total_debt for fi in financial_line_items if fi.total_debt is not None]
    equity_values = [fi.shareholders_equity for fi in financial_line_items if fi.shareholders_equity is not None]

    if debt_values and equity_values:
        recent_debt = debt_values[0]
        recent_equity = equity_values[0] if equity_values[0] and equity_values[0] != 0 else 1e-9
        de_ratio = recent_debt / recent_equity
        if de_ratio < 0.5:
            raw_score += 3
            details.append(f"Conservative debt-to-equity: {de_ratio:.2f} — well within sustainable range")
        elif de_ratio < 1.0:
            raw_score += 2
            details.append(f"Moderate debt-to-equity: {de_ratio:.2f}")
        elif de_ratio < 2.0:
            raw_score += 1
            details.append(f"Elevated debt-to-equity: {de_ratio:.2f} — approaching deleveraging risk")
        else:
            details.append(f"High debt-to-equity: {de_ratio:.2f} — unsustainable leverage per Dalio framework")
    else:
        details.append("Debt/equity data unavailable")

    # 2. Interest Coverage (EBIT / Interest Expense) (up to 3 pts)
    ebit_values = [fi.ebit for fi in financial_line_items if fi.ebit is not None]
    interest_values = [fi.interest_expense for fi in financial_line_items if fi.interest_expense is not None]

    if ebit_values and interest_values:
        recent_ebit = ebit_values[0]
        recent_interest = abs(interest_values[0]) if interest_values[0] else 1e-9
        if recent_interest > 0:
            coverage = recent_ebit / recent_interest
            if coverage > 10:
                raw_score += 3
                details.append(f"Strong interest coverage: {coverage:.1f}x — debt easily serviceable")
            elif coverage > 5:
                raw_score += 2
                details.append(f"Adequate interest coverage: {coverage:.1f}x")
            elif coverage > 2:
                raw_score += 1
                details.append(f"Thin interest coverage: {coverage:.1f}x — vulnerable in downturns")
            else:
                details.append(f"Weak interest coverage: {coverage:.1f}x — debt service risk elevated")
        else:
            raw_score += 3
            details.append("No interest expense — debt-free or minimal debt")
    else:
        details.append("Interest coverage data unavailable")

    # 3. Debt Growth vs. Income Growth (up to 4 pts)
    net_incomes = [fi.net_income for fi in financial_line_items if fi.net_income is not None]

    if len(debt_values) >= 2 and len(net_incomes) >= 2:
        debt_growth = (debt_values[0] - debt_values[-1]) / (abs(debt_values[-1]) + 1e-9)
        income_growth = (net_incomes[0] - net_incomes[-1]) / (abs(net_incomes[-1]) + 1e-9)

        if income_growth > debt_growth and income_growth > 0:
            raw_score += 4
            details.append(f"Virtuous cycle: income growing ({income_growth:.1%}) faster than debt ({debt_growth:.1%})")
        elif income_growth > 0 and debt_growth < 0.1:
            raw_score += 3
            details.append(f"Healthy: income growing ({income_growth:.1%}), debt stable ({debt_growth:.1%})")
        elif income_growth > 0:
            raw_score += 2
            details.append(f"Caution: debt growing ({debt_growth:.1%}) faster than income ({income_growth:.1%})")
        elif debt_growth < 0:
            raw_score += 2
            details.append(f"Deleveraging: debt shrinking ({debt_growth:.1%}) — could signal or enable recovery")
        else:
            details.append(f"Concerning: income declining ({income_growth:.1%}) while debt rising ({debt_growth:.1%})")
    else:
        details.append("Insufficient trend data for debt cycle assessment")

    return {"score": min(10, raw_score), "details": "; ".join(details)}


def analyze_productivity_and_growth(financial_line_items: list) -> dict:
    """
    Dalio emphasizes real productivity gains as the engine of sustainable wealth.
    Measures revenue CAGR, operating margin trend, and EPS growth.
    Max 9 raw points scaled to 0-10.
    """
    if not financial_line_items or len(financial_line_items) < 2:
        return {"score": 0, "details": "Insufficient data for productivity analysis"}

    details = []
    raw_score = 0  # max 9

    # 1. Revenue CAGR (up to 3 pts)
    revenues = [fi.revenue for fi in financial_line_items if fi.revenue is not None]
    if len(revenues) >= 2 and revenues[-1] and revenues[-1] > 0:
        num_years = len(revenues) - 1
        rev_cagr = (revenues[0] / revenues[-1]) ** (1 / num_years) - 1
        if rev_cagr > 0.10:
            raw_score += 3
            details.append(f"Strong real revenue CAGR: {rev_cagr:.1%}")
        elif rev_cagr > 0.05:
            raw_score += 2
            details.append(f"Moderate revenue CAGR: {rev_cagr:.1%}")
        elif rev_cagr > 0.01:
            raw_score += 1
            details.append(f"Low revenue CAGR: {rev_cagr:.1%}")
        else:
            details.append(f"Stagnant/declining revenue CAGR: {rev_cagr:.1%}")
    else:
        details.append("Revenue trend data unavailable")

    # 2. Operating Margin Trend (up to 3 pts)
    op_incomes = [fi.operating_income for fi in financial_line_items if fi.operating_income is not None]
    rev_for_margin = [fi.revenue for fi in financial_line_items if fi.revenue is not None]

    if len(op_incomes) >= 2 and len(rev_for_margin) >= 2:
        margins = []
        for i in range(min(len(op_incomes), len(rev_for_margin))):
            if rev_for_margin[i] and rev_for_margin[i] > 0:
                margins.append(op_incomes[i] / rev_for_margin[i])

        if len(margins) >= 2:
            latest_margin = margins[0]
            oldest_margin = margins[-1]
            margin_delta = latest_margin - oldest_margin

            if latest_margin > 0.15 and margin_delta >= 0:
                raw_score += 3
                details.append(f"High and expanding operating margin: {latest_margin:.1%} (+{margin_delta:.1%} trend)")
            elif latest_margin > 0.10:
                raw_score += 2
                details.append(f"Solid operating margin: {latest_margin:.1%} (trend: {margin_delta:+.1%})")
            elif latest_margin > 0.05:
                raw_score += 1
                details.append(f"Thin operating margin: {latest_margin:.1%}")
            else:
                details.append(f"Weak operating margin: {latest_margin:.1%}")
        else:
            details.append("Margin calculation data insufficient")
    else:
        details.append("Operating margin trend data unavailable")

    # 3. EPS Growth (up to 3 pts)
    eps_values = [fi.earnings_per_share for fi in financial_line_items if fi.earnings_per_share is not None]
    if len(eps_values) >= 2 and eps_values[-1] and eps_values[-1] > 0 and eps_values[0] > 0:
        num_years = len(eps_values) - 1
        eps_cagr = (eps_values[0] / eps_values[-1]) ** (1 / num_years) - 1
        if eps_cagr > 0.12:
            raw_score += 3
            details.append(f"Strong EPS CAGR: {eps_cagr:.1%}")
        elif eps_cagr > 0.06:
            raw_score += 2
            details.append(f"Moderate EPS CAGR: {eps_cagr:.1%}")
        elif eps_cagr > 0.01:
            raw_score += 1
            details.append(f"Slight EPS growth CAGR: {eps_cagr:.1%}")
        else:
            details.append(f"EPS stagnation or decline CAGR: {eps_cagr:.1%}")
    else:
        details.append("EPS trend data unavailable or negative base")

    final_score = min(10, (raw_score / 9) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_balance_sheet_resilience(financial_line_items: list) -> dict:
    """
    Dalio stresses survival through downturns — companies must have cash buffers
    and manageable liabilities to weather deleveraging cycles.
    Max 8 raw points scaled to 0-10.
    """
    if not financial_line_items:
        return {"score": 0, "details": "Insufficient data for balance sheet analysis"}

    details = []
    raw_score = 0  # max 8

    # 1. Cash-to-Debt ratio (up to 3 pts)
    cash_values = [fi.cash_and_equivalents for fi in financial_line_items if fi.cash_and_equivalents is not None]
    debt_values = [fi.total_debt for fi in financial_line_items if fi.total_debt is not None]

    if cash_values and debt_values:
        recent_cash = cash_values[0]
        recent_debt = debt_values[0] if debt_values[0] else 1e-9
        cash_to_debt = recent_cash / recent_debt if recent_debt > 0 else float("inf")

        if cash_to_debt > 1.0:
            raw_score += 3
            details.append(f"Cash exceeds debt ({cash_to_debt:.2f}x) — strong liquidity buffer")
        elif cash_to_debt > 0.5:
            raw_score += 2
            details.append(f"Adequate cash-to-debt ratio: {cash_to_debt:.2f}x")
        elif cash_to_debt > 0.2:
            raw_score += 1
            details.append(f"Modest cash buffer vs. debt: {cash_to_debt:.2f}x")
        else:
            details.append(f"Low cash-to-debt ratio: {cash_to_debt:.2f}x — vulnerable in downturns")
    else:
        details.append("Cash/debt data unavailable")

    # 2. Liabilities-to-Assets ratio (up to 3 pts)
    assets_values = [fi.total_assets for fi in financial_line_items if fi.total_assets is not None]
    liabilities_values = [fi.total_liabilities for fi in financial_line_items if fi.total_liabilities is not None]

    if assets_values and liabilities_values:
        recent_assets = assets_values[0]
        recent_liabilities = liabilities_values[0]
        if recent_assets > 0:
            liab_to_assets = recent_liabilities / recent_assets
            if liab_to_assets < 0.4:
                raw_score += 3
                details.append(f"Conservative liabilities-to-assets: {liab_to_assets:.2f} — asset-rich balance sheet")
            elif liab_to_assets < 0.6:
                raw_score += 2
                details.append(f"Moderate liabilities-to-assets: {liab_to_assets:.2f}")
            elif liab_to_assets < 0.8:
                raw_score += 1
                details.append(f"Elevated liabilities-to-assets: {liab_to_assets:.2f}")
            else:
                details.append(f"High liabilities-to-assets: {liab_to_assets:.2f} — limited buffer in stress scenarios")
    else:
        details.append("Assets/liabilities data unavailable")

    # 3. FCF Consistency (up to 2 pts)
    fcf_values = [fi.free_cash_flow for fi in financial_line_items if fi.free_cash_flow is not None]

    if len(fcf_values) >= 3:
        positive_fcf_count = sum(1 for f in fcf_values if f > 0)
        if positive_fcf_count == len(fcf_values):
            raw_score += 2
            details.append(f"Consistently positive FCF across all {len(fcf_values)} periods")
        elif positive_fcf_count >= len(fcf_values) * 0.7:
            raw_score += 1
            details.append(f"Mostly positive FCF: {positive_fcf_count}/{len(fcf_values)} periods")
        else:
            details.append(f"Inconsistent FCF: only {positive_fcf_count}/{len(fcf_values)} periods positive")
    else:
        details.append("Insufficient FCF history for consistency check")

    final_score = min(10, (raw_score / 8) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_all_weather_characteristics(financial_line_items: list, prices: list) -> dict:
    """
    Dalio's All Weather approach seeks assets that perform across economic environments.
    Proxies: gross margin stability (pricing power), FCF yield, and price volatility.
    Max 8 raw points scaled to 0-10.
    """
    if not financial_line_items:
        return {"score": 0, "details": "Insufficient data for all-weather analysis"}

    details = []
    raw_score = 0  # max 8

    # 1. Gross Margin Stability — pricing power proxy (up to 3 pts)
    gross_profits = [fi.gross_profit for fi in financial_line_items if fi.gross_profit is not None]
    revenues = [fi.revenue for fi in financial_line_items if fi.revenue is not None]

    gross_margins = []
    for i in range(min(len(gross_profits), len(revenues))):
        if revenues[i] and revenues[i] > 0:
            gross_margins.append(gross_profits[i] / revenues[i])

    if len(gross_margins) >= 2:
        avg_margin = statistics.mean(gross_margins)
        margin_stdev = statistics.pstdev(gross_margins) if len(gross_margins) > 1 else 0

        if avg_margin > 0.40 and margin_stdev < 0.05:
            raw_score += 3
            details.append(f"High, stable gross margins: avg {avg_margin:.1%} (σ={margin_stdev:.1%}) — strong pricing power")
        elif avg_margin > 0.25 and margin_stdev < 0.08:
            raw_score += 2
            details.append(f"Decent gross margins: avg {avg_margin:.1%} (σ={margin_stdev:.1%})")
        elif avg_margin > 0.15:
            raw_score += 1
            details.append(f"Thin gross margins: avg {avg_margin:.1%} — limited inflation protection")
        else:
            details.append(f"Low gross margins: avg {avg_margin:.1%} — poor pricing power")
    else:
        details.append("Gross margin data insufficient")

    # 2. FCF Yield (up to 3 pts) — market_cap passed via financial_line_items context
    fcf_values = [fi.free_cash_flow for fi in financial_line_items if fi.free_cash_flow is not None]

    if fcf_values and prices:
        # Estimate market cap from latest price × shares outstanding
        shares_list = [fi.outstanding_shares for fi in financial_line_items if fi.outstanding_shares is not None]
        if shares_list and prices:
            sorted_prices = sorted(prices, key=lambda p: p.time)
            latest_price = sorted_prices[-1].close if sorted_prices else None
            if latest_price and shares_list[0]:
                implied_mkt_cap = latest_price * shares_list[0]
                recent_fcf = fcf_values[0]
                if implied_mkt_cap > 0 and recent_fcf > 0:
                    fcf_yield = recent_fcf / implied_mkt_cap
                    if fcf_yield > 0.06:
                        raw_score += 3
                        details.append(f"High FCF yield: {fcf_yield:.1%} — attractive across all macro environments")
                    elif fcf_yield > 0.03:
                        raw_score += 2
                        details.append(f"Moderate FCF yield: {fcf_yield:.1%}")
                    elif fcf_yield > 0.01:
                        raw_score += 1
                        details.append(f"Low FCF yield: {fcf_yield:.1%}")
                    else:
                        details.append(f"Very low/negative FCF yield: {fcf_yield:.1%}")
                else:
                    details.append("FCF yield calculation unavailable (negative FCF or no price)")
            else:
                details.append("Shares outstanding or price data unavailable for FCF yield")
        else:
            details.append("Shares data unavailable for FCF yield")
    else:
        details.append("FCF or price data unavailable for FCF yield")

    # 3. Price Volatility — low volatility preferred for all-weather (up to 2 pts)
    if prices and len(prices) > 20:
        sorted_prices = sorted(prices, key=lambda p: p.time)
        close_prices = [p.close for p in sorted_prices if p.close is not None]
        if len(close_prices) > 10:
            daily_returns = [
                (close_prices[i] - close_prices[i - 1]) / close_prices[i - 1]
                for i in range(1, len(close_prices))
                if close_prices[i - 1] > 0
            ]
            if daily_returns:
                vol = statistics.pstdev(daily_returns)
                if vol < 0.015:
                    raw_score += 2
                    details.append(f"Low price volatility: {vol:.2%} annualized daily stdev — all-weather resilience")
                elif vol < 0.025:
                    raw_score += 1
                    details.append(f"Moderate price volatility: {vol:.2%}")
                else:
                    details.append(f"High price volatility: {vol:.2%} — cyclical or speculative behavior")
    else:
        details.append("Insufficient price data for volatility assessment")

    final_score = min(10, (raw_score / 8) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_macro_cycle_positioning(
    financial_line_items: list,
    market_cap: float | None,
    company_news: list,
) -> dict:
    """
    Assesses how well the company is positioned in the current macro environment.
    Considers valuation (P/FCF, EV/EBITDA), news sentiment, and earnings quality.
    Max 8 raw points scaled to 0-10.
    """
    if not financial_line_items:
        return {"score": 0, "details": "Insufficient data for macro cycle analysis"}

    details = []
    raw_score = 0  # max 8

    # 1. EV/EBITDA — macro-aware valuation (up to 3 pts)
    ebitda_values = [fi.ebitda for fi in financial_line_items if fi.ebitda is not None]
    debt_values = [fi.total_debt for fi in financial_line_items if fi.total_debt is not None]
    cash_values = [fi.cash_and_equivalents for fi in financial_line_items if fi.cash_and_equivalents is not None]

    if market_cap and ebitda_values:
        recent_ebitda = ebitda_values[0]
        recent_debt = debt_values[0] if debt_values else 0
        recent_cash = cash_values[0] if cash_values else 0
        enterprise_value = market_cap + recent_debt - recent_cash

        if enterprise_value > 0 and recent_ebitda and recent_ebitda > 0:
            ev_ebitda = enterprise_value / recent_ebitda
            if ev_ebitda < 10:
                raw_score += 3
                details.append(f"Attractive EV/EBITDA: {ev_ebitda:.1f}x — undervalued relative to cycle")
            elif ev_ebitda < 16:
                raw_score += 2
                details.append(f"Fair EV/EBITDA: {ev_ebitda:.1f}x")
            elif ev_ebitda < 25:
                raw_score += 1
                details.append(f"Full EV/EBITDA: {ev_ebitda:.1f}x — limited margin of safety")
            else:
                details.append(f"Expensive EV/EBITDA: {ev_ebitda:.1f}x — vulnerable to macro compression")
        else:
            details.append("EV/EBITDA unavailable")
    else:
        details.append("Market cap or EBITDA unavailable for valuation")

    # 2. P/FCF — another Dalio-preferred measure (up to 3 pts)
    fcf_values = [fi.free_cash_flow for fi in financial_line_items if fi.free_cash_flow is not None]

    if market_cap and fcf_values:
        recent_fcf = fcf_values[0]
        if recent_fcf and recent_fcf > 0:
            p_fcf = market_cap / recent_fcf
            if p_fcf < 15:
                raw_score += 3
                details.append(f"Compelling P/FCF: {p_fcf:.1f}x — strong real return potential")
            elif p_fcf < 25:
                raw_score += 2
                details.append(f"Reasonable P/FCF: {p_fcf:.1f}x")
            elif p_fcf < 40:
                raw_score += 1
                details.append(f"Elevated P/FCF: {p_fcf:.1f}x")
            else:
                details.append(f"Very high P/FCF: {p_fcf:.1f}x — priced for perfection")
        else:
            details.append("Negative or zero FCF — no P/FCF calculation")
    else:
        details.append("P/FCF data unavailable")

    # 3. News Sentiment — macro narrative awareness (up to 2 pts)
    if company_news:
        macro_risk_keywords = [
            "recession", "inflation", "interest rate", "fed", "monetary", "debt crisis",
            "bankruptcy", "default", "downgrade", "tariff", "geopolitical"
        ]
        positive_keywords = ["growth", "expansion", "record", "beat", "upgrade", "innovation", "partnership"]

        macro_risk_count = 0
        positive_count = 0
        for news in company_news:
            title_lower = (news.title or "").lower()
            if any(kw in title_lower for kw in macro_risk_keywords):
                macro_risk_count += 1
            if any(kw in title_lower for kw in positive_keywords):
                positive_count += 1

        if macro_risk_count == 0 and positive_count > len(company_news) * 0.2:
            raw_score += 2
            details.append(f"Positive macro narrative: {positive_count} positive vs {macro_risk_count} macro risk headlines")
        elif macro_risk_count < len(company_news) * 0.2:
            raw_score += 1
            details.append(f"Benign macro backdrop: low macro risk headlines ({macro_risk_count}/{len(company_news)})")
        else:
            details.append(f"Elevated macro risk signals in news: {macro_risk_count}/{len(company_news)} headlines")
    else:
        details.append("No news data for macro narrative assessment")

    final_score = min(10, (raw_score / 8) * 10)
    return {"score": final_score, "details": "; ".join(details)}


def generate_dalio_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
) -> RayDalioSignal:
    """Generates a JSON investment signal in the style of Ray Dalio."""
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Ray Dalio AI agent, making investment decisions using his principles:

1. Understand the debt cycle: unsustainable debt growth relative to income is the primary long-term risk.
2. Seek companies producing real productivity gains — sustainable wealth comes from real output.
3. Require balance sheet resilience — companies must survive downturns and deleveraging cycles.
4. Prioritize all-weather characteristics: strong pricing power, consistent FCF, low cyclicality.
5. Apply macro-aware valuation — price paid matters enormously in the context of the current cycle.
6. Diversify across uncorrelated drivers of return; avoid concentrated bets on a single outcome.
7. Be radically open-minded: weigh evidence, not conviction, and update views when facts change.

Decision rules:
- Strongly favor companies with income growing faster than debt (virtuous debt cycle).
- Penalize highly leveraged businesses with weak interest coverage — they fail in downturns.
- Reward durable pricing power (stable gross margins) as protection against inflation regimes.
- Value FCF yield and EV/EBITDA highly — these reflect real economic return.
- Be cautious of companies priced for perfection when macro conditions may tighten.
- Output a JSON object with signal, confidence, and detailed reasoning.

When providing reasoning, be specific:
1. Cite the debt cycle metrics and whether leverage is sustainable
2. Explain productivity and real growth quality
3. Assess balance sheet resilience for an economic downturn scenario
4. Comment on all-weather characteristics — inflation protection and FCF durability
5. Provide macro cycle valuation context
6. Use Ray Dalio's measured, principle-driven, and macro-aware voice

Example bullish: "The company exemplifies Dalio's virtuous debt cycle — net income CAGR of 14% over 5 years while total debt grew only 6%, improving the debt-to-income ratio meaningfully. Balance sheet carries 1.4x cash-to-debt coverage and EBIT covers interest 18x, providing resilience through any plausible macro downturn. Gross margins have held at 62% ± 2% across five years, demonstrating true pricing power. At 12x EV/EBITDA and 5.2% FCF yield, the valuation reflects appropriate respect for cycle risk..."
Example bearish: "Debt has grown at 22% CAGR while net income has been flat — a classic late-cycle leverage pattern Dalio warns against. Interest coverage at 2.3x provides little cushion in a contractionary environment. Gross margins have compressed 800bps, eroding the inflation protection that would otherwise sustain performance. At 32x EV/EBITDA with negative FCF, the market is pricing in a scenario that requires continued debt expansion to fund operations — a fragile configuration..."
""",
            ),
            (
                "human",
                """Based on the following analysis, create a Ray Dalio-style investment signal.

Analysis Data for {ticker}:
{analysis_data}

Return the trading signal in this JSON format:
{{
  "signal": "bullish/bearish/neutral",
  "confidence": float (0-100),
  "reasoning": "string"
}}
""",
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def create_default_signal():
        return RayDalioSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=RayDalioSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_signal,
    )
