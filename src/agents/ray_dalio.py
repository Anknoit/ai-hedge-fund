from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.tools.api import get_financial_metrics, get_market_data, get_macro_indicators, get_interest_rates
from src.utils.llm import call_llm
from src.utils.progress import progress


class RayDalioSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str
    regime: Literal["rising_growth", "falling_growth", "rising_inflation", "falling_inflation", "stagflation", "goldilocks"]
    correlation_risk: float  # 0-1 how correlated with other risks
    stress_test_result: Literal["pass", "caution", "fail"]


def ray_dalio_agent(state: AgentState, agent_id: str = "ray_dalio_agent"):
    """Analyzes investments using Ray Dalio's principles-based, macro-driven approach."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Collect all analysis for LLM reasoning
    analysis_data = {}
    dalio_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching macro environment data")
        
        # First get macro context (Dalio's top-down approach)
        macro_data = get_macro_indicators(end_date, [
            "debt_to_gdp",
            "inflation_rate",
            "unemployment_rate",
            "gdp_growth",
            "central_bank_balance_sheet",
            "yield_curve",
            "credit_spreads"
        ])
        
        progress.update_status(agent_id, ticker, "Getting interest rate environment")
        interest_rates = get_interest_rates(end_date)
        
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=8)
        
        progress.update_status(agent_id, ticker, "Getting market data for regime analysis")
        market_data = get_market_data(ticker, end_date, [
            "beta",
            "volatility",
            "correlation_sp500",
            "correlation_bonds",
            "liquidity_metrics"
        ])

        progress.update_status(agent_id, ticker, "Analyzing macro regime")
        regime_analysis = analyze_macro_regime(macro_data, interest_rates)
        
        progress.update_status(agent_id, ticker, "Analyzing debt cycle position")
        debt_cycle_analysis = analyze_debt_cycle(macro_data, metrics)
        
        progress.update_status(agent_id, ticker, "Analyzing correlation risk")
        correlation_analysis = analyze_correlation_risk(market_data, macro_data)
        
        progress.update_status(agent_id, ticker, "Analyzing leverage vulnerability")
        leverage_analysis = analyze_leverage_vulnerability(metrics, macro_data)
        
        progress.update_status(agent_id, ticker, "Stress testing assumptions")
        stress_test = conduct_stress_test(metrics, macro_data, regime_analysis)
        
        progress.update_status(agent_id, ticker, "Analyzing regime alignment")
        regime_alignment = analyze_regime_alignment(ticker, regime_analysis, metrics, market_data)
        
        progress.update_status(agent_id, ticker, "Analyzing behavioral factors")
        behavioral_analysis = analyze_behavioral_factors(market_data, macro_data)

        # Combine all analysis results for LLM evaluation
        analysis_data[ticker] = {
            "ticker": ticker,
            "regime_analysis": regime_analysis,
            "debt_cycle_analysis": debt_cycle_analysis,
            "correlation_analysis": correlation_analysis,
            "leverage_analysis": leverage_analysis,
            "stress_test": stress_test,
            "regime_alignment": regime_alignment,
            "behavioral_analysis": behavioral_analysis,
            "macro_data": macro_data,
            "interest_rates": interest_rates,
            "market_data": market_data,
            "financial_metrics": [m.model_dump() for m in metrics] if metrics else []
        }

        progress.update_status(agent_id, ticker, "Generating Ray Dalio analysis")
        dalio_output = generate_dalio_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        # Store analysis in consistent format with other agents
        dalio_analysis[ticker] = {
            "signal": dalio_output.signal,
            "confidence": dalio_output.confidence,
            "reasoning": dalio_output.reasoning,
            "regime": dalio_output.regime,
            "correlation_risk": dalio_output.correlation_risk,
            "stress_test_result": dalio_output.stress_test_result
        }

        progress.update_status(agent_id, ticker, "Done", analysis=dalio_output.reasoning)

    # Create the message
    message = HumanMessage(content=json.dumps(dalio_analysis), name=agent_id)

    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(dalio_analysis, agent_id)

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = dalio_analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


def analyze_macro_regime(macro_data: dict, interest_rates: dict) -> dict[str, any]:
    """Determine current macroeconomic regime based on Dalio's framework."""
    reasoning = []
    regime_scores = {
        "rising_growth": 0,
        "falling_growth": 0,
        "rising_inflation": 0,
        "falling_inflation": 0,
        "stagflation": 0,
        "goldilocks": 0
    }
    
    # Analyze growth conditions
    gdp_growth = macro_data.get("gdp_growth", 0)
    unemployment = macro_data.get("unemployment_rate", 5)
    
    if gdp_growth > 3 and unemployment < 5:
        regime_scores["rising_growth"] += 3
        reasoning.append(f"Strong growth: GDP {gdp_growth}%, unemployment {unemployment}%")
    elif gdp_growth < 1 or unemployment > 7:
        regime_scores["falling_growth"] += 3
        reasoning.append(f"Weak growth: GDP {gdp_growth}%, unemployment {unemployment}%")
    
    # Analyze inflation conditions
    inflation = macro_data.get("inflation_rate", 2)
    if inflation > 5:
        regime_scores["rising_inflation"] += 3
        reasoning.append(f"High inflation: {inflation}%")
    elif inflation < 1:
        regime_scores["falling_inflation"] += 2
        reasoning.append(f"Low inflation: {inflation}%")
    
    # Check for stagflation (high inflation + slow growth)
    if inflation > 4 and gdp_growth < 2:
        regime_scores["stagflation"] += 4
        reasoning.append(f"Stagflation risk: high inflation with slow growth")
    
    # Check for goldilocks (moderate growth + low inflation)
    if 2 < gdp_growth < 4 and 1 < inflation < 3:
        regime_scores["goldilocks"] += 3
        reasoning.append(f"Goldilocks environment: balanced growth and inflation")
    
    # Analyze yield curve
    yield_curve = macro_data.get("yield_curve", 0)
    if yield_curve < 0:
        regime_scores["falling_growth"] += 2
        reasoning.append(f"Inverted yield curve signals potential recession")
    
    # Determine dominant regime
    dominant_regime = max(regime_scores, key=regime_scores.get)
    
    return {
        "regime": dominant_regime,
        "scores": regime_scores,
        "details": "; ".join(reasoning),
        "key_indicators": {
            "gdp_growth": gdp_growth,
            "inflation": inflation,
            "unemployment": unemployment,
            "yield_curve": yield_curve
        }
    }


def analyze_debt_cycle(macro_data: dict, metrics: list) -> dict[str, any]:
    """Analyze position in debt cycle and vulnerability."""
    reasoning = []
    vulnerability_score = 0  # 0-10, higher = more vulnerable
    
    # Macro debt analysis
    debt_to_gdp = macro_data.get("debt_to_gdp", 100)
    if debt_to_gdp > 100:
        vulnerability_score += 3
        reasoning.append(f"High national debt-to-GDP: {debt_to_gdp}%")
    elif debt_to_gdp > 80:
        vulnerability_score += 2
        reasoning.append(f"Elevated national debt: {debt_to_gdp}%")
    
    # Corporate debt analysis
    if metrics and len(metrics) > 0:
        latest = metrics[0]
        if hasattr(latest, 'debt_to_equity') and latest.debt_to_equity:
            if latest.debt_to_equity > 1.5:
                vulnerability_score += 3
                reasoning.append(f"High corporate leverage: D/E {latest.debt_to_equity:.1f}")
            elif latest.debt_to_equity > 1.0:
                vulnerability_score += 2
                reasoning.append(f"Elevated corporate leverage: D/E {latest.debt_to_equity:.1f}")
    
    # Credit spread analysis
    credit_spreads = macro_data.get("credit_spreads", 0)
    if credit_spreads > 3:
        vulnerability_score += 2
        reasoning.append(f"Widening credit spreads: {credit_spreads}% - indicates rising default risk")
    
    # Central bank policy
    cb_balance_sheet = macro_data.get("central_bank_balance_sheet_change", 0)
    if cb_balance_sheet < -5:  # Quantitative tightening
        vulnerability_score += 1
        reasoning.append("Central bank tightening liquidity")
    
    # Determine cycle position
    if vulnerability_score >= 6:
        cycle_position = "late_cycle_vulnerable"
    elif vulnerability_score >= 3:
        cycle_position = "mid_cycle"
    else:
        cycle_position = "early_cycle"
    
    return {
        "cycle_position": cycle_position,
        "vulnerability_score": vulnerability_score,
        "debt_to_gdp": debt_to_gdp,
        "details": "; ".join(reasoning)
    }


def analyze_correlation_risk(market_data: dict, macro_data: dict) -> dict[str, any]:
    """Analyze correlation risk for portfolio diversification."""
    reasoning = []
    correlation_risk = 0.5  # Default moderate risk
    
    # Market correlations
    corr_sp500 = market_data.get("correlation_sp500", 0)
    corr_bonds = market_data.get("correlation_bonds", 0)
    
    if abs(corr_sp500) > 0.8:
        correlation_risk += 0.2
        reasoning.append(f"High correlation with S&P 500: {corr_sp500:.2f}")
    
    if abs(corr_bonds) > 0.7:
        correlation_risk += 0.15
        reasoning.append(f"Significant correlation with bonds: {corr_bonds:.2f}")
    
    # Beta analysis
    beta = market_data.get("beta", 1.0)
    if beta > 1.5:
        correlation_risk += 0.1
        reasoning.append(f"High beta: {beta:.2f} - amplifies market moves")
    elif beta < 0.5:
        correlation_risk -= 0.1
        reasoning.append(f"Low beta: {beta:.2f} - provides diversification")
    
    # Macro sensitivity
    interest_sensitivity = analyze_interest_rate_sensitivity(market_data, macro_data)
    if interest_sensitivity == "high":
        correlation_risk += 0.15
        reasoning.append("High sensitivity to interest rate changes")
    
    # Normalize to 0-1 range
    correlation_risk = max(0, min(1, correlation_risk))
    
    return {
        "correlation_risk": correlation_risk,
        "correlation_sp500": corr_sp500,
        "correlation_bonds": corr_bonds,
        "beta": beta,
        "details": "; ".join(reasoning) if reasoning else "Moderate correlation risk"
    }


def analyze_interest_rate_sensitivity(market_data: dict, macro_data: dict) -> str:
    """Determine sensitivity to interest rate changes."""
    # Simplified analysis - in production would use duration, sector analysis, etc.
    sector = market_data.get("sector", "")
    
    rate_sensitive_sectors = [
        "real_estate", "utilities", "financials", 
        "telecom", "consumer_discretionary"
    ]
    
    rate_insensitive_sectors = [
        "healthcare", "consumer_staples", 
        "energy", "materials"
    ]
    
    if sector in rate_sensitive_sectors:
        return "high"
    elif sector in rate_insensitive_sectors:
        return "low"
    else:
        return "medium"


def analyze_leverage_vulnerability(metrics: list, macro_data: dict) -> dict[str, any]:
    """Analyze leverage vulnerability in current macro environment."""
    reasoning = []
    vulnerability_score = 0
    max_score = 10
    
    if not metrics or len(metrics) == 0:
        return {
            "vulnerability_score": 0,
            "details": "Insufficient data for leverage analysis",
            "leverage_rating": "unknown"
        }
    
    latest = metrics[0]
    
    # Debt metrics
    if hasattr(latest, 'debt_to_equity') and latest.debt_to_equity:
        if latest.debt_to_equity > 2.0:
            vulnerability_score += 4
            reasoning.append(f"Extremely high debt-to-equity: {latest.debt_to_equity:.1f}")
        elif latest.debt_to_equity > 1.0:
            vulnerability_score += 2
            reasoning.append(f"High debt-to-equity: {latest.debt_to_equity:.1f}")
    
    # Interest coverage
    if hasattr(latest, 'interest_coverage') and latest.interest_coverage:
        if latest.interest_coverage < 3:
            vulnerability_score += 3
            reasoning.append(f"Low interest coverage: {latest.interest_coverage:.1f}x")
        elif latest.interest_coverage < 5:
            vulnerability_score += 1
            reasoning.append(f"Modest interest coverage: {latest.interest_coverage:.1f}x")
    
    # Current ratio (liquidity)
    if hasattr(latest, 'current_ratio') and latest.current_ratio:
        if latest.current_ratio < 1.0:
            vulnerability_score += 2
            reasoning.append(f"Poor liquidity: current ratio {latest.current_ratio:.1f}")
    
    # Debt maturity profile (simplified)
    short_term_debt_ratio = getattr(latest, 'short_term_debt_ratio', 0.3)
    if short_term_debt_ratio > 0.5:
        vulnerability_score += 1
        reasoning.append(f"High short-term debt exposure: {short_term_debt_ratio:.0%}")
    
    # Interest rate environment sensitivity
    rates_rising = macro_data.get("interest_rate_trend", "stable") == "rising"
    if rates_rising and vulnerability_score > 3:
        vulnerability_score += 1
        reasoning.append("Vulnerable in rising rate environment")
    
    # Determine leverage rating
    if vulnerability_score >= 6:
        leverage_rating = "high_risk"
    elif vulnerability_score >= 3:
        leverage_rating = "moderate_risk"
    else:
        leverage_rating = "low_risk"
    
    return {
        "vulnerability_score": vulnerability_score,
        "max_score": max_score,
        "leverage_rating": leverage_rating,
        "details": "; ".join(reasoning) if reasoning else "Moderate leverage risk"
    }


def conduct_stress_test(metrics: list, macro_data: dict, regime_analysis: dict) -> dict[str, any]:
    """Stress test investment under various adverse scenarios."""
    scenarios = {
        "recession": {"growth_change": -3, "rates_change": -2},
        "stagflation": {"growth_change": -2, "inflation_change": 4},
        "rate_hike": {"rates_change": 3, "growth_change": -1},
        "deflation": {"inflation_change": -3, "growth_change": -2},
        "liquidity_crisis": {"credit_spread_change": 5, "rates_change": 2}
    }
    
    results = {}
    worst_case = None
    worst_performance = 0
    
    for scenario, assumptions in scenarios.items():
        # Simplified stress test - in production would use detailed financial modeling
        performance_impact = 0
        
        # Interest rate sensitivity
        if assumptions.get("rates_change", 0) != 0:
            rate_sens = analyze_interest_rate_sensitivity({}, macro_data)
            if rate_sens == "high":
                performance_impact -= abs(assumptions["rates_change"]) * 0.15
        
        # Growth sensitivity
        if assumptions.get("growth_change", 0) < 0:
            # Assume cyclical companies suffer more
            if hasattr(metrics[0], 'operating_margin') and metrics[0].operating_margin < 0.1:
                performance_impact += assumptions["growth_change"] * 0.2
        
        # Leverage impact
        leverage_score = analyze_leverage_vulnerability(metrics, macro_data)["vulnerability_score"]
        performance_impact -= leverage_score * 0.05
        
        results[scenario] = {
            "performance_impact": performance_impact,
            "survival_likelihood": "high" if performance_impact > -0.3 else "medium" if performance_impact > -0.5 else "low"
        }
        
        if performance_impact < worst_performance:
            worst_performance = performance_impact
            worst_case = scenario
    
    # Overall stress test result
    if worst_performance > -0.2:
        overall_result = "pass"
    elif worst_performance > -0.4:
        overall_result = "caution"
    else:
        overall_result = "fail"
    
    return {
        "overall_result": overall_result,
        "worst_case_scenario": worst_case,
        "worst_case_impact": worst_performance,
        "scenario_results": results,
        "details": f"Stress test {overall_result}: worst case {worst_case} with {worst_performance:.1%} impact"
    }


def analyze_regime_alignment(ticker: str, regime_analysis: dict, metrics: list, market_data: dict) -> dict[str, any]:
    """Analyze how well the asset aligns with current and expected regimes."""
    current_regime = regime_analysis["regime"]
    reasoning = []
    alignment_score = 0  # -2 to +2
    
    # Sector-based regime alignment
    sector = market_data.get("sector", "")
    
    # Define which sectors perform well in which regimes
    regime_sector_alignment = {
        "rising_growth": ["technology", "consumer_discretionary", "financials"],
        "falling_growth": ["consumer_staples", "utilities", "healthcare"],
        "rising_inflation": ["energy", "materials", "real_estate"],
        "falling_inflation": ["technology", "consumer_discretionary"],
        "stagflation": ["energy", "consumer_staples"],
        "goldilocks": ["technology", "healthcare", "consumer_discretionary"]
    }
    
    if sector in regime_sector_alignment.get(current_regime, []):
        alignment_score += 1
        reasoning.append(f"Sector {sector} typically performs well in {current_regime} regime")
    else:
        alignment_score -= 0.5
        reasoning.append(f"Sector {sector} may underperform in {current_regime} regime")
    
    # Financial characteristic alignment
    if metrics and len(metrics) > 0:
        latest = metrics[0]
        
        # High growth companies in rising growth regime
        if current_regime == "rising_growth":
            if hasattr(latest, 'revenue_growth') and latest.revenue_growth > 0.15:
                alignment_score += 1
                reasoning.append("High growth characteristics align with rising growth regime")
        
        # Defensive characteristics in falling growth
        elif current_regime == "falling_growth":
            if hasattr(latest, 'operating_margin') and latest.operating_margin > 0.2:
                alignment_score += 0.5
                reasoning.append("Strong margins provide defense in falling growth")
        
        # Inflation hedge characteristics
        elif current_regime in ["rising_inflation", "stagflation"]:
            if hasattr(latest, 'pricing_power_score') and getattr(latest, 'pricing_power_score', 0) > 0.7:
                alignment_score += 1
                reasoning.append("Pricing power provides inflation protection")
    
    # Beta alignment
    beta = market_data.get("beta", 1.0)
    if current_regime == "rising_growth" and beta > 1.0:
        alignment_score += 0.5
        reasoning.append(f"High beta ({beta:.1f}) amplifies gains in rising growth")
    elif current_regime == "falling_growth" and beta < 1.0:
        alignment_score += 0.5
        reasoning.append(f"Low beta ({beta:.1f}) provides defense in falling growth")
    
    return {
        "alignment_score": alignment_score,
        "current_regime": current_regime,
        "sector_alignment": sector in regime_sector_alignment.get(current_regime, []),
        "details": "; ".join(reasoning) if reasoning else "Neutral regime alignment"
    }


def analyze_behavioral_factors(market_data: dict, macro_data: dict) -> dict[str, any]:
    """Analyze behavioral factors like crowding, narrative dominance, leverage buildup."""
    reasoning = []
    behavioral_risk = 0.5  # 0-1
    
    # Crowding indicators
    volatility = market_data.get("volatility", 0.2)
    if volatility < 0.15:
        behavioral_risk += 0.1
        reasoning.append(f"Low volatility ({volatility:.1%}) may indicate complacency")
    
    # Valuation extremes (simplified)
    pe_ratio = market_data.get("pe_ratio", 20)
    if pe_ratio > 25:
        behavioral_risk += 0.15
        reasoning.append(f"Elevated valuation (P/E: {pe_ratio}) suggests optimism")
    elif pe_ratio < 10:
        behavioral_risk -= 0.1
        reasoning.append(f"Depressed valuation (P/E: {pe_ratio}) suggests pessimism")
    
    # Sentiment indicators
    put_call_ratio = market_data.get("put_call_ratio", 0.7)
    if put_call_ratio < 0.6:
        behavioral_risk += 0.1
        reasoning.append(f"Low put/call ratio ({put_call_ratio:.2f}) indicates bullish sentiment")
    
    # Leverage in system
    credit_growth = macro_data.get("credit_growth", 5)
    if credit_growth > 10:
        behavioral_risk += 0.15
        reasoning.append(f"Rapid credit growth ({credit_growth}%) indicates leverage buildup")
    
    # Normalize to 0-1
    behavioral_risk = max(0, min(1, behavioral_risk))
    
    return {
        "behavioral_risk": behavioral_risk,
        "market_sentiment": "bullish" if behavioral_risk > 0.6 else "bearish" if behavioral_risk < 0.4 else "neutral",
        "details": "; ".join(reasoning) if reasoning else "Moderate behavioral risks"
    }


def generate_dalio_output(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
    agent_id: str = "ray_dalio_agent",
) -> RayDalioSignal:
    """Get investment decision from LLM with Dalio's principles-based approach"""
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are Ray Dalio, founder of Bridgewater Associates. Apply my principles-based, systematic investment approach to analyze this opportunity.

                MY CORE PHILOSOPHY:
                1. Reality is a machine driven by cause-effect relationships: Markets are mechanisms, not mysteries.
                2. Everything happens over and over again in cycles: Debt cycles, economic cycles, political cycles.
                3. Risk is the primary variable to manage: "If you're not worried, you need to worry. And if you're worried, you don't need to worry."
                4. Diversify well across uncorrelated return streams: Balance risks, not capital.
                5. Seek radical truth and radical transparency: Don't confuse what you wish were true with what is true.

                KEY FRAMEWORKS:
                • Debt Cycle Analysis: Where are we in the short-term and long-term debt cycle?
                • Economic Machine: How do productivity, short-term debt cycle, and long-term debt cycle interact?
                • All-Weather Portfolio: Design for all economic environments (rising/falling growth, rising/falling inflation).
                • Risk Parity: Balance risk contributions, not dollar amounts.

                MACRO REGIMES & ASSET BEHAVIOR:
                1. Rising Growth: Equities (especially cyclicals), credit, EM assets perform well
                2. Falling Growth: Bonds, defensive stocks, gold, yen perform well
                3. Rising Inflation: Inflation-linked bonds, commodities, real assets, value stocks
                4. Falling Inflation: Growth stocks, long-duration bonds, tech
                5. Stagflation: Gold, commodities, cash, short-duration assets
                6. Goldilocks: Balanced portfolio, growth assets with some inflation protection

                MY INVESTMENT PROCESS:
                Step 1: Macro Diagnosis - What regime are we in? Where are we in debt cycles?
                Step 2: Asset Behavior - How should this asset perform in current and alternative regimes?
                Step 3: Correlation Analysis - Does this provide true diversification or amplify existing risks?
                Step 4: Stress Testing - What happens in adverse scenarios? Can we survive being wrong?
                Step 5: Portfolio Fit - How does this balance our overall risk exposures?

                CRITICAL QUESTIONS I ALWAYS ASK:
                1. What is priced in? What are markets expecting?
                2. Who is leveraged? Where is there fragility in the system?
                3. What happens if growth slows or rates rise unexpectedly?
                4. What's the worst-case scenario? Can we survive it?
                5. How correlated is this with our other risks?
                6. Is this compensating us adequately for the risks we're taking?

                WHAT I AVOID:
                • Investments that depend on a single macro outcome
                • Assets requiring constant refinancing in tightening cycles
                • Anything driven by hype, narrative, or "this time is different" thinking
                • Concentrated bets without hedges or diversification
                • Overconfidence in any single view
                • Ignoring debt dynamics and central bank behavior

                DECISION-MAKING STYLE:
                • Systematic and rules-based, not emotional
                • Probabilistic thinking - assign probabilities, not certainties
                • Focus on not being wrong in irreversible ways
                • Let history and data guide, not opinions
                • "Pain + reflection = progress" - learn from mistakes

                REGIME DETERMINATION LOGIC:
                First identify the current economic regime using:
                - GDP growth relative to potential
                - Inflation relative to target
                - Unemployment trends
                - Yield curve shape
                - Central bank stance
                - Credit conditions

                ASSET EVALUATION CRITERIA:
                1. How does it behave in different regimes? (most important)
                2. What are its cash flow characteristics under stress?
                3. Who owns it and why? (crowding risk)
                4. What's its sensitivity to interest rates, growth, inflation?
                5. How does it fit in an all-weather portfolio?

                CONFIDENCE LEVELS:
                - 80-100%: Excellent regime alignment, strong diversification benefits, survives stress tests
                - 60-79%: Good fit, adequate risk compensation, moderate diversification
                - 40-59%: Mixed signals, requires hedging or position sizing adjustments
                - 20-39%: Poor regime alignment, high correlation risk, fails stress tests
                - 0-19%: Avoid - either too risky, overpriced, or inadequate diversification

                Remember: The goal is not to be right, it's to not be wrong in ways that cause irreversible loss. Diversification across uncorrelated return streams is the most important thing you can do. And always, always think in terms of probabilities and prepare for alternative scenarios.
                """,
            ),
            (
                "human",
                """Analyze this investment opportunity for {ticker} using my principles-based approach:

                COMPREHENSIVE ANALYSIS DATA:
                {analysis_data}

                Please provide your investment decision in exactly this JSON format:
                {{
                  "signal": "bullish" | "bearish" | "neutral",
                  "confidence": float between 0 and 100,
                  "reasoning": "string with your detailed Ray Dalio-style analysis",
                  "regime": "rising_growth" | "falling_growth" | "rising_inflation" | "falling_inflation" | "stagflation" | "goldilocks",
                  "correlation_risk": float between 0 and 1,
                  "stress_test_result": "pass" | "caution" | "fail"
                }}

                In your reasoning, structure your analysis as follows:

                1. MACRO DIAGNOSIS:
                   - Current economic regime identification
                   - Debt cycle position analysis
                   - Central bank policy assessment

                2. ASSET BEHAVIOR ANALYSIS:
                   - How this asset should perform in current regime
                   - Performance in alternative regimes
                   - Sensitivity to key macro variables (rates, growth, inflation)

                3. RISK ASSESSMENT:
                   - Leverage vulnerability analysis
                   - Correlation risk evaluation
                   - Stress test results summary
                   - Behavioral/crowding risks

                4. PORTFOLIO FIT:
                   - Diversification benefits assessment
                   - How it balances existing risk exposures
                   - Position sizing considerations

                5. DECISION RATIONALE:
                   - Clear explanation of signal and confidence level
                   - Key risks that could make you wrong
                   - What would change your view

                Write as Ray Dalio would speak - systematic, principles-based, with focus on risk and diversification. Reference historical parallels and economic mechanics.
                """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    # Default fallback signal in case parsing fails
    def create_default_ray_dalio_signal():
        return RayDalioSignal(
            signal="neutral", 
            confidence=0.0, 
            reasoning="Error in analysis, defaulting to neutral",
            regime="goldilocks",
            correlation_risk=0.5,
            stress_test_result="caution"
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=RayDalioSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_ray_dalio_signal,
    )