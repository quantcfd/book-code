"""
Bài tập 7 (BONUS) - Personal capstone project template
========================================================

Goal: Final project — your own personalized trading system.

This is a TEMPLATE em fill in với your own choices.
Each section has placeholders for your decisions.

After completing this template, em có complete documented
trading system ready for Phase 0 paper deployment.

QuantCFD Chapter 12 Capstone exercise.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


# ============================================================================
# SECTION 1: TRADER PROFILE
# Fill in your personal information
# ============================================================================

@dataclass
class TraderProfile:
    """YOUR personal trader profile."""
    name: str = "Tên Của Em"          # ← Fill in your name
    age: int = 30                       # ← Your age
    location: str = "TP HCM, Việt Nam"  # ← Your city
    occupation: str = "Software Engineer"  # ← Your day job
    monthly_income: float = 30000000    # ← Your monthly income (VND)
    monthly_expenses: float = 15000000  # ← Your monthly expenses (VND)
    family_status: str = "Married, 0 kids"  # ← Family situation
    family_aware: bool = True           # ← Family knows you trade?
    family_supportive: bool = True      # ← Family supports trading?

    # Trading background
    years_trading_experience: int = 1
    previous_account_blowups: int = 0
    has_mentor: bool = False            # ← Found a mentor yet?
    mentor_name: str = ""

    @property
    def emergency_fund_months(self) -> float:
        """Compute emergency fund coverage in months."""
        # Using 30M VND savings as example
        savings = 180000000  # ← Your savings (VND)
        return savings / self.monthly_expenses if self.monthly_expenses > 0 else 0


# ============================================================================
# SECTION 2: STRATEGY SELECTION
# Choose YOUR 3 strategies (can use book strategies or own)
# ============================================================================

@dataclass
class StrategyChoice:
    """YOUR strategy selection."""
    name: str
    asset: str
    timeframe: str
    description: str
    expected_sharpe: float
    expected_win_rate: float
    backtest_completed: bool = False
    paper_traded: bool = False


# ← Choose your 3 strategies
my_strategies = [
    StrategyChoice(
        name="Trend Following XAUUSD",     # ← Strategy 1 name
        asset="XAUUSD",                    # ← Asset
        timeframe="H4",                    # ← Timeframe
        description="MA(50) + MA(200) crossover with ADX > 25 filter",
        expected_sharpe=1.2,
        expected_win_rate=0.45,
    ),
    StrategyChoice(
        name="Mean Reversion EURUSD",
        asset="EURUSD",
        timeframe="H1",
        description="Bollinger Band 2σ + RSI(14) extremes",
        expected_sharpe=0.9,
        expected_win_rate=0.55,
    ),
    StrategyChoice(
        name="Volatility Breakout BTCUSD",
        asset="BTCUSD",
        timeframe="H4",
        description="Keltner channel breakout + NR7 contraction",
        expected_sharpe=1.0,
        expected_win_rate=0.40,
    ),
]


# ============================================================================
# SECTION 3: RISK PARAMETERS
# Define YOUR risk tolerance
# ============================================================================

@dataclass
class RiskConfig:
    """YOUR risk management parameters."""
    risk_per_trade_pct: float = 0.01      # 1% per trade default
    daily_loss_limit_pct: float = 0.03    # ← Adjust to your tolerance
    weekly_loss_limit_pct: float = 0.06
    monthly_loss_limit_pct: float = 0.10

    # DD-based scaling thresholds
    dd_warning_pct: float = -0.05
    dd_reduce_50pct: float = -0.10
    dd_reduce_75pct: float = -0.15
    dd_halt_pct: float = -0.20

    # Position limits
    max_concurrent_positions_per_strategy: int = 3
    max_total_concurrent_positions: int = 9

    # ← Customize: more conservative or aggressive?


# ============================================================================
# SECTION 4: DEPLOYMENT PLAN
# Map out YOUR specific deployment timeline
# ============================================================================

@dataclass
class DeploymentPlan:
    """YOUR phased deployment plan."""
    phase_0_start_date: str = "2026-06-01"   # ← When you start paper
    phase_0_duration_weeks: int = 8           # ← Paper duration

    phase_1_capital_usd: float = 1000         # ← Tiny live amount
    phase_1_target_duration_weeks: int = 8

    phase_2_capital_usd: float = 5000
    phase_2_target_duration_months: int = 5

    phase_3_capital_usd: float = 20000
    phase_3_target_duration_months: int = 9

    phase_4_capital_usd: float = 100000
    phase_4_target_duration_months: int = 18


# ============================================================================
# SECTION 5: BROKER SELECTION
# Choose YOUR brokers
# ============================================================================

@dataclass
class BrokerSetup:
    """YOUR broker configuration."""
    primary_broker: str = "IC Markets"     # ← Your primary
    primary_broker_account_type: str = "Raw Spread (ECN)"
    primary_broker_initial_deposit: float = 1000

    secondary_broker: str = "Pepperstone"  # ← For Phase 3+
    secondary_broker_account_type: str = "Razor (ECN)"

    crypto_exchange: str = "Binance"       # ← For BTC strategy
    crypto_exchange_account_type: str = "Spot + Futures"

    funding_method: str = "Bank wire (USD)"  # ← How you'll fund


# ============================================================================
# SECTION 6: LIFESTYLE & PSYCHOLOGY
# Define YOUR lifestyle integration
# ============================================================================

@dataclass
class LifestyleConfig:
    """YOUR lifestyle integration."""
    # Trading session
    primary_session: str = "London"         # London/NY/Asia
    daily_trading_hours: float = 4
    pre_market_routine_minutes: int = 30
    post_market_routine_minutes: int = 30

    # Habits (5 critical from Ch11)
    habit_1: str = "Daily journal entry"
    habit_2: str = "Pre-trade checklist 100%"
    habit_3: str = "Sleep 7+ hours"
    habit_4: str = "Exercise 4x/week"
    habit_5: str = "Weekly mentor consult"

    # Psychology
    daily_emotion_check: bool = True
    weekly_bias_detector: bool = True
    monthly_mentor_review: bool = True

    # Lifestyle
    sleep_hours_target: float = 7.5
    exercise_sessions_per_week: int = 4
    family_time_hours_per_week: float = 15


# ============================================================================
# SECTION 7: SUCCESS METRICS
# Define YOUR success criteria
# ============================================================================

@dataclass
class SuccessMetrics:
    """YOUR definition of success."""
    # Year 1 targets
    year_1_sharpe_target: float = 0.8
    year_1_max_dd_acceptable: float = -0.15
    year_1_rule_adherence_target: float = 0.90
    year_1_minimum_trades: int = 200

    # 5-year targets
    year_5_sharpe_target: float = 1.0
    year_5_total_capital_target_usd: float = 100000

    # Long-term goals
    long_term_goal: str = "Sustainable trading income while keeping day job"
    quit_day_job_capital_threshold: float = 500000  # USD
    quit_day_job_income_multiple: float = 1.5  # Trading income > 1.5x salary


# ============================================================================
# MAIN: GENERATE PERSONAL CAPSTONE PLAN
# ============================================================================

def generate_personal_capstone_plan():
    """Generate complete personal capstone plan document."""
    profile = TraderProfile()
    risk = RiskConfig()
    deployment = DeploymentPlan()
    broker = BrokerSetup()
    lifestyle = LifestyleConfig()
    metrics = SuccessMetrics()

    print("=" * 70)
    print("EXERCISE 7 (BONUS): PERSONAL CAPSTONE PLAN")
    print("=" * 70)
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("Status:    DRAFT — modify with your specific choices")
    print("=" * 70)

    # SECTION 1: Trader profile
    print("\n┌─ SECTION 1: TRADER PROFILE ─┐")
    print(f"  Name:                    {profile.name}")
    print(f"  Age:                     {profile.age}")
    print(f"  Location:                {profile.location}")
    print(f"  Day job:                 {profile.occupation}")
    print(f"  Monthly income:          {profile.monthly_income:,.0f} VND")
    print(f"  Monthly expenses:        {profile.monthly_expenses:,.0f} VND")
    print(f"  Family status:           {profile.family_status}")
    print(f"  Family supportive:       {'Yes' if profile.family_supportive else 'No'}")
    print(f"  Emergency fund cover:    {profile.emergency_fund_months:.1f} months")
    print(f"  Has mentor:              {'Yes' if profile.has_mentor else 'NEED TO FIND'}")

    if not profile.has_mentor:
        print("\n  ⚠️  ACTION ITEM: Find mentor before Phase 1")

    # SECTION 2: Strategies
    print("\n┌─ SECTION 2: STRATEGY SELECTION ─┐")
    for i, strat in enumerate(my_strategies, 1):
        print(f"\n  Strategy {i}: {strat.name}")
        print(f"    Asset/Timeframe:    {strat.asset} / {strat.timeframe}")
        print(f"    Description:        {strat.description}")
        print(f"    Expected Sharpe:    {strat.expected_sharpe}")
        print(f"    Expected win rate:  {strat.expected_win_rate*100:.0f}%")
        print(f"    Backtest:           {'✓' if strat.backtest_completed else '✗ TODO'}")
        print(f"    Paper traded:       {'✓' if strat.paper_traded else '✗ TODO'}")

    # SECTION 3: Risk
    print("\n┌─ SECTION 3: RISK PARAMETERS ─┐")
    print(f"  Per-trade risk:          {risk.risk_per_trade_pct*100}%")
    print(f"  Daily loss limit:        {risk.daily_loss_limit_pct*100}%")
    print(f"  Weekly loss limit:       {risk.weekly_loss_limit_pct*100}%")
    print(f"  Monthly loss limit:      {risk.monthly_loss_limit_pct*100}%")
    print(f"  Max DD before halt:      {risk.dd_halt_pct*100}%")
    print(f"  Max concurrent trades:   {risk.max_total_concurrent_positions}")

    # SECTION 4: Deployment
    print("\n┌─ SECTION 4: DEPLOYMENT TIMELINE ─┐")
    print(f"  Phase 0 (paper):         start {deployment.phase_0_start_date}, "
          f"{deployment.phase_0_duration_weeks} weeks")
    print(f"  Phase 1 (${deployment.phase_1_capital_usd:,.0f}):     "
          f"{deployment.phase_1_target_duration_weeks} weeks")
    print(f"  Phase 2 (${deployment.phase_2_capital_usd:,.0f}):     "
          f"{deployment.phase_2_target_duration_months} months")
    print(f"  Phase 3 (${deployment.phase_3_capital_usd:,.0f}):    "
          f"{deployment.phase_3_target_duration_months} months")
    print(f"  Phase 4 (${deployment.phase_4_capital_usd:,.0f}):   "
          f"{deployment.phase_4_target_duration_months} months")

    # Total time to Phase 4
    total_months = (
        deployment.phase_0_duration_weeks / 4 +
        deployment.phase_1_target_duration_weeks / 4 +
        deployment.phase_2_target_duration_months +
        deployment.phase_3_target_duration_months +
        deployment.phase_4_target_duration_months
    )
    print(f"\n  Total time to Phase 4:   ~{total_months:.0f} months "
          f"({total_months/12:.1f} years)")

    # SECTION 5: Brokers
    print("\n┌─ SECTION 5: BROKER SETUP ─┐")
    print(f"  Primary:                 {broker.primary_broker} ({broker.primary_broker_account_type})")
    print(f"  Initial deposit:         ${broker.primary_broker_initial_deposit:,.0f}")
    print(f"  Secondary:               {broker.secondary_broker} ({broker.secondary_broker_account_type})")
    print(f"  Crypto exchange:         {broker.crypto_exchange}")
    print(f"  Funding method:          {broker.funding_method}")

    # SECTION 6: Lifestyle
    print("\n┌─ SECTION 6: LIFESTYLE INTEGRATION ─┐")
    print(f"  Trading session:         {lifestyle.primary_session}")
    print(f"  Daily trading hours:     {lifestyle.daily_trading_hours}")
    print(f"  Sleep target:            {lifestyle.sleep_hours_target} hours/night")
    print(f"  Exercise:                {lifestyle.exercise_sessions_per_week}x/week")
    print(f"  Family time:             {lifestyle.family_time_hours_per_week} hours/week")
    print(f"\n  Top 5 habits:")
    print(f"    1. {lifestyle.habit_1}")
    print(f"    2. {lifestyle.habit_2}")
    print(f"    3. {lifestyle.habit_3}")
    print(f"    4. {lifestyle.habit_4}")
    print(f"    5. {lifestyle.habit_5}")

    # SECTION 7: Success metrics
    print("\n┌─ SECTION 7: SUCCESS DEFINITION ─┐")
    print("  Year 1 targets:")
    print(f"    Sharpe >          {metrics.year_1_sharpe_target}")
    print(f"    Max DD acceptable: {metrics.year_1_max_dd_acceptable*100:.0f}%")
    print(f"    Rule adherence:   {metrics.year_1_rule_adherence_target*100:.0f}%")
    print(f"    Min trades:       {metrics.year_1_minimum_trades}")
    print("\n  5-year targets:")
    print(f"    Sharpe >          {metrics.year_5_sharpe_target}")
    print(f"    Total capital:    ${metrics.year_5_total_capital_target_usd:,.0f}")
    print(f"\n  Long-term goal:")
    print(f"    {metrics.long_term_goal}")
    print(f"\n  Quit day job criteria:")
    print(f"    Capital >         ${metrics.quit_day_job_capital_threshold:,.0f}")
    print(f"    Trading income >  {metrics.quit_day_job_income_multiple}x salary")

    # FINAL ACTION ITEMS
    print("\n\n" + "=" * 70)
    print("YOUR PERSONAL ACTION ITEMS")
    print("=" * 70)
    print("\nThis week:")
    print("  ☐ Complete strategy backtests Ch7-9")
    print("  ☐ Set up development environment")
    print("  ☐ Open broker paper account")
    print("  ☐ Find mentor (Discord, community, paid)")
    print("  ☐ Tell family about trading plans")

    print("\nThis month:")
    print("  ☐ Run pre-launch checklist (78 items)")
    print("  ☐ Address any infrastructure gaps")
    print("  ☐ Build journal database")
    print("  ☐ Practice pre-trade workflow 50x")
    print("  ☐ Begin daily journal habit")

    print("\nNext 3 months:")
    print("  ☐ Phase 0 paper trading 8 weeks")
    print("  ☐ Open live broker account")
    print("  ☐ Phase 1 deploy $1,000")
    print("  ☐ First 30 live trades")

    print("\n\n" + "=" * 70)
    print("PERSONAL CAPSTONE PLAN GENERATED")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Edit this template với your real choices")
    print("2. Save as PDF for reference")
    print("3. Review monthly với mentor")
    print("4. Update as you learn and grow")
    print("5. This becomes your trading business plan")
    print("\nGood luck. Hành trình bắt đầu từ đây.")
    print("=" * 70)


if __name__ == "__main__":
    generate_personal_capstone_plan()
