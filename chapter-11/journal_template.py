"""
QuantCFD - Chapter 11
journal_template.py - Templates and helpers for trade journal entries

Pre-built templates for daily, weekly, monthly, quarterly journal entries.
"""

from __future__ import annotations
from datetime import datetime, date, timedelta
from typing import Dict, Optional


PER_TRADE_TEMPLATE = """
Trade ID: {trade_id}
Date/Time: {datetime}
Asset: {asset}
Strategy: {strategy}

--- SETUP ---
Entry trigger: {entry_trigger}
Confidence: {confidence}/5
Setup grade: {setup_grade}
Pre-trade emotion: {pre_emotion}/10

--- EXECUTION ---
Entry: {entry_price}
Stop: {stop_loss}
Target: {profit_target}
Position: {position_size}
Risk: ${risk_amount}

--- DURATION ---
Hold time: {hold_time}
Did I check P&L during? {checked_pnl}
Did I want to override rules? {wanted_override}
Override taken? {override_taken}

--- OUTCOME ---
Exit: {exit_price}
P&L: {pnl}
R-multiple: {r_multiple}
Followed rules: {followed_rules}

--- REFLECTION ---
What I did right: {what_went_right}
What I could improve: {what_could_improve}
Emotional state during: {emotional_state}
Lesson learned: {lesson_learned}
"""


DAILY_TEMPLATE = """
Daily Journal - {date}

Total trades: {n_trades}
Wins / Losses: {n_wins} / {n_losses}
Net P&L: ${net_pnl}

--- PROCESS ---
Pre-trade checklist completion: {checklist_completion}
Rules followed: {rules_followed_pct}%
Position sizing per formula: {position_sizing_ok}
Loss limits respected: {loss_limits_ok}

--- EMOTIONAL ---
Stress level: {stress}/10
Sleep quality last night: {sleep}/10
Energy level: {energy}/10
Patience level: {patience}/10

--- LEARNING ---
Best trade today: {best_trade}
Worst trade today: {worst_trade}
Mistake pattern this week: {mistake_pattern}
Action item tomorrow: {action_item}

--- LIFE CONTEXT ---
Family/work stress: {life_stress}
Exercise today: {exercise}
Diet quality: {diet}
Anything affecting trading: {context}
"""


WEEKLY_TEMPLATE = """
Weekly Review - Week of {week_start} to {week_end}

--- METRICS ---
Trades: {n_trades}
Win rate: {win_rate_pct}%
Net P&L: ${net_pnl}
Sharpe rolling 4-week: {rolling_sharpe}
Max DD this week: {max_dd_pct}%
Days hit daily limit: {days_limit}
Days no signal: {days_no_signal}

--- PATTERN ANALYSIS ---
Best setups this week: {best_setups}
Worst setups this week: {worst_setups}
Time-of-day pattern: {time_pattern}
Day-of-week pattern: {day_pattern}

--- EMOTIONAL ---
Average stress: {avg_stress}/10
Sleep average: {avg_sleep} hrs
Exercise sessions: {exercise_count}
Energy correlation với P&L: {energy_correlation}

--- ACTION ITEMS ---
1. {action_1}
2. {action_2}
3. {action_3}
"""


MONTHLY_TEMPLATE = """
Monthly Review - {month_year}

--- PERFORMANCE ---
Trades: {n_trades}
Wins: {n_wins} ({win_rate_pct}%)
Net P&L: ${net_pnl} ({pct_return}%)
Sharpe (monthly): {sharpe}
Max DD: {max_dd_pct}%
Days hit daily limit: {days_limit}

--- PROCESS ADHERENCE ---
Pre-trade checklist completion: {checklist_pct}%
Position sizing per formula: {sizing_pct}%
Loss limits respected: {limits_pct}%
Daily journal completion: {journal_pct}%
Weekly reviews completed: {reviews_completed}/4

--- STRATEGIES ---
{strategy_breakdown}

--- BIASES IDENTIFIED ---
{biases_identified}

--- EMOTIONAL TRENDS ---
Stress avg: {avg_stress}/10
Sleep avg: {avg_sleep} hrs
Exercise: {exercise_sessions} sessions
Energy correlation: {energy_correlation}

--- MENTOR REVIEW ---
Discussion points: {mentor_topics}
Action items from mentor: {mentor_action_items}

--- NEXT MONTH GOALS ---
{next_month_goals}
"""


QUARTERLY_TEMPLATE = """
Quarterly Review - {quarter}

--- PERFORMANCE METRICS ---
Total trades: {n_trades}
Winning trades: {n_wins} ({win_rate_pct}%)
Net P&L: ${net_pnl} ({pct_return}%)
Sharpe (quarterly): {sharpe}
Max DD: {max_dd_pct}%
Best month: {best_month}
Worst month: {worst_month}

--- STRATEGY PERFORMANCE ---
{strategy_breakdown}

--- PROCESS METRICS ---
Checklist adherence: {checklist_pct}%
Rule violations: {rule_violations} total
Pattern: {violations_pattern}

--- PSYCHOLOGY EVOLUTION ---
{psychology_progress}

--- LIFESTYLE METRICS ---
Average sleep: {avg_sleep} hrs
Exercise sessions: {exercise_sessions}
Family quality time: {family_time}
Stress baseline: {stress_baseline}/10
Burnout signals: {burnout_signals}

--- GOALS ACHIEVED ---
{goals_achieved}

--- GOALS NEXT QUARTER ---
{next_goals}

--- LONG-TERM PERSPECTIVE ---
{long_term}
"""


def format_per_trade_template(**kwargs) -> str:
    """Format per-trade journal entry. Missing fields filled với '?'."""
    fields = [
        "trade_id", "datetime", "asset", "strategy",
        "entry_trigger", "confidence", "setup_grade", "pre_emotion",
        "entry_price", "stop_loss", "profit_target", "position_size", "risk_amount",
        "hold_time", "checked_pnl", "wanted_override", "override_taken",
        "exit_price", "pnl", "r_multiple", "followed_rules",
        "what_went_right", "what_could_improve", "emotional_state", "lesson_learned",
    ]
    defaults = {f: "?" for f in fields}
    defaults.update(kwargs)
    return PER_TRADE_TEMPLATE.format(**defaults)


def format_daily_template(**kwargs) -> str:
    """Format daily journal entry."""
    fields = [
        "date", "n_trades", "n_wins", "n_losses", "net_pnl",
        "checklist_completion", "rules_followed_pct",
        "position_sizing_ok", "loss_limits_ok",
        "stress", "sleep", "energy", "patience",
        "best_trade", "worst_trade", "mistake_pattern", "action_item",
        "life_stress", "exercise", "diet", "context",
    ]
    defaults = {f: "?" for f in fields}
    defaults.update(kwargs)
    return DAILY_TEMPLATE.format(**defaults)


def get_today_template_filled(default_fill: str = "(fill in)") -> str:
    """Get daily template với today's date filled in."""
    return format_daily_template(
        date=date.today().isoformat(),
        n_trades=default_fill,
        n_wins=default_fill,
        n_losses=default_fill,
        net_pnl=default_fill,
        checklist_completion=default_fill,
        rules_followed_pct=default_fill,
        position_sizing_ok=default_fill,
        loss_limits_ok=default_fill,
        stress=default_fill,
        sleep=default_fill,
        energy=default_fill,
        patience=default_fill,
        best_trade=default_fill,
        worst_trade=default_fill,
        mistake_pattern=default_fill,
        action_item=default_fill,
        life_stress=default_fill,
        exercise=default_fill,
        diet=default_fill,
        context=default_fill,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Journal Template Demo")
    print("=" * 60)

    # Per-trade template
    print("\n--- Per-Trade Template (filled) ---")
    print(format_per_trade_template(
        trade_id="TR-2025-0314-XAU-01",
        datetime="2025-03-14 14:30 VN",
        asset="XAUUSD",
        strategy="Trend (Ch7 MA crossover)",
        entry_trigger="50EMA cross 200EMA + ADX > 25",
        confidence="4",
        setup_grade="A",
        pre_emotion="3",
        entry_price="$2,015",
        stop_loss="$1,995",
        profit_target="$2,055",
        position_size="0.05 lots",
        risk_amount="200",
        hold_time="3 days",
        checked_pnl="N",
        wanted_override="N",
        override_taken="N",
        exit_price="$2,055",
        pnl="+$400",
        r_multiple="+2.0",
        followed_rules="Y",
        what_went_right="Pre-defined exits, walked away, no monitoring",
        what_could_improve="Nothing critical, textbook trade",
        emotional_state="Calm, did not check chart",
        lesson_learned="Walking away after entry = best practice",
    ))

    # Daily template
    print("\n--- Daily Template (today, blank) ---")
    print(get_today_template_filled())
