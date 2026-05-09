# Chapter 11 — Trading Psychology

**QuantCFD: Giao Dịch Định Lượng CFD Cho Trader Việt**
**Tác giả:** Anthony Nguyễn — Monta Capital
**Website:** [quantcfd.com](https://quantcfd.com) · **Discord:** [discord.gg/CC6xsZ8tcf](https://discord.gg/CC6xsZ8tcf) · **GitHub:** [github.com/quantcfd](https://github.com/quantcfd)

---

## Tổng quan

Chương 11 covers **trading psychology** — vũ khí cuối cùng và quan trọng nhất của trader chuyên nghiệp.

Sau khi có:
- **Strategy edge** (Ch7-9: trend / mean reversion / vol breakout)
- **Risk framework** (Ch10: position sizing, DD, kill switches)

Câu hỏi còn lại: **why most traders với good strategy still blow up?**

Câu trả lời: **psychology**.

Chapter 11 fills the missing piece — psychological discipline để execute under stress.

## Story Arc

**Bảo, 38 tuổi** — project manager senior tech company Singapore TPHCM, lương $8000/tháng, có vợ Lan + 2 con.

**4 lần blow up trong 3 năm:**
- 2021: $30k → $2k (-93%)
- 2022: $25k → $5k (-80%)
- 2023: $40k → $11k (-73%)
- Đầu 2024: $50k → $8k (-84%)

**Tổng cộng mất ~$120k.** Strategy edge real (Sharpe 1.2 OOS), risk framework đúng. Vấn đề: **không follow rules under stress**.

Sau Ch11 framework:
- 6 tháng: $8k → $14k (+47%)
- 9 tháng: $8k → $18.5k (+131%)
- 18 tháng sustained profitability (lần đầu trong sự nghiệp)

## File Structure

```
chapter-11/
├── trade_journal.py            # Structured trade logging với pattern analysis
├── emotion_tracker.py          # 5-dimensional emotion tracking
├── bias_detector.py            # Statistical bias detection từ history
├── pre_trade_checklist.py      # Hardcoded checklist enforcer
├── habit_tracker.py            # 5 critical habits với streaks
├── decision_quality.py         # Annie Duke 4-quadrant framework
├── post_trade_review.py        # Mistake/noise classification
├── mental_simulation.py        # 7 scenarios mental rehearsal
├── journal_template.py         # Per-trade/daily/weekly/monthly/quarterly templates
├── behavioral_dashboard.py     # Combined dashboard với HTML output
├── integrated_psychology_system.py  # Master orchestrator
├── solution_exercise_1.py      # Personal bias assessment
├── solution_exercise_2.py      # Trade journal builder + analysis
├── solution_exercise_3.py      # Emotion tracking + correlation
├── solution_exercise_6.py      # BONUS: Production dashboard
└── solution_exercise_7.py      # BONUS: Full system integration
```

## Module Summaries

### `trade_journal.py`
- **TradeEntry** dataclass — full trade context (setup, execution, outcome, reflection)
- **TradeJournal** class — collection management + pattern analysis
- Methods: `metrics()`, `metrics_by_grade()`, `metrics_by_strategy()`, `emotional_correlation()`, `report()`
- Helper: `generate_synthetic_journal()` for testing

### `emotion_tracker.py`
- **EmotionReading** dataclass — 5 dimensions (stress, confidence, patience, focus, energy 1-10)
- **EmotionTracker** class — log readings 3x daily, identify patterns
- Methods: `log()`, `daily_average()`, `trends()`, `correlate_with_trades()`, `detect_patterns()`
- Property: `composite_score`, `trading_recommendation`

### `bias_detector.py`
- **BiasFinding** — single bias detection result (severity, evidence, recommendation)
- **BiasDetector** — analyzes trade history, detects 5 biases:
  - Loss aversion (cut winners early, hold losers)
  - Overconfidence (size up after wins)
  - Revenge trading (size up after losses)
  - Recency bias (rule violations after losses)
  - P&L obsession (checking during trades)

### `pre_trade_checklist.py`
- **ChecklistResult** — pass/fail với details
- **PreTradeChecklist** — 30-second hardcoded checks:
  - Setup verification (signal, criteria, grade A/B only)
  - Risk check (position size, daily loss limit)
  - Mental check (stress, energy, patience, FOMO, revenge)

### `habit_tracker.py`
- **HabitDefinition** — habit metadata
- **HabitTracker** — track 5 critical habits với:
  - `current_streak`, `longest_streak`
  - `completion_rate` (30-day, 60-day)
  - `identity_strength` (0-100 score combining streak + rate + consistency)
- Default habits: journal, checklist, weekly_review, mentor_call, exercise

### `decision_quality.py` (Annie Duke framework)
- **Decision** dataclass — capture reasoning at decision time
- **DecisionQuality** class:
  - 4-quadrant classification (Skill / Bad luck / Lucky / Just deserts)
  - `calibration_check()` — predicted vs actual win rates
  - `overall_calibration_score()` — 0-100
  - `resulting_fallacy_check()` — detect outcome-only thinking

### `post_trade_review.py`
- **TradeReview** — review structure
- **PostTradeReview** — collection management:
  - Mistake (process violation) vs Noise (bad luck) classification
  - `mistake_rate`, `noise_rate`
  - `common_violations()`, `violation_cost()`
  - `quadrant_summary()`

### `mental_simulation.py`
- **Scenario** — high-stress scenario definition
- **MentalSimulation** — 7 standard scenarios for weekly rehearsal:
  - Major news event
  - Winning streak (overconfidence trigger)
  - Losing streak (revenge trigger)
  - Daily loss limit hit
  - Friend bragging (FOMO trigger)
  - Approaching stop loss
  - Crisis market environment

### `journal_template.py`
- 5 pre-built templates: per-trade, daily, weekly, monthly, quarterly
- Format helpers: `format_per_trade_template()`, `format_daily_template()`
- `get_today_template_filled()` for quick start

### `behavioral_dashboard.py`
- **BehavioralDashboard** — combines all components
- Methods:
  - `overall_health_score()` — 0-100 score
  - `status_summary()` — quick status mỗi area
  - `report()` — comprehensive text report
  - `to_html()` — HTML dashboard for sharing/sharing với mentor

### `integrated_psychology_system.py` (Master orchestrator)
- **IntegratedPsychologySystem** — top-level interface combining all components
- Workflows:
  - `pre_trade_workflow()` — emotion → bias → checklist (returns go/no-go)
  - `post_trade_workflow()` — close → review → update state
  - `daily_summary()` — end-of-day metrics
  - `weekly_review()` — automated weekly review
  - `mentor_report()` — structured format cho mentor consultation

## 7 Cognitive Biases Catalog

| Bias | Trigger | Manifestation | Countermeasure |
|------|---------|--------------|----------------|
| **Loss Aversion** | $100 loss feels 2.25x worse than $100 gain | Cut winners early, hold losers | Hardcode profit targets, no early exits |
| **Overconfidence** | Recent winning streak | Size up 2-3x normal | Lock position size formula |
| **Revenge Trading** | Recent loss streak | Aggressive trades to recover | Mandatory cooling-off rules |
| **FOMO** | External signals (Discord, news) | Chase price after move | Set decision rules, ignore externals |
| **Anchoring** | First number seen | Hold at entry price hoping breakeven | Decisions based on current chart only |
| **Confirmation Bias** | Pre-existing belief | Only see confirming info | Steel-man opposite view weekly |
| **Mental Accounting** | Money labels | "House money" gambling | All money is your money |

## 5 Critical Habits

1. **Trade journaling** (daily, 15 min) — #1 long-term improvement
2. **Pre-trade checklist** (each trade, 30 sec) — prevents 80% mistakes
3. **Weekly review** (Saturday, 1 hour) — pattern recognition
4. **Mentor consultation** (weekly, 30 min) — outside perspective
5. **Exercise** (daily, 30-60 min) — mental sustainability

## 90-Day Implementation Roadmap

- **Week 1-2**: Read Ch11, write personal psychology assessment
- **Week 3-4**: Daily journal habit (15 min/day) + emotion tracking 3x/day
- **Week 5-6**: Pre-trade checklist hardcoded + post-trade ritual
- **Week 7-8**: Lifestyle foundations (sleep 7h+, exercise 4x/week)
- **Week 9-10**: Mental rehearsal 5 min/day, identify own biases
- **Week 11-12**: Weekly review + mentor relationship started

After 90 days: 90%+ rule adherence consistently.
After 6 months: significant behavior change.
After 12 months: identity transformation.

## Quick Start

```python
from integrated_psychology_system import IntegratedPsychologySystem
from trade_journal import TradeEntry
from datetime import datetime

# Initialize
system = IntegratedPsychologySystem(
    backtest_avg_win=1.5,
    backtest_avg_loss=-1.0,
    max_risk_per_trade_pct=0.01,
)

# Pre-trade workflow
result = system.pre_trade_workflow(
    signal_triggered=True,
    setup_grade="A",
    position_size=0.05,
    risk_amount=200,
    current_equity=20000,
    stress=4, confidence=6, patience=7, focus=8, energy=7,
)

if result.can_trade:
    # Execute and log trade
    entry = TradeEntry(
        trade_id="T-001",
        date_entry=datetime.now(),
        asset="XAUUSD",
        strategy="trend",
        entry_trigger="MA cross",
        confidence=4, setup_grade="A", pre_trade_emotion=4,
        entry_price=2015, stop_loss=1995, profit_target=2055,
        position_size=0.05, risk_amount=200,
    )
    system.log_trade(entry)

    # After exit, post-trade workflow
    post_result = system.post_trade_workflow(
        trade_id="T-001",
        exit_price=2055,
        followed_rules=True,
        what_went_right="Walked away after entry",
    )

# Weekly review (run Saturday morning)
print(system.weekly_review())

# Mentor report (run before consultation)
print(system.mentor_report())

# Generate HTML dashboard
dashboard = system.get_dashboard()
dashboard.to_html("my_dashboard.html")
```

## 5 Case Studies (Vietnamese Traders)

1. **Thảo (HN)** — Perfectionism paralysis → forced execution rule
2. **Hùng (TPHCM)** — Overconfidence cycle → wife-locked account access
3. **Mai (ĐN)** — Emotional flooding → smaller positions + meditation
4. **Đông (CT)** — Analysis paralysis → hard deadline + small live trade
5. **Trang (HN)** — Community influence → mute Discord during trading

Each case shows: edge adequate, implementation correct, **failure mode = unique psychological pattern**.

## Key Insights

1. **Psychology > Strategy + Risk Management** — survival probability 74% psychology, 20% risk, 5% strategy, 1% tools
2. **Edge plays out over 1000+ trades** — not 10. Probabilistic thinking essential.
3. **Process > Outcome** — judge yourself by execution, not P&L (Annie Duke)
4. **Identity matters** — "I am process executor" >> "I am profitable trader"
5. **Sustainability > Heroics** — 15%/yr × 30 years compounding > 30%/yr với blow ups

## Dependencies

```
numpy >= 1.24
pandas >= 2.0
```

## Running the Modules

```bash
cd chapter-11
python3 trade_journal.py          # Demo journal
python3 emotion_tracker.py        # Demo tracker
python3 bias_detector.py          # Demo biases
python3 pre_trade_checklist.py    # Demo checklist
python3 habit_tracker.py          # Demo habits
python3 decision_quality.py       # Demo Annie Duke framework
python3 post_trade_review.py      # Demo reviews
python3 mental_simulation.py      # Demo rehearsal
python3 behavioral_dashboard.py   # Demo dashboard
python3 integrated_psychology_system.py  # Demo full system

# Exercises
python3 solution_exercise_1.py    # Bias assessment
python3 solution_exercise_2.py    # Journal builder
python3 solution_exercise_3.py    # Emotion correlation
python3 solution_exercise_6.py    # BONUS: Production dashboard
python3 solution_exercise_7.py    # BONUS: Full integration
```

## References

- Mark Douglas — *Trading in the Zone* (2000)
- Brett Steenbarger — *The Psychology of Trading* (2003), *Trading Psychology 2.0* (2015)
- Van Tharp — *Trade Your Way to Financial Freedom* (1998)
- Annie Duke — *Thinking in Bets* (2018)
- Daniel Kahneman — *Thinking, Fast and Slow* (2011)
- James Clear — *Atomic Habits* (2018)
- Anders Ericsson — *Peak* (2016) — deliberate practice
- Mihaly Csikszentmihalyi — *Flow* (1990)

## Liên hệ

- Website: [quantcfd.com](https://quantcfd.com)
- Discord: [discord.gg/CC6xsZ8tcf](https://discord.gg/CC6xsZ8tcf) — channel `#chapter-11`
- GitHub: [github.com/quantcfd/book-code](https://github.com/quantcfd)

---

**Strategy is rented. Risk framework is rented. Psychology is built.**

*Anthony Nguyễn, Monta Capital*
