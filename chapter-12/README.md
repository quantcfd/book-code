# QuantCFD - Chapter 12: Capstone Code Package

**Tác giả:** Anthony Nguyễn (Phuoc Nguyen Thanh)
**Publisher:** Monta Capital Investment Company Limited
**Domain:** quantcfd.com
**Discord:** discord.gg/CC6xsZ8tcf
**GitHub:** github.com/quantcfd

---

## Tổng quan

Chapter 12 là **Capstone** — final chapter integrating tất cả Ch7-11 thành single deployable trading system. Code này không phải lý thuyết — đây là production-grade architecture em có thể deploy với real money sau khi customize.

Package này gồm **15 modules**:
- 7 core modules (subsystems)
- 7 solution exercises (5 standard + 2 bonus)
- 1 README (this file)

---

## Cấu trúc package

```
chapter-12/
├── README.md                           # File này
├── portfolio_orchestrator.py           # Multi-strategy capital allocation
├── pre_launch_checklist.py             # 78-item validation enforcer
├── deployment_phases.py                # Phase 0-5 progression manager
├── live_monitoring.py                  # Real-time dashboard + alerts
├── performance_attribution.py          # Per-strategy P&L decomp
├── scaling_engine.py                   # Capital scaling decisions
├── capstone_system.py                  # Main entry point integrating all
├── solution_exercise_1.py              # Pre-launch self-audit
├── solution_exercise_2.py              # Multi-strategy portfolio
├── solution_exercise_3.py              # Live monitoring dashboard
├── solution_exercise_4.py              # Performance attribution
├── solution_exercise_5.py              # Phased deployment simulation
├── solution_exercise_6.py              # BONUS - Full integration
└── solution_exercise_7.py              # BONUS - Personal capstone plan
```

---

## Kiến trúc tổng thể (7 layers)

```
LAYER 7: BUSINESS         (tax, financial planning, career)
LAYER 6: PSYCHOLOGY       (Ch11 — pre/post trade workflow)
LAYER 5: RISK             (Ch10 — sizing, limits, kill switches)
LAYER 4: PORTFOLIO        (Ch12 NEW — multi-strategy allocation)
LAYER 3: STRATEGY         (Ch7-9 — Trend, MR, Vol BO)
LAYER 2: EXECUTION        (Ch4 — broker API, orders)
LAYER 1: DATA             (Ch3 — feeds, storage, news)
```

---

## Yêu cầu môi trường

### Python version
- Python 3.10+
- Tested on Python 3.10, 3.11, 3.12

### Dependencies
```bash
pip install pandas numpy
```

Các modules core của Ch12 chỉ cần `pandas` và `numpy`. Modules tích hợp (production) sẽ cần thêm:
```bash
pip install MetaTrader5      # Broker API (Windows only)
pip install ccxt              # Crypto exchange API
pip install discord-webhook   # Alert notifications
pip install vectorbt          # Backtesting (Ch5)
```

### Hardware
- **Trading PC:** 8GB RAM minimum, 16GB recommended
- **Internet:** Stable broadband + mobile hotspot backup
- **Optional VPS:** AWS/DigitalOcean $20-50/tháng cho 24/7 monitoring

---

## Quick start

### 1. Test mỗi module độc lập
```bash
cd chapter-12/
python3 portfolio_orchestrator.py
python3 pre_launch_checklist.py
python3 deployment_phases.py
python3 live_monitoring.py
python3 performance_attribution.py
python3 scaling_engine.py
python3 capstone_system.py
```

Mỗi module chạy demo riêng, in kết quả ra console.

### 2. Run all solution exercises
```bash
python3 solution_exercise_1.py    # Pre-launch self-audit
python3 solution_exercise_2.py    # Multi-strategy portfolio
python3 solution_exercise_3.py    # Live monitoring
python3 solution_exercise_4.py    # Performance attribution
python3 solution_exercise_5.py    # Phased deployment journey
python3 solution_exercise_6.py    # BONUS Full integration
python3 solution_exercise_7.py    # BONUS Personal capstone
```

### 3. Test all at once
```bash
for f in *.py; do
  python3 $f >/dev/null 2>&1 && echo "PASS: $f" || echo "FAIL: $f"
done
```

---

## Module reference

### `portfolio_orchestrator.py`

**Mục đích:** Quản lý multi-strategy portfolio với equal-weight allocation.

**Chính class:** `PortfolioOrchestrator`

**Capabilities:**
- Equal-weight allocation (33/33/33 default)
- Per-strategy capital tracking
- Cross-strategy correlation matrix
- Trade validation (portfolio-level)
- Monthly rebalancing
- Performance attribution

**Usage:**
```python
from portfolio_orchestrator import PortfolioOrchestrator

portfolio = PortfolioOrchestrator(
    total_capital=30000,
    strategy_names=['trend_xau', 'mr_eur', 'vol_bo_btc'],
)

# Validate trade
result = portfolio.validate_new_trade(
    strategy_name='trend_xau',
    trade_risk=80,
    current_equity=30000,
    current_open_risk=200,
)
```

---

### `pre_launch_checklist.py`

**Mục đích:** Enforce 78-item checklist trước khi deploy live.

**Chính class:** `PreLaunchChecklist`

**6 sections:**
- Section A: Data infrastructure (12 items)
- Section B: Strategy validation (15 items)
- Section C: Risk infrastructure (12 items)
- Section D: Psychology infrastructure (12 items)
- Section E: Execution infrastructure (15 items)
- Section F: Operations + monitoring (12 items)

**Rule:** Tất cả 78 items phải ✓ trước khi authorize live deployment.

**Usage:**
```python
from pre_launch_checklist import PreLaunchChecklist

checklist = PreLaunchChecklist()
checklist.mark_completed('A1', notes='Data feed verified')

deployment = checklist.can_deploy_live()
print(deployment['authorized'])  # True if all 78 done
```

---

### `deployment_phases.py`

**Mục đích:** Track progression through Phase 0-5 với exit criteria validation.

**6 phases:**
- Phase 0: Paper trading (4-8 weeks)
- Phase 1: Tiny live $500-1k (4-8 weeks)
- Phase 2: Small live $5k (3-6 months)
- Phase 3: Medium $20k (6-12 months)
- Phase 4: Significant $100k (12-24 months)
- Phase 5: Institutional $500k+ (24+ months)

**Exit criteria each phase:**
- Minimum duration
- Minimum trade count
- Sharpe ratio threshold
- Max DD acceptable
- Rule adherence
- Mentor approval
- Family support

**Usage:**
```python
from deployment_phases import PhasedDeploymentManager, DeploymentPhase

manager = PhasedDeploymentManager()
manager.initialize_at_phase(
    phase=DeploymentPhase.PHASE_0_PAPER,
    starting_capital=0,
)

# Update stats
manager.update_stats(
    closed_trades=58,
    rolling_sharpe=0.95,
    rule_adherence_pct=0.93,
)

# Check if ready to advance
criteria = manager.check_exit_criteria()
if criteria['ready_to_advance']:
    manager.advance_to_next_phase(new_capital=1000)
```

---

### `live_monitoring.py`

**Mục đích:** Real-time monitoring dashboard với Discord/Telegram alerts.

**Chính class:** `LiveMonitor`

**Alert priorities:**
- 🔴 CRITICAL: immediate action (kill switch, system error)
- 🟠 HIGH: response trong 1 giờ (DD warning, large slippage)
- 🟡 MEDIUM: response trong 24 giờ (trade events, daily summary)
- 🟢 LOW: informational (regime changes, habit reminders)

**Auto-alert thresholds:**
- Daily loss approaching limit (-2%)
- Daily loss limit hit (-3%) → KILL SWITCH
- Critical drawdown (-15%)
- Data feed down
- Broker connection lost

**Usage:**
```python
from live_monitoring import LiveMonitor, AlertPriority

monitor = LiveMonitor(discord_webhook_url='https://discord.com/api/webhooks/...')

# Update dashboard
snapshot = monitor.update_dashboard(
    account_equity=30000,
    today_pnl=-150,
    open_positions=3,
    ...
)

# Manual alert
monitor.create_alert(
    priority=AlertPriority.HIGH,
    title='Large slippage detected',
    message='XAUUSD slippage 3 pips vs normal 0.5',
)
```

---

### `performance_attribution.py`

**Mục đích:** Decompose portfolio P&L by strategy, identify under/over-performers.

**Chính class:** `PerformanceAttribution`

**Metrics computed per strategy:**
- Total P&L + return %
- Sharpe ratio
- Max drawdown
- Win rate, avg win/loss
- Profit factor
- Rolling Sharpe (90-day)

**Usage:**
```python
from performance_attribution import PerformanceAttribution

attr = PerformanceAttribution()

# Record daily returns
attr.record_daily_return('trend_xau', date, return_pct=0.012)

# Generate full report
report = attr.generate_attribution_report({
    'trend_xau': 10000,
    'mr_eur': 10000,
    'vol_bo_btc': 10000,
})
print(report)

# Find best/worst performers
best, worst = attr.identify_best_worst(period_days=30)
```

---

### `scaling_engine.py`

**Mục đích:** Capital scaling decisions với 8-criteria validation + 50% rule.

**Chính class:** `ScalingEngine`

**8 scaling criteria (ALL must pass):**
1. Time at current phase ≥ requirement
2. Closed trades ≥ minimum
3. Sharpe sustained ≥ target
4. Max DD acceptable
5. Rule adherence ≥ 90%
6. Emotional state stable
7. Mentor approval
8. Personal life stable

**50% rule (DD-based sizing):**
- 0-5% DD: 100% normal size
- 5-10% DD: 75% size
- 10-15% DD: 50% size
- 15-20% DD: 25% size
- 20%+ DD: HALT

**Usage:**
```python
from scaling_engine import ScalingEngine, ScalingCriteria

engine = ScalingEngine(current_capital=20000)

criteria = ScalingCriteria(
    months_at_phase=8,
    closed_trades=180,
    sharpe_3month=1.15,
    max_dd_at_size=-0.10,
    rule_adherence_pct=0.94,
    emotional_state_stable=True,
    mentor_approval=True,
    personal_life_stable=True,
)

# Try scale up
result = engine.execute_scale_up(new_capital=25000, criteria=criteria)
```

---

### `capstone_system.py`

**Mục đích:** Main integration orchestrating all 6 subsystems.

**Chính class:** `CapstoneSystem`

**State machine:**
```
initializing → pre_launch → phase_validation → live_trading
                                                     ↓
                                              drawdown_recovery
                                                     ↓
                                                live_trading
```

**Usage:**
```python
from capstone_system import CapstoneSystem, SystemConfig

config = SystemConfig(
    trader_name="Trí",
    starting_capital=20000,
    strategy_names=['trend_xau', 'mr_eur', 'vol_bo_btc'],
)

system = CapstoneSystem(config)
system.begin_pre_launch()

# Trade evaluation pipeline
result = system.evaluate_trade_proposal(
    strategy_name='trend_xau',
    trade_risk=60,
    current_equity=20000,
    current_open_risk=120,
    psychology_passed=True,
)
if result['approved']:
    # Execute trade...
    pass

# Daily updates
system.update_daily_state(
    equity=20300,
    today_pnl=300,
    open_positions=4,
    ...
)
```

---

## Solution exercises

### Exercise 1: Pre-launch self-audit (Beginner, 90 min)
Walk through 78-item checklist, identify gaps, build action plan.
- Output: detailed gap analysis + priority-ordered action plan
- Time estimate: 2-10 weeks to complete all gaps

### Exercise 2: Multi-strategy portfolio (Intermediate, 120 min)
Build complete 3-strategy portfolio với equal weight, correlation, rebalancing.
- Simulates 252 days of returns
- Computes diversification benefit
- Demonstrates monthly rebalancing

### Exercise 3: Live monitoring dashboard (Advanced, 180 min)
Simulate 5-day trading week với various scenarios.
- Day 1: normal day
- Day 2-3: drawdown beginning
- Day 4: kill switch triggered
- Day 5: recovery với reduced sizing

### Exercise 4: Performance attribution (Advanced, 120 min)
6-month attribution analysis across 3 strategies.
- Per-strategy detailed breakdown
- Rolling Sharpe analysis
- Best/worst identification
- Allocation effectiveness assessment

### Exercise 5: Phased deployment (Advanced, 90 min)
30-month journey simulation Phase 0 → Phase 4.
- Each phase exit criteria validation
- Capital progression $0 → $122k
- Realistic Sharpe progression

### Exercise 6 (BONUS): Full capstone integration (240 min)
End-to-end production simulation combining all 6 subsystems.
- 90-day full lifecycle
- Trade evaluation pipeline
- State machine transitions
- Integration validation

### Exercise 7 (BONUS): Personal capstone plan (360 min)
Template em fill in với your own choices.
- Trader profile
- Strategy selection
- Risk parameters
- Deployment timeline
- Broker setup
- Lifestyle integration
- Success metrics
- **Output:** your personal trading business plan

---

## Best practices

### Daily routine
1. **Pre-market (07:00 VN):** review economic calendar, check overnight, emotion log
2. **London open (15:00 VN):** active trading
3. **End of day (22:00 VN):** close positions appropriate, daily journal, tomorrow plan
4. **Weekly (Saturday morning):** review metrics, mentor consultation
5. **Monthly (last Sunday):** full attribution, strategy review, lifestyle check

### Code maintenance
- Run all tests weekly: `for f in *.py; do python3 $f; done`
- Backup trade logs daily
- Git commit changes weekly
- Mentor code review monthly

### Common pitfalls
- ❌ Skipping Phase 0 paper trading
- ❌ Quitting day job before Phase 4
- ❌ Not following pre-trade checklist
- ❌ Override during drawdown
- ❌ Adding strategies before mastering existing
- ❌ Trading family"s emergency fund

### Success patterns
- ✅ Day job throughout
- ✅ Daily journal religious
- ✅ Mentor relationship deep
- ✅ Family transparency
- ✅ Process > outcome focus
- ✅ Patience với phase progression

---

## Resources

### Books referenced
- Schwager — Market Wizards series
- Chan — Quantitative Trading
- Lo — Adaptive Markets
- Pardo — Evaluation and Optimization

### Online communities
- Discord: discord.gg/CC6xsZ8tcf
- GitHub: github.com/quantcfd
- Website: quantcfd.com

### VN-specific
- Brokers: IC Markets, Pepperstone, Exness (Tier 1)
- Crypto: Binance, OKX
- Tax: consult VN accountant familiar với forex

---

## Disclaimer

**Trading is risky.** Past performance không guarantee future results.

Code này provided as-is, không có warranties. Use at your own risk.

Tác giả không đảm bảo profitability. Trading CFDs có thể lose all capital.

Always:
- Test thoroughly before live
- Start small, scale gradually
- Keep day job until proven (5+ years)
- Family transparency essential
- Mentor relationship critical
- Mental health > P&L

---

## License

MIT License - free to use, modify, distribute.

If you find this useful, consider:
- Star repo on GitHub
- Join Discord community
- Share results với community
- Help other VN traders learn

---

**Hành trình bắt đầu từ đây.**

Ch1-12 = roadmap. Em là driver. 20-năm career đợi.

Patience. Discipline. Compounding.

— Anthony Nguyễn, Monta Capital, 2025
