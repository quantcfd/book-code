# QuantCFD — Chapter 10: Risk Management

> Risk management — quan trọng hơn strategy

Code cho Chương 10 của sách **QuantCFD: Giao Dịch Định Lượng CFD Cho Trader Việt** by Anthony Nguyễn (Monta Capital).

## Tổng quan

Chương 10 covers framework đầy đủ về risk management — từ position sizing cơ bản đến full multi-strategy risk system với crisis playbook. Đây là chương quan trọng nhất sách: strategy edge tồn tại, nhưng risk management determines survival.

**Story arc**: Đức (32 tuổi, kỹ sư fintech TPHCM) blew up $50k → $19k (-62%) trong 5 tháng với 3 GOOD strategies vì bad risk management. Sau khi học framework Ch10, recovered $24k (+26%) với SAME strategies + good risk management.

## Cấu trúc files

### Core modules (10 files)

| File | Section | Mô tả |
|------|---------|-------|
| `position_sizing.py` | 10.3 | 4 sizing methods: fixed dollar, fixed fractional, ATR sizing, vol targeting |
| `kelly_calculator.py` | 10.4 | Kelly fraction + Monte Carlo simulation full vs fractional |
| `vol_targeting.py` | 10.7 | Portfolio vol targeting với leverage adaptation |
| `loss_limit_manager.py` | 10.9 | Pre-committed daily/weekly/monthly/total DD limits |
| `correlation_tracker.py` | 10.6 | Pairwise correlation tracking + spike detection |
| `risk_budgeting.py` | 10.5.5 | Equal risk contribution + risk parity allocation |
| `dd_control.py` | 10.8 | DD-based size scaling + equity curve filter + streak loss scaling |
| `risk_dashboard.py` | 10.13 | Per-strategy + portfolio metrics aggregator |
| `stress_test.py` | 10.16 | 5-scenario portfolio stress test framework |
| `kill_switch.py` | 10.12 | Emergency halt mechanism với 6 standard triggers |

### Advanced integration (2 files)

| File | Section | Mô tả |
|------|---------|-------|
| `crisis_playbook.py` | 10.10 | 4-phase crisis state machine (detection → stabilization → assessment → reentry) |
| `combined_risk_system.py` | 10.* | Full integration class — tất cả components trong single CombinedRiskSystem |

### Solution exercises (5 files)

| File | Bài tập | Mô tả |
|------|---------|-------|
| `solution_exercise_1.py` | Bài 1 | Position sizing comparison across 4 methods + ATR scenarios |
| `solution_exercise_2.py` | Bài 2 | Kelly Monte Carlo cho 3 strategy types + correlation discount |
| `solution_exercise_3.py` | Bài 3 | LossLimitManager với 4 scenarios (single big loss, gradual, recovery, terminal) |
| `solution_exercise_6.py` | Bài 6 (BONUS) | Production dashboard + HTML report + JSON snapshot + retirement criteria |
| `solution_exercise_7.py` | Bài 7 (BONUS) | Stress test framework với 5 scenarios + multi-config comparison |

## Quick start

### 1. Run any module standalone

Mỗi file có `if __name__ == "__main__"` block với demo. Chạy trực tiếp:

```bash
cd chapter-10
python position_sizing.py
python kelly_calculator.py
python loss_limit_manager.py
# etc.
```

### 2. Run all tests

```bash
for f in *.py; do
    echo "=== $f ==="
    python $f 2>&1 | tail -5
done
```

### 3. Use combined system

```python
from combined_risk_system import CombinedRiskSystem
import pandas as pd

# Initialize
crs = CombinedRiskSystem(
    initial_equity=25000,
    strategies=["trend", "mr", "vol_bo"],
    target_vol_annual=0.10,
)

# Compute position size for a trade
sizing = crs.compute_position_size(
    strategy_name="trend",
    atr=20,
    contract_value_per_point=100,
)
print(f"Position size: {sizing['size']} lots")

# Master can_trade check (combines all halts)
result = crs.can_trade(
    current_date=pd.Timestamp.now(),
    current_state={
        "daily_loss_pct": -0.01,
        "current_equity": 25000,
        "consecutive_losses": 0,
    },
)
print(f"Allow trade: {result['allow_trade']}")
```

## Risk parameters (defaults)

```python
STRATEGY_RISK_CONFIG = {
    "trend":  {"risk_pct": 0.010, "atr_stop_mult": 2.5, "max_dd_halt": 0.20},
    "mr":     {"risk_pct": 0.005, "atr_stop_mult": 1.5, "max_dd_halt": 0.15},
    "vol_bo": {"risk_pct": 0.007, "atr_stop_mult": 1.5, "max_dd_halt": 0.15},
}

LOSS_LIMITS = {
    "daily": -0.03,    # halt 24h
    "weekly": -0.07,   # halt 7 days
    "monthly": -0.15,  # halt 14 days
    "total_dd": -0.20, # halt 30 days
    "terminal": -0.40, # stop trading 6+ months
}

VOL_TARGET = 0.10  # 10% annualized portfolio vol
```

## Dependencies

```
numpy
pandas
scipy   # optional, only for risk_budgeting risk parity (falls back to equal risk contrib)
```

No external broker APIs required cho demos — uses synthetic data via `np.random.seed(42)` for reproducibility.

## Key concepts

### 1. Position sizing fundamentals (10.3)
- **Fixed dollar**: simplest, doesn't scale with equity
- **Fixed fractional**: scales with equity (recommended for beginners, 0.5-1%)
- **ATR sizing**: vol-adaptive, real risk constant across regimes (recommended)
- **Vol targeting**: portfolio-level, adapts to market conditions (multi-asset funds)

### 2. Kelly Criterion (10.4)
- Full Kelly: maximizes geometric growth but high blow up risk (10-20%)
- Half Kelly: 75% growth với 5x less DD risk
- **Quarter Kelly: sweet spot** for most retail (70% growth, 0.1% blow up)
- Apply correlation discount cho multi-strategy: `actual = individual / sqrt(1 + (n-1) × ρ)`

### 3. Loss limits (10.9)
Pre-committed circuit breakers prevent emotional override:
- Daily: -3% (halt 24h)
- Weekly: -7% (halt 7 days)
- Monthly: -15% (halt 14 days)
- Total DD: -20% (halt 30 days, full review)
- Terminal: -40% (stop trading 6+ months)

### 4. DD control (10.8)
- DD-based size scaling: 5% → 100%, 10% → 75%, 15% → 50%, 20% → halt
- Equity curve filter: trade only when equity > 50-day MA
- Streak loss scaling: after 5 losses cut 50%, after 8 cut 75%

### 5. Vol targeting (10.7)
- Compute realized vol (EWMA, 63-day lookback)
- Leverage = target_vol / realized_vol
- Adapts automatically: calm market = leverage up, vol market = leverage down
- Reduces DD trong crisis 50-70%

### 6. Correlation budgeting (10.6)
- Track rolling 60-day pairwise correlations
- Alert if avg pairwise > 0.5 (crisis indicator)
- Crisis correlation typically 3-5x normal (capped 0.85)
- Halt new entries during correlation spike

### 7. Crisis playbook (10.10)
4-phase state machine:
1. **DETECTION** (Day 1-3): halt new entries, reduce 50%
2. **STABILIZATION** (Day 4-14): 70% cash, monitor only
3. **ASSESSMENT** (Week 3-6): review strategies, decide retirement
4. **REENTRY** (Week 7+): scale up gradually 25% → 50% → 100%

### 8. Stress testing (10.16)
5 historical scenarios:
- 2008 GFC (Sep 2008 - Mar 2009)
- 2010 Flash Crash (May 6-15)
- 2015 SNB CHF unpegging (Jan 15-25)
- 2020 COVID (Feb 19 - Apr 30)
- 2022 Russia-Ukraine + Fed hike (Feb 24 - Apr 30)

PASS criteria: max_dd > -30% in ALL scenarios.

## Recommended deployment workflow

### Phase 1: Foundation (Tháng 1-3)
1. Đọc Ch10 đầy đủ
2. Write personal risk policy (1-2 pages)
3. Implement `position_sizing.py` integrated into existing strategies
4. Implement `loss_limit_manager.py` enforced systematically
5. Paper trade 1 month với full Q1 framework

### Phase 2: Multi-strategy + vol target (Tháng 4-6)
1. Build `correlation_tracker.py` integrated
2. Implement `risk_budgeting.py` allocation
3. Add `vol_targeting.py` overlay
4. Live deployment 25% capital

### Phase 3: Stress + dashboard (Tháng 7-9)
1. Run `stress_test.py` on portfolio
2. Build `risk_dashboard.py` daily refresh
3. Scale to 50% capital
4. First quarterly review

### Phase 4: Crisis + full deployment (Tháng 10-12)
1. Implement `crisis_playbook.py` simulation
2. Add `kill_switch.py` triggers
3. Practice crisis scenarios
4. Scale to 100% target capital
5. Annual review

Total time investment: ~370 hours Year 1 = ~6-10 hours/week.

## Câu chuyện Đức — bài học $19k

**Tháng 6/2024**: Đức blew up $50k → $19k (-62%) với 3 good strategies (Trend XAU + MR EUR + Vol BO BTC).

5 sai lầm:
1. Risk per trade 2% × 3 strategies = 6% cluster risk
2. Không correlation budget — crisis correlations hit 0.7+
3. Không max DD limit — kept sizing up trong DD
4. Không recovery plan — increased size to "recover"
5. Không daily/weekly loss limit — single events ate huge capital

**Tháng 12/2024**: Đức recovered $24k (+26%) với SAME strategies + framework Ch10:
- Risk per trade 0.7% per strategy
- Daily limit -3%, weekly -7% — pre-committed
- DD halt 15% per strategy, 20% portfolio
- Vol target 10% (auto-deleverage in crisis)
- Correlation alerts active

**Lesson**: Strategy edge tồn tại. Risk management determine survival.

## Resources

- **Sách**: Chương 10 trong QuantCFD (70 pages, full theory + examples)
- **Repo**: `github.com/quantcfd/book-code/chapter-10`
- **Discord**: `discord.gg/CC6xsZ8tcf` — channel `#chapter-10`
- **Domain**: `quantcfd.com`

## License

MIT — see `../LICENSE` in repo root.

## Author

**Anthony Nguyễn**
Monta Capital Investment Company Limited
[quantcfd.com](https://quantcfd.com)
