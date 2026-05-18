name: Pull Request
about: Propose a change to the quant energy pipeline
title: '[TYPE] Brief description'
labels: ''
assignees: ''

---

## Type of Change
- [ ] 🐛 Bug fix (non-breaking)
- [ ] ✨ New feature (strategy / model)
- [ ] 🔧 Refactor (no behaviour change)
- [ ] 📊 Data layer change (data_agent.py)
- [ ] ⚠ Risk parameter change (requires compliance sign-off)
- [ ] 📚 Documentation

## Summary
<!-- What does this PR do? -->

## Motivation
<!-- Why is this change needed? Which book/model/strategy does it reference? -->

## Changes Made

### Files Modified
- [ ] `data_agent.py`
- [ ] `strategy_agent.py`
- [ ] `risk_agent.py`  ⚠ _Requires risk-committee approval if Section 1 is modified_
- [ ] `main.py`
- [ ] `tests/`
- [ ] `CLAUDE.md`

### Pricing / Model Changes
<!-- If Black-76 / BSM / Monte Carlo logic changed, document the exact formula change -->

## Risk Impact Assessment

### Does this PR modify hardcoded risk parameters in `risk_agent.py` Section 1?
- [ ] No
- [ ] Yes → **Requires risk committee sign-off before merge**

### VaR / Stress Test Impact
<!-- If applicable: how does this change affect VaR, Expected Shortfall, or stress test results? -->

### Greeks Limit Impact
<!-- If new options strategies added: confirm Delta/Gamma/Vega/Theta limits still enforced -->

## Testing

### Tests Added / Updated
- [ ] `tests/test_data_agent.py`
- [ ] `tests/test_strategy_agent.py`
- [ ] `tests/test_risk_agent.py`

### Test Results
```
# Paste pytest output here
```

### Manual Testing
<!-- Describe any manual testing performed (offline / synthetic data only for CI) -->

## Knowledge Base Reference
<!-- Which book(s) from the 21-book knowledge base informed this change? -->
- [ ] Oil Trader Academy
- [ ] Gkinis (Modelling Energy Markets)
- [ ] Hull (Risk Management & Financial Institutions)
- [ ] Bittman (Trading Options as a Professional)
- [ ] Sherbin (How to Price and Trade Options)
- [ ] Trafigura (Commodities Demystified)
- [ ] NYMEX Chapter 200
- [ ] QuantStart (Successful Algorithmic Trading)
- [ ] Other: _______________

## Checklist
- [ ] CI passes (lint + type + tests + risk-guard)
- [ ] No live order-routing or broker API calls added
- [ ] No `execute_trade()` function added
- [ ] All ML features lag ≥1 period (no look-ahead bias)
- [ ] Energy options use `black76()`, not `black_scholes()` directly
- [ ] `evaluate_trade()` called before any signal enters `approved[]`
- [ ] `CLAUDE.md` updated if architecture changed
- [ ] No API keys or credentials committed
