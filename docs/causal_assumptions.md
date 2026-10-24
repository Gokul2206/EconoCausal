```markdown
# Causal Assumptions (Week 1)

## Goal

Estimate how much a discount changes a customer's probability of repeating a purchase within 90 days, separately for $10 and $20 offers.

## Variables

- **Treatment:** `discount_amount`, one of $0, $10, $20.
- **Outcome:** `purchase_made`, repeat purchase within 90 days (1 = yes, 0 = no).
- **Confounders:** variables known before the offer that affect both who gets an offer and whether the customer repeats.

## Causal graph

```
income ─────────┬──> discount_amount ──> purchase_made
prior_orders ───┤                          ▲
tenure_days ────┤                          │
season ─────────┴──────────────────────────┘
income, prior_orders, tenure_days, season ──> purchase_made
```

Every confounder points to both the treatment and the outcome. The treatment points to the outcome.

## Why each arrow exists

**income**
- Affects the offer: higher-income customers are more likely to be targeted, since they can spend more.
- Affects the outcome: higher-income customers are more likely to buy again.
- Measured before the offer: yes.

**prior_orders**
- Affects the offer: past buyers are treated as loyal customers and targeted.
- Affects the outcome: customers who bought before are more likely to buy again.
- Measured before the offer: yes.

**tenure_days**
- Affects the offer: long-standing customers receive more offers.
- Affects the outcome: long-standing customers have more reasons to return.
- Measured before the offer: yes.

**season**
- Affects the offer: campaigns run in certain seasons.
- Affects the outcome: buying patterns change with the season.
- Measured before the offer: yes.

**discount_amount → purchase_made**
- This is the effect we want to estimate.

## Excluded on purpose

- **Planted columns** (`true_ite`, `true_ite_10`, `true_ite_20`): validation only, never model inputs.
- **Anything measured after the offer:** would be a mediator or a collider, not a confounder.

## Known limits

- **Unmeasured confounders:** if something not in the data, such as browsing intent, affects both the offer and the repeat purchase, the estimates are biased. The mock has no such variable, so this is a limit of the design, not something the analysis can detect.
- **DoWhy's identification is conditional:** it shows the effect is identifiable if this graph is correct. The graph is an assumption. Refutation tests at Mid-Review check the assumptions against the data, but they do not prove them.
```