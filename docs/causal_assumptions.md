CAUSAL ASSUMPTIONS (WEEK 1)

GOAL
Estimate how much a discount changes a customer's probability of repeating a purchase within 90 days, separately for $10 and $20 offers.

VARIABLES
Treatment: discount_amount, one of $0, $10, $20.
Outcome: purchase_made, repeat purchase within 90 days (1 = yes, 0 = no).
Confounders: variables measured before the offer that affect both who gets an offer and whether the customer repeats.

CAUSAL GRAPH
income -> discount_amount -> purchase_made
prior_orders -> discount_amount
tenure_days -> discount_amount
season -> discount_amount
engagement_proxy -> discount_amount
income -> purchase_made
prior_orders -> purchase_made
tenure_days -> purchase_made
season -> purchase_made
engagement_proxy -> purchase_made
device_mobile -> purchase_made
engagement -> discount_amount (unobserved, not in data)
engagement -> purchase_made (unobserved, not in data)

Every observed confounder points to both the treatment and the outcome. The treatment points to the outcome. device_mobile points to the outcome only.

WHY EACH ARROW EXISTS

income
Affects the offer: higher-income customers are more likely to be targeted.
Affects the outcome: higher-income customers are more likely to buy again.
Measured before the offer: yes, but with measurement noise (sd 0.3). The analysis adjusts for the noisy version, so some confounding from income remains.

prior_orders
Affects the offer: past buyers are treated as loyal customers and targeted.
Affects the outcome: customers who bought before are more likely to buy again.
Measured before the offer: yes.

tenure_days
Affects the offer: long-standing customers receive more offers.
Affects the outcome: long-standing customers have more reasons to return.
Measured before the offer: yes.

season
Affects the offer: campaigns run in certain seasons.
Affects the outcome: buying patterns change with the season.
Measured before the offer: yes.

engagement_proxy
Affects the offer and the outcome through engagement, the hidden driver.
Measured before the offer: yes. It is a noisy, measured trace of engagement.
Adjusting for it removes only part of the hidden bias. The remaining bias is bounded by the sensitivity analysis, not removed.

device_mobile
Affects the outcome only: mobile customers buy differently.
Does not affect the offer, so it is not a confounder and is excluded from the treatment model's adjustment set. It is included in the outcome model as an outcome-only predictor.

discount_amount -> purchase_made
This is the effect we want to estimate.

UNOBSERVED DRIVER (stated limitation)
engagement affects both the offer and the purchase. It is not recorded in the data. Its trace, engagement_proxy, is adjusted for, which removes part of its bias. The remaining bias is not removed by the analysis. It is bounded by the sensitivity analysis.

EXCLUDED ON PURPOSE
Planted columns (true_ite, true_ite_10, true_ite_20): validation only, never model inputs.
Anything measured after the offer: it would be a mediator or collider, not a confounder.

KNOWN LIMITS
Unobserved confounding: the proxy reduces but does not eliminate the bias from engagement. Sensitivity analysis gives the strength of unobserved confounding that would change the result; it does not prove the graph is complete.
Measurement error: income is observed with noise, so adjustment for income is partial. The proxy is a design choice in the mock, not a measured quantity.
DoWhy's identification is conditional: it shows the effect is identifiable if this graph is correct. The graph is an assumption. Refutation tests check stability under the graph; they do not prove it.