from supplychain_risk.notify.policy import policy_recommendation

def test_policy_rules():
    assert policy_recommendation({"risk_label":"Low"}) == "Notify & monitor"
    assert policy_recommendation({"risk_label":"High","promised_days_left":0}) == "Expedite to 2-day"
    assert policy_recommendation({"risk_label":"High","dc_stock":10}) == "Split-ship from alternate DC"
