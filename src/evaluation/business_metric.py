def business_metric(
    y_true,
    y_pred,
    amount,
    gain_legitimate_accept_rate=0.02,
    gain_fraud_refuse=50,
    loss_legitimate_refuse=5
):
    """
    Calcula o benefício econômico total de uma política de decisão
    para detecção de fraudes em transações financeiras.

    Parâmetros
    ----------
    y_true : array-like
        Rótulos reais (1 = fraude, 0 = legítima)
    y_pred : array-like
        Decisões do modelo (1 = recusar, 0 = aceitar)
    amount : array-like
        Valor monetário de cada transação

    Retorno
    -------
    float
        Benefício econômico total
    """
        
    # Máscaras de Decisão
    tp = (y_true == 1) & (y_pred == 1)
    fn = (y_true == 1) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    tn = (y_true == 0) & (y_pred == 0)

    # Componentes econômicos
    benefit_fraud_refuse = tp.sum() * gain_fraud_refuse
    loss_fraud_accept = -amount[fn].sum()
    loss_legitimate_refuse_cost = -fp.sum() * loss_legitimate_refuse
    benefit_legitimate_accept = (amount[tn] * gain_legitimate_accept_rate).sum()

    return (
        benefit_fraud_refuse
        + loss_fraud_accept
        + loss_legitimate_refuse_cost
        + benefit_legitimate_accept
    )
