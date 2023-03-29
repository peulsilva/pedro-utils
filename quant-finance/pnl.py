import pandas as pd

def calculate_pnl(
    price : pd.Series,
    positions : pd.Series[int],
    traded_price : pd.Series[float]
)-> pd.Series:
    """Calculates the pnl of a buy-sell trading
    strategy

    Args:
        price (pd.Series): stock (or asset) price
        positions (pd.Series): long or short position (must
        be integer)
        traded_price (pd.Series): price that you bought/sold

    Returns:
        pd.Series: pnl over time
    """    

    cash = (traded_price * positions)\
        .cumsum()\
        .reindex(price.index)\
        .fillna(method='ffill')\
        .fillna(0)
    
    value_on_position = positions\
        .cumsum()\
        .reindex(price.index)\
        .fillna(method= 'ffill')\
        .fillna(0)
    
    return cash * value_on_position