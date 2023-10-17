import pandas as pd
from typing import Union

class FeaturesTransformer():
    def __init__(self) -> None:
        pass
    def momentum(
        self,
        prices_df : Union[pd.DataFrame, pd.Series],
        lookback_window : int
    )-> pd.DataFrame:
        """Get the momentum feature of price 
        (defined as the relative change in price)

        Args:
            prices_df (pd.DataFrame): _description_
            lookback_window (int): i

        Returns:
            pd.DataFrame: momentum of price
        """    
        return prices_df.pct_change(lookback_window)

    def mean_returns(
        self,
        prices_df : Union[pd.DataFrame, pd.Series],
        lookback_window: int
    )-> pd.DataFrame:
        """Create features based on the mean returns of
        the price

        Args:
            prices_df (pd.DataFrame): _description_
            lookback_window (int): _description_

        Returns:
            pd.DataFrame: _description_
        """    
        daily_returns = prices_df.pct_change()
        return daily_returns.rolling(lookback_window)\
            .mean()
        
class TargetTransformer():
    
    def future_returns(
        self,
        prices_df : Union[pd.DataFrame, pd.Series],
        future_window : int
    )-> Union[pd.DataFrame, pd.Series]:
        ...
