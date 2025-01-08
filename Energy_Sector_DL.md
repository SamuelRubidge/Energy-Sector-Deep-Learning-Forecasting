## Relevant Research
- LSTM: "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
- Attention LSTM: "Attention-Based LSTM for Aspect-Level Sentiment Classification" (Luong et al., 2015)
- GRU: "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (Cho et al., 2014)

## Model Explorations

The data is from Yahoo Finance and has a 2-minute frequency for 5 stocks. These stocks are all in the energy sector, specifically related to fossil fuels and drilling. The data spans from November 20, 2024, to January 6, 2025 (note that Yahoo Finance's 2-minute data is limited). Given this data, we will be using the following models:

- Transformer
- LSTM
- Attention LSTM
- GRU




```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from torchviz import make_dot
import matplotlib.pyplot as plt
from IPython.display import Image, display
import contextlib
import io

```


```python
tickers = ['BTU', 'CNX', 'NOV', 'RRC', 'HP', 'NBR']
with contextlib.redirect_stdout(io.StringIO()):
    df = [yf.download(ticker, start=d, end=d + pd.Timedelta(days=1), interval="2m").reset_index().assign(ticker=ticker)
          for ticker in tickers for d in pd.date_range('2024-11-20', '2025-01-06', freq='D')]
df = pd.concat(df)[['ticker', 'Datetime', 'Close', 'Volume']].sort_values(['ticker', 'Datetime'])

df['Period_Return'] = df['Close'] / df.groupby('ticker')['Close'].shift() - 1

df = df.merge(
    df.groupby('Datetime')['Period_Return'].mean().reset_index().merge(
        df.groupby('Datetime')['Volume'].sum().reset_index(), on= 'Datetime'
    ).rename(columns= {'Period_Return': 'Group_Period_Return', 'Volume': 'Group_Total_Volume'})
).dropna()

df
```

    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['BTU']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-23 00:00:00 -> 2024-11-24 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['BTU']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-24 00:00:00 -> 2024-11-25 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['BTU']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-28 00:00:00 -> 2024-11-29 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['BTU']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-30 00:00:00 -> 2024-12-01 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['BTU']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-01 00:00:00 -> 2024-12-02 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['BTU']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-07 00:00:00 -> 2024-12-08 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['BTU']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-08 00:00:00 -> 2024-12-09 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['BTU']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-14 00:00:00 -> 2024-12-15 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['BTU']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-15 00:00:00 -> 2024-12-16 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['BTU']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-21 00:00:00 -> 2024-12-22 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['BTU']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-22 00:00:00 -> 2024-12-23 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['BTU']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-25 00:00:00 -> 2024-12-26 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['BTU']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-28 00:00:00 -> 2024-12-29 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['BTU']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-29 00:00:00 -> 2024-12-30 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['BTU']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2025-01-01 00:00:00 -> 2025-01-02 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['BTU']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2025-01-04 00:00:00 -> 2025-01-05 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['BTU']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2025-01-05 00:00:00 -> 2025-01-06 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['CNX']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-23 00:00:00 -> 2024-11-24 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['CNX']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-24 00:00:00 -> 2024-11-25 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['CNX']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-28 00:00:00 -> 2024-11-29 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['CNX']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-30 00:00:00 -> 2024-12-01 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['CNX']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-01 00:00:00 -> 2024-12-02 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['CNX']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-07 00:00:00 -> 2024-12-08 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['CNX']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-08 00:00:00 -> 2024-12-09 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['CNX']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-14 00:00:00 -> 2024-12-15 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['CNX']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-15 00:00:00 -> 2024-12-16 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['CNX']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-21 00:00:00 -> 2024-12-22 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['CNX']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-22 00:00:00 -> 2024-12-23 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['CNX']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-25 00:00:00 -> 2024-12-26 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['CNX']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-28 00:00:00 -> 2024-12-29 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['CNX']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-29 00:00:00 -> 2024-12-30 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['CNX']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2025-01-01 00:00:00 -> 2025-01-02 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['CNX']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2025-01-04 00:00:00 -> 2025-01-05 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['CNX']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2025-01-05 00:00:00 -> 2025-01-06 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NOV']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-23 00:00:00 -> 2024-11-24 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NOV']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-24 00:00:00 -> 2024-11-25 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NOV']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-28 00:00:00 -> 2024-11-29 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NOV']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-30 00:00:00 -> 2024-12-01 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NOV']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-01 00:00:00 -> 2024-12-02 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NOV']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-07 00:00:00 -> 2024-12-08 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NOV']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-08 00:00:00 -> 2024-12-09 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NOV']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-14 00:00:00 -> 2024-12-15 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NOV']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-15 00:00:00 -> 2024-12-16 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NOV']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-21 00:00:00 -> 2024-12-22 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NOV']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-22 00:00:00 -> 2024-12-23 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NOV']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-25 00:00:00 -> 2024-12-26 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NOV']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-28 00:00:00 -> 2024-12-29 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NOV']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-29 00:00:00 -> 2024-12-30 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NOV']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2025-01-01 00:00:00 -> 2025-01-02 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NOV']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2025-01-04 00:00:00 -> 2025-01-05 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NOV']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2025-01-05 00:00:00 -> 2025-01-06 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['RRC']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-23 00:00:00 -> 2024-11-24 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['RRC']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-24 00:00:00 -> 2024-11-25 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['RRC']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-28 00:00:00 -> 2024-11-29 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['RRC']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-30 00:00:00 -> 2024-12-01 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['RRC']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-01 00:00:00 -> 2024-12-02 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['RRC']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-07 00:00:00 -> 2024-12-08 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['RRC']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-08 00:00:00 -> 2024-12-09 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['RRC']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-14 00:00:00 -> 2024-12-15 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['RRC']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-15 00:00:00 -> 2024-12-16 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['RRC']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-21 00:00:00 -> 2024-12-22 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['RRC']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-22 00:00:00 -> 2024-12-23 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['RRC']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-25 00:00:00 -> 2024-12-26 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['RRC']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-28 00:00:00 -> 2024-12-29 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['RRC']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-29 00:00:00 -> 2024-12-30 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['RRC']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2025-01-01 00:00:00 -> 2025-01-02 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['RRC']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2025-01-04 00:00:00 -> 2025-01-05 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['RRC']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2025-01-05 00:00:00 -> 2025-01-06 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['HP']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-23 00:00:00 -> 2024-11-24 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['HP']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-24 00:00:00 -> 2024-11-25 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['HP']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-28 00:00:00 -> 2024-11-29 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['HP']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-30 00:00:00 -> 2024-12-01 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['HP']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-01 00:00:00 -> 2024-12-02 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['HP']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-07 00:00:00 -> 2024-12-08 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['HP']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-08 00:00:00 -> 2024-12-09 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['HP']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-14 00:00:00 -> 2024-12-15 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['HP']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-15 00:00:00 -> 2024-12-16 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['HP']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-21 00:00:00 -> 2024-12-22 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['HP']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-22 00:00:00 -> 2024-12-23 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['HP']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-25 00:00:00 -> 2024-12-26 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['HP']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-28 00:00:00 -> 2024-12-29 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['HP']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-29 00:00:00 -> 2024-12-30 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['HP']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2025-01-01 00:00:00 -> 2025-01-02 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['HP']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2025-01-04 00:00:00 -> 2025-01-05 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['HP']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2025-01-05 00:00:00 -> 2025-01-06 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NBR']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-23 00:00:00 -> 2024-11-24 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NBR']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-24 00:00:00 -> 2024-11-25 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NBR']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-28 00:00:00 -> 2024-11-29 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NBR']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-11-30 00:00:00 -> 2024-12-01 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NBR']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-01 00:00:00 -> 2024-12-02 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NBR']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-07 00:00:00 -> 2024-12-08 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NBR']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-08 00:00:00 -> 2024-12-09 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NBR']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-14 00:00:00 -> 2024-12-15 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NBR']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-15 00:00:00 -> 2024-12-16 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NBR']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-21 00:00:00 -> 2024-12-22 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NBR']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-22 00:00:00 -> 2024-12-23 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NBR']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-25 00:00:00 -> 2024-12-26 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NBR']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-28 00:00:00 -> 2024-12-29 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NBR']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2024-12-29 00:00:00 -> 2024-12-30 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NBR']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2025-01-01 00:00:00 -> 2025-01-02 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NBR']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2025-01-04 00:00:00 -> 2025-01-05 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    
    1 Failed download:
    ['NBR']: YFPricesMissingError('$%ticker%: possibly delisted; No price data found  (2m 2025-01-05 00:00:00 -> 2025-01-06 00:00:00)')
    [*********************100%%**********************]  1 of 1 completed
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ticker</th>
      <th>Datetime</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Period_Return</th>
      <th>Group_Period_Return</th>
      <th>Group_Total_Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>CNX</td>
      <td>2024-11-20 09:32:00-05:00</td>
      <td>39.845001</td>
      <td>2007.0</td>
      <td>0.000377</td>
      <td>-0.000680</td>
      <td>34993.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HP</td>
      <td>2024-11-20 09:32:00-05:00</td>
      <td>33.150002</td>
      <td>2305.0</td>
      <td>-0.004056</td>
      <td>-0.000680</td>
      <td>34993.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NOV</td>
      <td>2024-11-20 09:32:00-05:00</td>
      <td>16.090000</td>
      <td>6334.0</td>
      <td>0.001245</td>
      <td>-0.000680</td>
      <td>34993.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RRC</td>
      <td>2024-11-20 09:32:00-05:00</td>
      <td>34.810001</td>
      <td>4889.0</td>
      <td>-0.000287</td>
      <td>-0.000680</td>
      <td>34993.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BTU</td>
      <td>2024-11-20 09:36:00-05:00</td>
      <td>26.689899</td>
      <td>1292.0</td>
      <td>0.003191</td>
      <td>0.000208</td>
      <td>18644.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>33797</th>
      <td>CNX</td>
      <td>2025-01-02 14:28:00-05:00</td>
      <td>37.180000</td>
      <td>1154.0</td>
      <td>0.000135</td>
      <td>-0.000157</td>
      <td>28211.0</td>
    </tr>
    <tr>
      <th>33798</th>
      <td>HP</td>
      <td>2025-01-02 14:28:00-05:00</td>
      <td>33.020000</td>
      <td>2840.0</td>
      <td>-0.000303</td>
      <td>-0.000157</td>
      <td>28211.0</td>
    </tr>
    <tr>
      <th>33799</th>
      <td>NBR</td>
      <td>2025-01-02 14:28:00-05:00</td>
      <td>59.990002</td>
      <td>559.0</td>
      <td>0.000000</td>
      <td>-0.000157</td>
      <td>28211.0</td>
    </tr>
    <tr>
      <th>33800</th>
      <td>NOV</td>
      <td>2025-01-02 14:28:00-05:00</td>
      <td>14.745000</td>
      <td>10597.0</td>
      <td>-0.000339</td>
      <td>-0.000157</td>
      <td>28211.0</td>
    </tr>
    <tr>
      <th>33801</th>
      <td>RRC</td>
      <td>2025-01-02 14:28:00-05:00</td>
      <td>35.970001</td>
      <td>13061.0</td>
      <td>-0.000278</td>
      <td>-0.000157</td>
      <td>28211.0</td>
    </tr>
  </tbody>
</table>
<p>33796 rows × 7 columns</p>
</div>




```python
df = df.rename(columns={'Datetime': 'datetime', 'Period_Return': 'period_return'})

```

## Visualization


```python
def visualize_model(model, model_name):
    sample_input = torch.randn(1, 10, 3)
    y = model(sample_input)

    # Generate and save visualization
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render(f"{model_name}_architecture", format="png", cleanup=True)

    # Display the image
    display(Image(filename=f"{model_name}_architecture.png"))

```

## Data Prep


```python
# train sequence
class StockDataset(Dataset):
    def __init__(self, data, sequence_length=10):
        self.data = data
        self.sequence_length = sequence_length

        # Create sequences of features and targets
        self.X = []
        self.y = []

        for ticker in data['ticker'].unique():
            ticker_data = data[data['ticker'] == ticker]

            features = ticker_data[['period_return', 'Group_Period_Return', 'Group_Total_Volume']].values
            for i in range(len(features) - sequence_length):
                self.X.append(features[i:i+sequence_length])
                self.y.append(features[i+sequence_length, 0])  # predict next period return

        self.X = torch.FloatTensor(self.X)
        self.y = torch.FloatTensor(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

## Transformer Model

### Advantages
- Captures long dependencies well  
- Highly parallel, trains quickly  
- Handles varied sequence lengths  

### Disadvantages
- Needs large data/computation  
- High memory usage overhead  
- Complex to implement well  


```python
# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead), num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1])  # use last sequence output
        return x.squeeze()

```


```python
sample_input = torch.randn(1, 10, 3)  # batch_size=1, seq_length=10, features=3
transformer = TransformerModel(input_dim=3)
y = transformer(sample_input)

# Generate and save visualization
make_dot(y, params=dict(transformer.named_parameters())).render("transformer_architecture", format="png")


print("Transformer Architecture:")
visualize_model(TransformerModel(input_dim=3), "transformer")
```

    c:\ProgramData\anaconda3\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
      warnings.warn(
    

    Transformer Architecture:
    


    
![png](output_12_2.png)
    


## LSTM Model

### Advantages
- Great for sequential tasks  
- Manages long-term dependencies  
- Easy to interpret outputs  

### Disadvantages
- Slower for longer sequences  
- Prone to gradient issues  
- Harder to parallelize effectively


```python
# LSTM Model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1])  # use last sequence output
        return x.squeeze()
```


```python
sample_input = torch.randn(1, 10, 3)
rnn = RNNModel(input_dim=3)
y = rnn(sample_input)
make_dot(y, params=dict(rnn.named_parameters())).render("lstm_architecture", format="png")

# Visualize LSTM
print("\nLSTM Architecture:")
visualize_model(RNNModel(input_dim=3), "lstm")


```

    
    LSTM Architecture:
    


    
![png](output_16_1.png)
    


## Attention LSTM Model

### Advantages
- Combines LSTM and attention  
- Enhanced interpretability with focus  
- Handles longer dependencies effectively  

### Disadvantages
- More computationally expensive  
- Requires careful attention tuning  
- Complex, harder to maintain  


```python
## Attention LSTM Model

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        x = self.fc(attn_out[:, -1])
        return x.squeeze()
```


```python
sample_input = torch.randn(1, 10, 3)
attention_lstm = AttentionLSTM(input_dim=3)
y = attention_lstm(sample_input)
make_dot(y, params=dict(attention_lstm.named_parameters())).render("attention_lstm_architecture", format="png")

# Visualize Attention LSTM
print("\nAttention LSTM Architecture:")
visualize_model(AttentionLSTM(input_dim=3), "attention_lstm")
```

    
    Attention LSTM Architecture:
    


    
![png](output_20_1.png)
    


## GRU MODEL

### Advantages
- Fewer parameters, faster training  
- Often matches LSTM performance  

### Disadvantages
- May miss long-range patterns  
- Performance varies across datasets  
- Less parallel than attention-based


```python
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        x = self.fc(gru_out[:, -1])  # use last sequence output
        return x.squeeze()
```


```python
sample_input = torch.randn(1, 10, 3)
gru = GRUModel(input_dim=3)
y = gru(sample_input)
make_dot(y, params=dict(gru.named_parameters())).render("gru_architecture", format="png")

# Visualize GRU
print("\nGRU Architecture:")
visualize_model(GRUModel(input_dim=3), "gru")
```

    
    GRU Architecture:
    


    
![png](output_24_1.png)
    


## Training


```python
# Training function
def train_model(model, train_loader, test_loader, epochs=50):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in test_loader:
                y_pred = model(X)
                test_loss += criterion(y_pred, y).item()

        train_losses.append(math.sqrt(train_loss/len(train_loader)))
        test_losses.append(math.sqrt(test_loss/len(test_loader)))

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Train RMSE: {train_losses[-1]:.6f}, Test RMSE: {test_losses[-1]:.6f}')

    return train_losses, test_losses
```

## Data Split
 - Ensuring train data is before test data (No look back bias)



```python
# date prep
sequence_length = 10
dataset = StockDataset(df, sequence_length)

# Calculate split point (80% of data)
split_idx = int(0.8 * len(dataset))

# Print basic dataset info
print("Dataset size:", len(dataset))
print("Split index:", split_idx)

# Get the original timestamps
df_sorted = df.sort_values('datetime')
unique_dates = df_sorted['datetime'].unique()

# Calculate the actual dates considering the sequence structure
# We need sequence_length observations to make first prediction
start_date = unique_dates[sequence_length]
split_date = unique_dates[int(0.8 * (len(unique_dates) - sequence_length)) + sequence_length]
end_date = unique_dates[-1]


# Create train/test splits chronologically
train_dataset = torch.utils.data.Subset(dataset, range(split_idx))
test_dataset = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training sequences: {len(train_dataset)} ({len(train_dataset)/len(dataset)*100:.1f}%)")
print(f"Testing sequences: {len(test_dataset)} ({len(test_dataset)/len(dataset)*100:.1f}%)")

print("\nTrain data from:", start_date)
print("Train data to:", split_date)
print("Test data from:", split_date)
print("Test data to:", end_date)

```

    Dataset size: 33736
    Split index: 26988
    Training sequences: 26988 (80.0%)
    Testing sequences: 6748 (20.0%)
    
    Train data from: 2024-11-20 09:52:00-05:00
    Train data to: 2024-12-27 09:34:00-05:00
    Test data from: 2024-12-27 09:34:00-05:00
    Test data to: 2025-01-06 15:58:00-05:00
    

    C:\Users\kramm\AppData\Local\Temp\ipykernel_9484\757364174.py:19: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at C:\cb\pytorch_1000000000000\work\torch\csrc\utils\tensor_new.cpp:281.)
      self.X = torch.FloatTensor(self.X)
    

## Testing


```python
# Train all models
transformer = TransformerModel(input_dim=3)
rnn = RNNModel(input_dim=3)
gru = GRUModel(input_dim=3)
attention_lstm = AttentionLSTM(input_dim=3)


print("Training Transformer model...")
transformer_losses = train_model(transformer, train_loader, test_loader)

print("\nTraining LSTM model...")
rnn_losses = train_model(rnn, train_loader, test_loader)
print("\nTraining Attention LSTM model...")
attention_lstm_losses = train_model(attention_lstm, train_loader, test_loader)

print("\nTraining GRU model...")
gru_losses = train_model(gru, train_loader, test_loader)



# Plot results
plt.figure(figsize=(12, 6))
plt.plot(transformer_losses[1], label='Transformer RMSE')
plt.plot(rnn_losses[1], label='LSTM RMSE')
plt.plot(gru_losses[1], label='GRU RMSE')
plt.plot(attention_lstm_losses[1], label='Attention LSTM RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Model Comparison: Transformer vs LSTM vs GRU vs Attention LSTM')
plt.legend()
plt.grid(True)
plt.show()

# Print final RMSE scores
print(f"\nFinal Test RMSE:")
print(f"Transformer: {transformer_losses[1][-1]:.6f}")
print(f"LSTM: {rnn_losses[1][-1]:.6f}")
print(f"Attention LSTM: {attention_lstm_losses[1][-1]:.6f}")
print(f"GRU: {gru_losses[1][-1]:.6f}")

```

    c:\ProgramData\anaconda3\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
      warnings.warn(
    

    Training Transformer model...
    Epoch 10, Train RMSE: 0.002281, Test RMSE: 0.003642
    Epoch 20, Train RMSE: 0.002027, Test RMSE: 0.002830
    Epoch 30, Train RMSE: 0.001885, Test RMSE: 0.002859
    Epoch 40, Train RMSE: 0.001865, Test RMSE: 0.003294
    Epoch 50, Train RMSE: 0.001755, Test RMSE: 0.002931
    
    Training LSTM model...
    Epoch 10, Train RMSE: 0.001821, Test RMSE: 0.002830
    Epoch 20, Train RMSE: 0.001743, Test RMSE: 0.002910
    Epoch 30, Train RMSE: 0.001715, Test RMSE: 0.002912
    Epoch 40, Train RMSE: 0.001705, Test RMSE: 0.002917
    Epoch 50, Train RMSE: 0.001692, Test RMSE: 0.002914
    
    Training Attention LSTM model...
    Epoch 10, Train RMSE: 0.001694, Test RMSE: 0.002926
    Epoch 20, Train RMSE: 0.001674, Test RMSE: 0.002863
    Epoch 30, Train RMSE: 0.001674, Test RMSE: 0.002851
    Epoch 40, Train RMSE: 0.001674, Test RMSE: 0.002858
    Epoch 50, Train RMSE: 0.001674, Test RMSE: 0.002862
    
    Training GRU model...
    Epoch 10, Train RMSE: 0.002217, Test RMSE: 0.003121
    Epoch 20, Train RMSE: 0.001902, Test RMSE: 0.002829
    Epoch 30, Train RMSE: 0.001817, Test RMSE: 0.002830
    Epoch 40, Train RMSE: 0.001797, Test RMSE: 0.003226
    Epoch 50, Train RMSE: 0.001741, Test RMSE: 0.002909
    


    
![png](output_30_2.png)
    


    
    Final Test RMSE:
    Transformer: 0.002931
    LSTM: 0.002914
    Attention LSTM: 0.002862
    GRU: 0.002909
    
