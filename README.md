A description of this code from Cladue Code so you don't have to use tokens.
Modifed by a human, the original code writer, for clarity and to provide additional information and suggestions.
Claude got it though.  Wow.  This was expected , for the current state of AI.  

Theory:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4137779

Practice:
https://medium.com/@nttp/intro-to-the-market-vector-auto-regression-application-46ad7cee02b8
https://medium.com/@nttp/market-vector-auto-regression-from-the-command-line-75634b135cec
https://medium.com/@nttp/basic-parameter-scans-of-a-market-vector-auto-regression-model-56b2fc863716

Here's a high-level description of several.py.
  
  What it is:
  
  several.py is a daily financial time-series forecasting and backtesting engine. Given a set of stock and/or
  crypto symbols, it predicts the near-term direction of a target symbol (the first symbol you pass) and
  evaluates how well several statistical models would have traded it historically. It's the standalone/server
  counterpart to a companion iOS / macOS app which was shipped as Market Vector Auto Regression or MVAR,
  but is not currently available on the app stores as of Aug 2026.  The MVAR app ran the python code
  on an AWS server, but the need to reduce costs has led us to park that system for now.  The benefit
  of running locally from a script is myriad, though:  Your computer is likely much faster than
  our inexpensive EC2 node, and you can drive this script with an optimizer or parameter study
  script or program for more advanced work. Also, linux and windows users can rejoice!
  It runs on their systems now! Also you can swap in your own models
  or make other changes to the script (perhaps adding more lags?)  We don't have an easy swap method built in, 
  but since our Model 3 is scikit-learn based, the API format is standard.  We only use 2 days worth of lags
  in our models (a more local model).  With the common view that "ah, everything's already
  priced in after 2 days."  So, yeah, these are not a long-term memory models.
  
  The specific quantity it forecasts is the target's close-minus-open for the next day
  (CLOSE_MINUS_OPEN_TARGET_COL) - effectively "will the target close higher or lower than it opened," which is a
  tradeable up/down signal.

  The pipeline (roughly top-to-bottom)

  1. Parse arguments -- everything is key=val (options) followed by symbols. Options tune the models, costs, data
  keys, etc. A background monitor() thread watches a file named 'running' and SIGKILLs the process if it's deleted
  (remote kill-switch).
  2. Pull data  for each symbol, download ~2000 days of daily OHLCV from Polygon.io (stocks, needs polyiokey)
  or CryptoCompare (crypto, needs cryptocomparekey); a symlookup table routes crypto tickers. Can reuse prior
  data with reuseMergedRaw=1.  [With paid polygon.io plans, you can get more data.  The script would have to be adjusted to
  use this extra data.]
  3. Align by date -- merge the per-symbol CSVs into mergedraw via an inner join on date (only days present in
  all symbols), keeping each symbol's OHLCV. This is the alignInPython / new alignInPandas code block.
  The external program we write to align the data quickly in C/C++ is not provided here, but the Pandas code should
  be faster than our original manual Python method which is kind of slow for larger symbol counts.  However,
  it is acceptable for smaller symbol counts.  If you want to avoid pandas, just set alignInPandas to False
  and it will use our original code.  Since the original code was on Python 2.7, we only used more basic
  Python constructs originally, not the latest fancy libraries.
  4. Build the regression table -- target column = the target's close-open; predictor columns = lagged
  closes/opens (and optionally volatility) of every symbol. Width scales with symbol count (ncolsrt).  Note
  that traditional vector autoregression is a linear model, but we allow modifications to this:
  the weeding out of low contributing variables, and some nonlinear models on the regression table.
  5. Backtest loop (when ntrials >= 0) -- step back day by day, and on each historical day fit three models on a
  trailing (rolling) window (windowsize) and predict the next day's direction:
    - Model 1 -- a coefficient vector optimized by simulated-annealing/BFGS (coolrate) minimizing a min-abs or
  directional objective, with a "poor man's" bootstrap resample of the window (noboot to disable).
    - Model 2 -- ordinary least squares (numpy.linalg.lstsq) for reference.  If your fancy models can't do better than linear, 
      you've got problems.
    - Model 3 -- LARS/Lasso regression or a KNN classifier (uselars / uselogit / lassolarsbic).
  6. Score & tune -- tally directional correctness and confusion matrices (pos/neg * pos/neg) per model, compute
  p-values, and search for the ideal Z-tolerances m1ZTol/m2ZTol/m3ZTol - thresholds that filter out
  low-magnitude (noisy) forecasts to keep only high-conviction signals.
  7. Trading performance -- simulate P&L using shareCount, costPerTrade, allowShorting; compute cumulative gains,
  Sharpe/Sortino ratios (using riskFreeRate), and a buy-and-hold benchmark.  Traders do not live by p-values alone,
  they must backtest.  The trading system is crude but usable as a first guess.  Experts can code in additional 
  trading system complexity if they desire.
  8. Output -- write everything to a JSON file named "status" (forecasts, correctness, confusion, gains, tuned
  tolerances, echoed params) and render PDF plots (cumulative bad-forecast "stairstep" charts, gains-over-time,
  sensitivity).

  Two run modes

  - Backtest (ntrials >= 0): the historical evaluation above; computes the ideal m*ZTol values for the three model backtests.

  - Forecast (ntrials = -1): skips the tuning loop and predicts the next (unknown) day, using m*ZTol supplied on
  the command line -- the exact backtest->forecast handoff the run-defaults.sh automates.

  When you run the forecast, to use the same model that you backtested, 
  the only things you should change are setting ntrials to -1 and 
  passing in the found m*ZTol values that the backtest found (from the output status file, which is in JSON format).  
  These are zero tolerances that suggest "no trade" if the forecast was close to zero (indeterminate). 

  In one sentence

  It downloads multi-asset daily prices, aligns them, and runs three (optionally) bootstrapped statistical models to forecast
  a target's next-day up/down move -- backtesting them to measure directional accuracy, tune indeterminate forecast
  thresholds, and report trading metrics (Sharpe/Sortino/gains vs. buy-and-hold) as JSON and PDF plots.

  One might notice that tuning these indeterminacy thresholds on a backtest may belie the concept of a backtest,
  since you are getting a new parameter from that backtest; not for a model but for a trading system.  
  Hence, you should probably then backtest additionally without a parameter tune 
  (or run the model forward many days) to further test these indeterminacy tolerances.  
  Or, you could just set the indeterminacy tolerances
  to some value that you like a priori and don't use the tuning method.  I don't believe we have
  an easy parameter setting for this yet, so that would require code changes.
