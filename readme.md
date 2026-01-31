<h1>Next-Day Stock Return Prediction (Ensemble Learning)</h1>

<h2>Overview</h2>

<p>
This project predicts a stock’s next-day adjusted-close return and price using:
</p>

<ul>
  <li>The stock’s own historical behavior</li>
  <li>Market proxy (QQQ)</li>
  <li>Sector proxy (peer-based index)</li>
</ul>

<p>
It applies leakage-safe walk-forward validation and an ensemble (stacking) approach using multiple regression models.<br>
The system is designed for research and educational purposes only.
</p>

<p><strong>Full evaluation metrics and test results are documented at the end of this README.</strong></p>


<h2>Dataset</h2>

<ul>
  <li><strong>Source:</strong> Kaggle — Stock Market Dataset (NASDAQ)</li>
  <li><strong>Link:</strong> https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset</li>
  <li><strong>Format:</strong> One CSV per ticker (OHLCV + Adj Close)</li>
  <li><strong>End Date:</strong> April 1, 2020</li>
  <li><strong>Used Price:</strong> Adj Close</li>
</ul>

<p>
Place all CSV files inside the <code>data/</code> directory.
</p>


<h2>Project Structure</h2>

<pre>
project/
│
├── data/                  # Kaggle CSV files
├── artifacts/             # Saved models, features, outputs
├── src/                   # Core pipeline modules
│   ├── data.py
│   ├── features.py
│   ├── sector.py
│   ├── models.py
│   ├── stack.py
│   ├── evaluate.py
│   └── utils.py
│
├── train.py
├── batch_train.py
└── predict_live.py
</pre>


<h2>Core Scripts</h2>

<h3>train.py</h3>

<p>Main training and evaluation pipeline.</p>

<p><strong>Responsibilities:</strong></p>

<ul>
  <li>Loads full ticker universe</li>
  <li>Computes daily returns</li>
  <li>Selects sector peers (correlation-based)</li>
  <li>Builds leakage-safe features</li>
  <li>Creates walk-forward splits (Train/Val/Test)</li>
  <li>Trains base models</li>
  <li>Trains stacking meta-model</li>
  <li>Evaluates on test set</li>
  <li>Saves artifacts and predictions</li>
</ul>

<p><strong>Output:</strong></p>

<ul>
  <li>Trained models (.pkl)</li>
  <li>Meta model</li>
  <li>Feature parquet file</li>
  <li>Test prediction CSV</li>
</ul>


<h3>batch_train.py</h3>

<p>Runs training for multiple tickers sequentially.</p>

<p><strong>Responsibilities:</strong></p>

<ul>
  <li>Iterates over predefined tickers</li>
  <li>Calls train.py for each</li>
  <li>Handles failures safely</li>
</ul>

<p>Used for multi-stock experiments.</p>


<h3>predict_live.py</h3>

<p>Generates a next-day prediction using trained models.</p>

<p><strong>Responsibilities:</strong></p>

<ul>
  <li>Loads saved base and meta models</li>
  <li>Loads latest available data</li>
  <li>Rebuilds features</li>
  <li>Uses stored sector peers</li>
  <li>Predicts next-day return and price</li>
</ul>

<p>
This simulates “live” inference on the most recent date in the dataset.
</p>


<h2>src/ Module Overview</h2>

<h3>src/data.py</h3>

<ul>
  <li>Loads all CSV files</li>
  <li>Builds ticker → DataFrame mapping</li>
</ul>


<h3>src/features.py</h3>

<p>Computes:</p>

<ul>
  <li>Lagged returns</li>
  <li>Rolling statistics</li>
  <li>Volume signals</li>
  <li>Technical gaps</li>
</ul>

<p>Builds expanding OLS factor features</p>


<h3>src/sector.py</h3>

<ul>
  <li>Selects peers using return correlations</li>
  <li>Builds peer-based sector index</li>
</ul>


<h3>src/models.py</h3>

<p>Defines base learners:</p>

<ul>
  <li>Elastic Net</li>
  <li>Random Forest</li>
  <li>Gradient Boosting</li>
</ul>

<p>Handles fitting and persistence</p>


<h3>src/stack.py</h3>

<ul>
  <li>Trains meta-model for stacking</li>
  <li>Uses validation predictions</li>
</ul>


<h3>src/evaluate.py</h3>

<ul>
  <li>Computes regression metrics</li>
  <li>Saves prediction outputs</li>
</ul>


<h3>src/utils.py</h3>

<ul>
  <li>Directory creation</li>
  <li>Pickle save/load helpers</li>
</ul>


<h2>Methodology Summary</h2>

<h3>Target</h3>

<p>Next-day return:</p>

<pre>
r(t+1) = (AdjClose(t+1) - AdjClose(t)) / AdjClose(t)
</pre>

<p>Derived price:</p>

<pre>
P̂(t+1) = P(t) × (1 + r̂(t+1))
</pre>


<h2>Features (Leakage-Safe)</h2>

<p>All features use past data only:</p>

<ul>
  <li>Own-stock lags and rolling stats</li>
  <li>Market (QQQ) returns</li>
  <li>Sector index returns</li>
  <li>Expanding OLS betas and residuals</li>
  <li>Calendar features</li>
</ul>

<p>Scalers are fitted on training data only.</p>


<h2>Sector Construction</h2>

<ul>
  <li>Correlation-based peer selection (Train only)</li>
  <li>Top-K peers (default: 10)</li>
  <li>Equal-weighted return average</li>
  <li>Peers are frozen after training</li>
</ul>


<h2>Models</h2>

<p><strong>Base learners:</strong></p>

<ul>
  <li>Elastic Net</li>
  <li>Random Forest</li>
  <li>Gradient Boosting</li>
</ul>

<p><strong>Ensemble:</strong><br>
Stacking with linear meta-model
</p>

<p><strong>Training order:</strong></p>

<ol>
  <li>Train base models on Train</li>
  <li>Generate validation predictions</li>
  <li>Train meta-model</li>
  <li>Refit bases on Train+Val</li>
  <li>Test on frozen model</li>
</ol>


<h2>Installation</h2>

<pre>
pip install -r requirements.txt
</pre>


<p><strong>Main dependencies:</strong></p>

<ul>
  <li>pandas</li>
  <li>numpy</li>
  <li>scikit-learn</li>
  <li>pyarrow</li>
  <li>joblib</li>
</ul>


<h2>Usage</h2>

<h3>1. Prepare Data</h3>

<p>
Download Kaggle dataset and extract to:<br>
<code>data/</code>
</p>


<h3>2. Train Single Stock</h3>

<pre>
python train.py
</pre>

<p>
Default ticker: AAPL<br>
To modify, edit main() arguments.
</p>


<h3>3. Train Multiple Stocks</h3>

<pre>
python batch_train.py
</pre>

<p>Edit TICKERS list inside the file.</p>


<h3>4. Run Inference</h3>

<pre>
python predict_live.py
</pre>

<p>
Outputs predicted return and price for the latest date.
</p>


<h2>Outputs</h2>

<p>All results are saved in:</p>

<pre>
artifacts/
</pre>

<p>Includes:</p>

<ul>
  <li>Trained models</li>
  <li>Meta model</li>
  <li>Features (Parquet)</li>
  <li>Test predictions (CSV)</li>
  <li>Sector peers (Pickle)</li>
</ul>


<h2>Design Principles</h2>

<ul>
  <li>Strict time-aware splitting</li>
  <li>No future data leakage</li>
  <li>Validation-only tuning</li>
  <li>Reproducible pipelines</li>
  <li>Modular architecture</li>
</ul>


<h2>Limitations</h2>

<ul>
  <li>No transaction costs</li>
  <li>No slippage modeling</li>
  <li>No portfolio construction</li>
  <li>Dataset ends in 2020</li>
  <li>Not suitable for real trading</li>
</ul>

<p>
This project is for academic and experimental use.
</p>


<h2>Disclaimer</h2>

<p>
This project does not provide financial advice.<br>
All outputs are for learning and research only.<br><br>
<strong>Reproducibility</strong>: Data and trained models are not included. 
Kindly download the dataset from Kaggle and run training scripts.
</p>


<h2><strong>Model Performance</strong></h2>

<p>
All models were evaluated on a held-out test set (2020 Q1: Jan–Mar) using a walk-forward, leakage-safe
training procedure. Metrics were generated by running:
</p>

<pre>
python batch_train.py
</pre>

<h3>Test Set Results (2020 Q1)</h3>

<table>
  <thead>
    <tr>
      <th>Ticker</th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>Directional Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>AAPL</td>
      <td>0.0288</td>
      <td>0.0413</td>
      <td>59.7%</td>
    </tr>
    <tr>
      <td>MSFT</td>
      <td>0.0300</td>
      <td>0.0425</td>
      <td>54.8%</td>
    </tr>
    <tr>
      <td>GOOGL</td>
      <td>0.0244</td>
      <td>0.0348</td>
      <td>56.5%</td>
    </tr>
  </tbody>
</table>


<h3>Metric Definitions</h3>

<ul>
  <li><strong>MAE:</strong> Mean Absolute Error of predicted returns</li>
  <li><strong>RMSE:</strong> Root Mean Squared Error of predicted returns</li>
  <li><strong>Directional Accuracy:</strong> Percentage of days where the predicted return sign matches the true return</li>
</ul>


<h3>Remarks on Model Effectiveness (Overview)</h3>

<p>
Stock return prediction is a high-noise, low-signal problem. Short-horizon equity returns are known to be
difficult to forecast due to market efficiency, regime shifts, and external shocks.
</p>

<p>
Directional accuracies in the range of 55–60% are considered non-trivial in daily return forecasting and
indicate that the ensemble is extracting weak but consistent signals from historical, market, and sector data.
</p>

<p>
The primary goal of this project is not to maximize short-term trading profitability, but to demonstrate:
</p>

<ul>
  <li>Leakage-safe time-series modeling</li>
  <li>Robust feature engineering</li>
  <li>Ensemble learning with stacking</li>
  <li>Reproducible evaluation pipelines</li>
</ul>

<p>
Performance is expected to vary across tickers and market regimes. Results may be improved through
additional features, alternative sector definitions, regime detection, and longer training histories.
</p>

<p>
All reported metrics are produced directly by the training pipeline and can be reproduced by following the
instructions in this repository.
</p>