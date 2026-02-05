Step 1 Data acquisition The project begins by downloading historical financial time series data using the yfinance library The chosen dataset is Apple stock data which provides realistic non-stationary behavior The Close price is used as the prediction target and Volume is included as an exogenous variable Missing values caused by market holidays are removed to ensure sequence continuity

Step 2 Feature engineering Raw price alone is insufficient for deep time series models so additional temporal features are created Percentage returns capture short-term momentum Rolling means over 5 and 10 days capture local trends Rolling standard deviation captures volatility These features transform the raw signal into a multivariate time series suitable for neural networks Rows with NaN values introduced by rolling windows are dropped

Step 3 Normalization Neural networks are sensitive to scale so StandardScaler is applied separately to input features and target values The scalers are fit only once and later used to inverse-transform predictions This prevents data leakage and ensures stable training

Step 4 Time-series split Unlike random splits this project uses chronological splitting The dataset is divided into training validation and test sets using fixed time boundaries This preserves temporal causality which is critical for forecasting tasks

Step 5 Sequence construction The time series is converted into supervised learning format Each input sample consists of a fixed-length sliding window of past observations defined by SEQ_LEN The target is the next time step value This structure enables sequence-to-one forecasting while remaining compatible with Seq2Seq models

Step 6 Dataset and DataLoader A custom PyTorch Dataset class generates sequences on demand DataLoader handles batching and memory efficiency Shuffling is disabled to preserve temporal order which is essential for time series learning

Step 7 Baseline LSTM model The baseline model is a standard LSTM with a single recurrent layer The model processes the input sequence and uses the final hidden state to generate a prediction through a linear layer This model establishes a reference performance level without attention

Step 8 Attention mechanism The Bahdanau attention module computes relevance scores between the decoder hidden state and all encoder outputs A softmax converts these scores into attention weights A context vector is produced as a weighted sum of encoder outputs This allows the model to focus on the most informative time steps

Step 9 Seq2Seq with attention model The encoder LSTM processes the full input sequence and outputs hidden states for every time step The final hidden state is combined with the attention-derived context vector Both are concatenated and passed through a fully connected layer to produce the final forecast This architecture improves interpretability and long-range dependency modeling

Step 10 Loss function Mean Squared Error is used as the optimization objective because it penalizes large forecasting errors and is standard in regression-based time series forecasting

Step 11 Training loop For each epoch batches are passed through the model Gradients are computed using backpropagation Adam optimizer updates parameters Learning rate and batch size are explicitly controlled to ensure stable convergence

Step 12 Evaluation procedure During evaluation gradient computation is disabled Predictions are collected across the entire test set Outputs are inverse-scaled back to original price units to allow meaningful interpretation

Step 13 Performance metrics Root Mean Squared Error measures overall prediction accuracy Mean Absolute Error provides robustness to outliers Directional Accuracy measures the model’s ability to correctly predict price movement direction which is especially important in financial forecasting

Step 14 Baseline comparison Both the baseline LSTM and attention-augmented model are trained and evaluated under identical conditions Performance metrics are printed side-by-side This directly demonstrates the impact of the attention mechanism on accuracy and directional performance

Step 15 Visualization Predicted and actual prices are plotted over the same time horizon Visual inspection highlights regions where the attention model tracks trend reversals and volatility more effectively than the baseline

Step 16 Final deliverables The single Python file contains production-quality code with modular structure clear model definitions training logic and evaluation The printed metrics plots and model comparison support a rigorous technical report analyzing convergence speed performance gains and interpretability benefits from attention
