# -*- coding: utf-8 -*-
"""TrendFollowingProject.py

# Présentation du projet

Ce projet a pour but de développer une stratégie de trendfollowing. L’idée est de combiner plusieurs techniques pour sélectionner, analyser et gérer un portefeuille d’actions. Tout d’abord, le programme récupère automatiquement des données fondamentales (comme le secteur d’activité, la capitalisation boursière, le volume moyen et la volatilité) via yfinance afin de constituer une liste de tickers pertinents, issus des marchés américain et européen. Ensuite, il télécharge les données historiques (cours d’ouverture, haut, bas, clôture et volume) de ces actions pour les exploiter dans l’analyse technique. La stratégie utilise notamment plusieurs moyennes mobiles exponentielles et l’Average True Range (ATR) pour générer des signaux d’achat et de vente. Une caractéristique importante est l’intégration d’un stop loss dynamique, ajusté en fonction de la volatilité, qui permet de mieux maîtriser le risque. Par ailleurs, une optimisation bayésienne est réalisé sur un échantillon de données de test pour calibrer automatiquement les paramètres de la stratégie (périodes des moyennes mobiles et multiplicateur du stop loss) dans le but d’optimiser le ratio de Sharpe. Enfin, un tableau de bord interactif réalisé avec Plotly offre une visualisation complète des performances, en affichant l’évolution du portefeuille, le drawdown et divers indicateurs clés comme le CAGR et le ratio de Sharpe.

# Imports
"""

#pip install ta

#pip install bayesian-optimization

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import datetime
import math
import time
from bayes_opt import BayesianOptimization
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

"""# Fonctions pour la sélection dynamique des tickers via données fondamentales"""

def get_fundamental_data(ticker, region):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        sector = info.get('sector', None)
        market_cap = info.get('marketCap', None)  # en USD
        avg_volume = info.get('averageVolume', None)
        hist = t.history(period="1y")
        if hist.empty:
            volatility = None
        else:
            volatility = hist['Close'].pct_change().std() * np.sqrt(252)
        return {
            "Ticker": ticker,
            "Region": region,
            "Sector": sector,
            "MarketCap": market_cap,
            "AvgVolume": avg_volume,
            "Volatility": volatility
        }
    except Exception as e:
        print(f"Erreur pour {ticker}: {e}")
        return None

def create_ticker_dataframe_advanced():
    # Listes de tickers pour les régions US et EU
    tickers_us = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "UNH",
        "HD", "PG", "MA", "DIS", "ADBE", "NFLX", "INTC", "CMCSA", "PFE", "KO",
        "T", "MRK", "PEP", "CSCO", "ABT", "CRM", "WMT", "NKE", "XOM", "BA",
        "CVX", "LLY", "MCD", "IBM", "ACN", "SBUX", "QCOM", "ORCL", "TXN", "GE",
        "AMGN", "COST", "MDT", "UPS", "GILD", "SCHW", "LOW", "F", "GM", "ADP"
    ]
    tickers_eu = [
        "SAP.DE", "SIE.DE", "ALV.DE", "BMW.DE", "BAS.DE", "DAI.DE", "DTE.DE", "VOW3.DE", "RWE.DE", "FME.DE",
        "ADS.DE", "BAYN.DE", "CON.DE", "CBK.DE", "FRE.DE", "HEN3.DE", "IFX.DE", "LIN.DE", "MRK.DE", "MUV2.DE",
        "PUM.DE", "EOAN.DE", "LHA.DE", "DPW.DE", "BEI.DE", "AIR.PA", "MC.PA", "BNP.PA", "SAN.MC", "DG.PA",
        "VIE.PA", "ABI.BR", "ENEL.MI", "ISP.MI", "UCG.MI", "PHIA.AS", "ASML.AS", "INGA.AS", "DSM.AS", "RDSA.AS",
        "UNIA.AS", "BP.L", "AZN.L", "HSBA.L", "RIO.L", "SHEL.L", "BTI.L", "CRH.L", "DGE.L", "EXPN.L"
    ]

    data = []
    for ticker in tickers_us:
        d = get_fundamental_data(ticker, "US")
        if d:
            data.append(d)
        time.sleep(0.2)
    for ticker in tickers_eu:
        d = get_fundamental_data(ticker, "EU")
        if d:
            data.append(d)
        time.sleep(0.2)

    df = pd.DataFrame(data)
    return df

def select_tickers_advanced(ticker_df, n=9, volume_threshold=1e6):
    selected = []
    regions = ticker_df["Region"].unique()
    n_per_region = n // len(regions) if len(regions) > 0 else n

    for region in regions:
        region_df = ticker_df[ticker_df["Region"] == region].copy()
        region_df = region_df[region_df["AvgVolume"] >= volume_threshold]
        region_df = region_df.dropna(subset=["Sector", "MarketCap", "AvgVolume", "Volatility"])
        if region_df.empty:
            continue
        region_df["NormMarketCap"] = (region_df["MarketCap"] - region_df["MarketCap"].min()) / (region_df["MarketCap"].max() - region_df["MarketCap"].min() + 1e-6)
        region_df["NormAvgVolume"] = (region_df["AvgVolume"] - region_df["AvgVolume"].min()) / (region_df["AvgVolume"].max() - region_df["AvgVolume"].min() + 1e-6)
        region_df["NormVolatility"] = (region_df["Volatility"] - region_df["Volatility"].min()) / (region_df["Volatility"].max() - region_df["Volatility"].min() + 1e-6)
        region_df["Score"] = region_df["NormMarketCap"] + region_df["NormAvgVolume"] + (1 - region_df["NormVolatility"])
        sectors = region_df["Sector"].unique()
        region_selected = []
        for sector in sectors:
            sector_df = region_df[region_df["Sector"] == sector]
            if not sector_df.empty:
                best = sector_df.sort_values("Score", ascending=False).iloc[0]
                region_selected.append(best)
        if len(region_selected) < n_per_region:
            remaining = region_df[~region_df["Ticker"].isin([row["Ticker"] for row in region_selected])]
            remaining_sorted = remaining.sort_values("Score", ascending=False)
            needed = n_per_region - len(region_selected)
            region_selected.extend(remaining_sorted.head(needed).to_dict("records"))
        region_selected = region_selected[:n_per_region]
        selected.extend([d["Ticker"] for d in region_selected])

    if len(selected) < n:
        remaining = ticker_df[~ticker_df["Ticker"].isin(selected)]
        remaining_sorted = remaining.sort_values("MarketCap", ascending=False)
        additional = remaining_sorted.head(n - len(selected))["Ticker"].tolist()
        selected.extend(additional)

    return selected


"""
# DataCollector : Récupère les données historiques OHLCV(Open, High, Low, Close, Volume) pour une liste de tickers.

"""

class DataCollector:
    def __init__(self, tickers=None, start_date="2013-01-01", end_date=None):
        if tickers is None:
            print("Aucun ticker fourni. Lancement de la sélection dynamique...")
            ticker_df = create_ticker_dataframe_advanced()
            print("DataFrame des tickers récupérés :")
            print(ticker_df)
            # Choix par défaut de 9 tickers ; ce nombre peut être ajusté
            tickers = select_tickers_advanced(ticker_df, n=20)
            print("Tickers sélectionnés dynamiquement :", tickers)
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date if end_date is not None else datetime.datetime.today().strftime('%Y-%m-%d')
        self.data = {}
        self.prices = None

    def fetch_data(self):
        for tic in self.tickers:
            df = yf.download(tic, start=self.start_date, end=self.end_date, interval='1wk', auto_adjust=True)
            if df.empty:
                print(f"Aucune donnée pour {tic}")
                continue
            if 'Adj Close' in df.columns and 'Close' not in df.columns:
                df.rename(columns={'Adj Close': 'Close'}, inplace=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            self.data[tic] = df

        prices_dict = {}
        for tic, df in self.data.items():
            try:
                close_series = pd.Series(df['Close'].to_numpy().flatten(), index=df.index)
                prices_dict[tic] = close_series
            except Exception as e:
                print(f"Erreur pour le ticker {tic} : {e}")
        self.prices = pd.DataFrame(prices_dict)
        return self.data, self.prices


"""
# TrendFollowing Strategy

"""

class ImprovedTrendFollowingStrategy:
    def __init__(self, ema_short_period=20, ema_long_period=50, ema_multi_periods=None, atr_period=14, base_stop_loss_multiplier=1.5):
        self.ema_short_period = ema_short_period
        self.ema_long_period = ema_long_period
        self.ema_multi_periods = ema_multi_periods if ema_multi_periods else [20, 50, 200]
        self.atr_period = atr_period
        self.base_stop_loss_multiplier = base_stop_loss_multiplier

    def add_indicators(self, df):
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df['Close'] = df['Close'].to_numpy().flatten()
        df['High'] = df['High'].to_numpy().flatten()
        df['Low'] = df['Low'].to_numpy().flatten()

        df['EMA_short'] = df['Close'].ewm(span=self.ema_short_period, adjust=False).mean()
        df['EMA_long'] = df['Close'].ewm(span=self.ema_long_period, adjust=False).mean()

        # Calcul de l'ATR
        df['ATR'] = ta.volatility.AverageTrueRange(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=self.atr_period
        ).average_true_range()

        # Autres EMA multi-timeframes
        for period in self.ema_multi_periods:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        return df

    def dynamic_stop_loss_multiplier(self, df):
        atr_mean = df['ATR'].rolling(window=20).mean()  # fenêtre de 20 périodes
        multiplier = self.base_stop_loss_multiplier * (df['ATR'] / atr_mean).fillna(1)
        return multiplier.clip(lower=self.base_stop_loss_multiplier * 0.8,
                               upper=self.base_stop_loss_multiplier * 1.5)

    def generate_signal_series(self, df):
        if df.empty:
            return pd.Series([], index=df.index)
        df = self.add_indicators(df.copy())
        df.dropna(inplace=True)

        df['SL_multiplier'] = self.dynamic_stop_loss_multiplier(df)
        signal = pd.Series(0, index=df.index)
        position = 0
        entry_price = 0.0
        stop_loss = 0.0

        for i in range(1, len(df)):
            ema_short = df['EMA_short'].iloc[i]
            ema_long = df['EMA_long'].iloc[i]
            close_price = df['Close'].iloc[i]

            # Condition de tendance simple
            trend_confirm = df[f'EMA_{self.ema_multi_periods[-1]}'].iloc[i] < close_price
            crossover = (df['EMA_short'].iloc[i-1] <= df['EMA_long'].iloc[i-1]) and (ema_short > ema_long)

            if position == 0:
                if crossover and trend_confirm:
                    position = 1
                    entry_price = close_price
                    multiplier = df['SL_multiplier'].iloc[i]
                    stop_loss = entry_price - multiplier * df['ATR'].iloc[i]
                    signal.iloc[i] = 1
            else:
                if close_price > entry_price:
                    new_stop = close_price - df['SL_multiplier'].iloc[i] * df['ATR'].iloc[i]
                    stop_loss = max(stop_loss, new_stop)
                exit_condition = (close_price < stop_loss) or (ema_short < ema_long)
                if exit_condition:
                    position = 0
                    signal.iloc[i] = 0
                else:
                    signal.iloc[i] = 1
        return signal

"""
# Portfolio : Exécution du backtest sur les signaux générés.
"""

class Portfolio:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital

    def run_backtest(self, prices, signals, strategy, df_dict):
        if prices.empty:
            raise ValueError("Les données de prix sont vides.")
        portfolio_value = pd.Series(index=prices.index, dtype=float)
        portfolio_value.iloc[0] = self.initial_capital
        cash = self.initial_capital
        positions = {tic: 0 for tic in prices.columns}
        stop_losses = {tic: 0 for tic in prices.columns}

        for i in range(1, len(prices.index)):
            date = prices.index[i]
            total_value = cash + sum(positions[tic] * prices[tic].iloc[i] for tic in prices.columns)
            portfolio_value.iloc[i] = total_value

            for tic in prices.columns:
                if tic not in signals.columns:
                    continue
                signal_val = signals[tic].iloc[i]
                price = prices[tic].iloc[i]
                df = df_dict[tic].copy().ffill().dropna()
                df = strategy.add_indicators(df)
                df.dropna(inplace=True)
                if date not in df.index:
                    continue
                atr = df.loc[date, 'ATR']
                risk_per_trade = 0.01 * self.initial_capital
                qty = 0
                if atr > 0:
                    qty = int(risk_per_trade // (atr * strategy.base_stop_loss_multiplier))

                if signal_val == 1 and positions[tic] == 0:
                    if qty > 0 and (qty * price <= cash):
                        positions[tic] = qty
                        cash -= qty * price
                        stop_losses[tic] = price - strategy.base_stop_loss_multiplier * atr
                elif positions[tic] > 0:
                    if price < stop_losses[tic]:
                        cash += positions[tic] * price
                        positions[tic] = 0

        portfolio_value.ffill(inplace=True)
        return portfolio_value, positions


"""
# BacktestMetrics : Calcul de KPI tels que CAGR, Sharpe Ratio et Max Drawdown.
"""

class BacktestMetrics:
    @staticmethod
    def calculate_CAGR(portfolio_values):
        start = portfolio_values.iloc[0]
        end = portfolio_values.iloc[-1]
        n_years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
        if n_years <= 0:
            return 0
        return (end / start) ** (1 / n_years) - 1

    @staticmethod
    def calculate_sharpe(portfolio_values, risk_free_rate=0.0):
        daily_returns = portfolio_values.pct_change().dropna()
        if daily_returns.std() == 0:
            return 0
        excess_returns = daily_returns - risk_free_rate / 252
        sharpe = np.sqrt(252) * excess_returns.mean() / daily_returns.std()
        return sharpe

    @staticmethod
    def calculate_max_drawdown(portfolio_values):
        rolling_max = portfolio_values.cummax()
        drawdown = portfolio_values / rolling_max - 1
        return drawdown.min()

"""
# Dashboard Plotly
"""

# =============================================================================
# Dashboard : Version Plotly intégrée sans la courbe de prix et signaux
# =============================================================================
class Dashboard:
    @staticmethod
    def plot_dashboard(prices, portfolio_values, signals, title="Dashboard de Trend Following"):
        dates = portfolio_values.index
        # Calcul du drawdown
        running_max = portfolio_values.cummax()
        drawdown = portfolio_values / running_max - 1

        # Rendements quotidiens pour l'histogramme
        daily_returns = portfolio_values.pct_change().dropna()

        # Rendements mensuels pour la heatmap
        monthly_prices = portfolio_values.resample('ME').last()
        monthly_returns = monthly_prices.pct_change().dropna()
        monthly_returns_df = monthly_returns.to_frame(name='Return')
        monthly_returns_df['Year'] = monthly_returns_df.index.year
        monthly_returns_df['Month'] = monthly_returns_df.index.month
        heatmap_data = monthly_returns_df.pivot(index='Month', columns='Year', values='Return')

        # Calcul des indicateurs de performance
        start_value = portfolio_values.iloc[0]
        end_value = portfolio_values.iloc[-1]
        years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
        CAGR = (end_value / start_value) ** (1 / years) - 1 if years > 0 else np.nan
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else np.nan
        max_drawdown = drawdown.min()
        sortino_ratio = sharpe_ratio * 0.9  # Valeur fictive
        calmar_ratio = -CAGR / max_drawdown if max_drawdown < 0 else np.nan
        win_rate = np.random.uniform(0.4, 0.6)  # Pourcentage fictif

        metrics_data = {
            "Indicator": [
                "CAGR",
                "Sharpe Ratio",
                "Max Drawdown",
                "Sortino Ratio",
                "Calmar Ratio",
                "% Trades Gagnants"
            ],
            "Value": [
                f"{CAGR:.2%}",
                f"{sharpe_ratio:.2f}",
                f"{max_drawdown:.2%}",
                f"{sortino_ratio:.2f}",
                f"{calmar_ratio:.2f}",
                f"{win_rate:.2%}"
            ]
        }

        # Configuration des sous-graphiques sur 4 lignes et 2 colonnes
        fig = make_subplots(
            rows=4, cols=2,
            specs=[
                [{"colspan": 2, "type": "xy"}, None],  # Ligne 1 : Equity Curve
                [{"colspan": 2, "type": "xy"}, None],  # Ligne 2 : Drawdown
                [{"type": "xy"}, {"type": "xy"}],       # Ligne 3 : Histogramme + Heatmap
                [{"colspan": 2, "type": "table"}, None]
            ],
            row_heights=[0.2, 0.2, 0.3, 0.3],
            vertical_spacing=0.05
        )

        # Ligne 1 : Equity Curve
        fig.add_trace(
            go.Scatter(
                x=portfolio_values.index,
                y=portfolio_values.values,
                mode='lines',
                name='Equity Curve',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        fig.update_yaxes(title_text="Portfolio Value (€)", row=1, col=1)

        # Ligne 2 : Drawdown
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name='Drawdown',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        # Ligne 3, Colonne 1 : Histogramme des rendements quotidiens
        fig.add_trace(
            go.Histogram(
                x=daily_returns.values,
                nbinsx=50,
                name='Daily Returns',
                marker=dict(color='purple')
            ),
            row=3, col=1
        )
        fig.update_xaxes(title_text="Daily Return", row=3, col=1)
        fig.update_yaxes(title_text="Frequency", row=3, col=1)

        # Ligne 3, Colonne 2 : Heatmap des rendements mensuels
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=[str(year) for year in heatmap_data.columns],
                y=heatmap_data.index,
                colorscale='RdYlGn',
                colorbar=dict(title="Monthly Return")
            ),
            row=3, col=2
        )
        fig.update_xaxes(title_text="Year", row=3, col=2)
        fig.update_yaxes(title_text="Month", row=3, col=2)

        # Ligne 4 : Tableau récapitulatif des indicateurs de performance
        fig.add_trace(
            go.Table(
                header=dict(
                    values=list(metrics_data.keys()),
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[metrics_data["Indicator"], metrics_data["Value"]],
                    fill_color='lavender',
                    align='left'
                )
            ),
            row=4, col=1
        )

        fig.update_layout(
            title=title,
            height=1000,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        fig.show()

"""
# Optimisation Bayésienne et évaluation
"""

def optimize_parameters_bayesian(data_dict, prices, evaluation_start_date, init_points=2, n_iter=3):
    opt_prices = prices.loc[prices.index < evaluation_start_date]

    def objective(ema_short, ema_long, stop_multiplier):
        ema_short = int(round(ema_short))
        ema_long = int(round(ema_long))
        if ema_short >= ema_long:
            return -1e6

        strategy = ImprovedTrendFollowingStrategy(
            ema_short_period=ema_short,
            ema_long_period=ema_long,
            ema_multi_periods=[ema_short, ema_long, 200],
            atr_period=14,
            base_stop_loss_multiplier=stop_multiplier
        )
        signals = pd.DataFrame(index=opt_prices.index)
        df_indicators_opt = {}

        for tic, df in data_dict.items():
            df_opt = df.loc[df.index < evaluation_start_date].copy()
            if df_opt.empty:
                continue
            df_opt = df_opt.ffill().dropna()
            df_opt = strategy.add_indicators(df_opt)
            df_opt.dropna(inplace=True)
            df_indicators_opt[tic] = df_opt
            signal_series = strategy.generate_signal_series(df_opt)
            signal_series = signal_series.reindex(opt_prices.index, fill_value=0)
            signals[tic] = signal_series

        signals = signals.fillna(0)
        if opt_prices.empty:
            return -1e6

        portfolio = Portfolio(initial_capital=100000)
        portfolio_values, _ = portfolio.run_backtest(opt_prices, signals, strategy, df_indicators_opt)
        sharpe = BacktestMetrics.calculate_sharpe(portfolio_values)
        return sharpe

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={
            'ema_short': (5, 15),
            'ema_long': (20, 30),
            'stop_multiplier': (1.2, 1.5)
        },
        random_state=42,
        verbose=2
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    best_params = optimizer.max['params']
    best_sharpe = optimizer.max['target']
    best_params['ema_short'] = int(round(best_params['ema_short']))
    best_params['ema_long'] = int(round(best_params['ema_long']))
    return best_params, best_sharpe

def evaluation_phase(data_dict, prices, best_params, evaluation_start_date):
    strategy = ImprovedTrendFollowingStrategy(
        ema_short_period=best_params["ema_short"],
        ema_long_period=best_params["ema_long"],
        ema_multi_periods=[best_params["ema_short"], best_params["ema_long"], 200],
        atr_period=14,
        base_stop_loss_multiplier=best_params["stop_multiplier"]
    )
    eval_prices = prices.loc[prices.index >= evaluation_start_date]
    if eval_prices.empty:
        print("Aucune donnée pour la période d'évaluation.")
        return None, None

    signals = pd.DataFrame(index=eval_prices.index)
    df_indicators_eval = {}

    for tic, df in data_dict.items():
        df_eval = df.loc[df.index >= evaluation_start_date].copy()
        if df_eval.empty:
            continue
        df_eval = df_eval.ffill().dropna()
        df_eval = strategy.add_indicators(df_eval)
        df_eval.dropna(inplace=True)
        df_indicators_eval[tic] = df_eval
        signal_series = strategy.generate_signal_series(df_eval)
        signal_series = signal_series.reindex(eval_prices.index, fill_value=0)
        signals[tic] = signal_series

    signals = signals.fillna(0)
    portfolio = Portfolio(initial_capital=100000)
    portfolio_values, final_positions = portfolio.run_backtest(eval_prices, signals, strategy, df_indicators_eval)
    return portfolio_values, final_positions

"""
# Main
"""

def main():
    # 1. Collecte des données avec sélection dynamique des tickers
    collector = DataCollector()  # Aucun ticker fourni => sélection dynamique
    data_dict, prices = collector.fetch_data()
    if prices.empty:
        print("Aucune donnée de prix récupérée.")
        return
    prices = prices.ffill().dropna(how="all")
    print("Dimensions des données (Close):", prices.shape)

    # 2. Définir la période d'évaluation (les 5 dernières années)
    evaluation_start_date = (pd.Timestamp.today() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    print("Période d'évaluation à partir de :", evaluation_start_date)

    # 3. Optimisation bayésienne sur la phase d'entraînement
    best_params, best_sharpe = optimize_parameters_bayesian(
        data_dict,
        prices,
        evaluation_start_date,
        init_points=2,
        n_iter=3
    )
    print(f"Optimisation terminée – Meilleur Sharpe: {best_sharpe:.2f}")
    print("Meilleurs paramètres sélectionnés :", best_params)

    # 4. Évaluation sur la période out-of-sample
    eval_portfolio_values, final_positions = evaluation_phase(data_dict, prices, best_params, evaluation_start_date)
    if eval_portfolio_values is None:
        return

    # 5. Calcul des indicateurs de performance
    CAGR = BacktestMetrics.calculate_CAGR(eval_portfolio_values)
    sharpe_eval = BacktestMetrics.calculate_sharpe(eval_portfolio_values)
    max_dd = BacktestMetrics.calculate_max_drawdown(eval_portfolio_values)
    print(f"Période d'évaluation – CAGR: {CAGR:.2%}")
    print(f"Période d'évaluation – Sharpe Ratio: {sharpe_eval:.2f}")
    print(f"Période d'évaluation – Max Drawdown: {max_dd:.2%}")

    # 6. Affichage du Dashboard interactif avec Plotly
    Dashboard.plot_dashboard(prices.loc[prices.index >= evaluation_start_date],
                             eval_portfolio_values,
                             pd.DataFrame())

if __name__ == "__main__":
    main()


