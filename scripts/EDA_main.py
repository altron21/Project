from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from IPython.display import (display)
# показывать все столбцы
pd.set_option('display.max_columns', None)

pd.set_option('display.width', 0)          # 0 = автоопределение ширины
pd.set_option('display.max_colwidth', None)  # не обрезать длинные строки
plt.style.use("ggplot")
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def load_and_prepare(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    rename_map = {
        "Name": "model_full",
        "Year": "year",
        "Mileage": "mileage",
        "Engine_capacity": "engine_capacity",
        "Engine_type": "engine_type",
        "Power": "power_hp",
        "Transmission": "transmission",
        "Drive": "drive",
        "Fuel_consumption_mixed": "fuel_consumption",
        "Acceleration_to_100": "acceleration_100",
        "Cost": "cost",
    }
    df = df.rename(columns=rename_map)
    # переделываем в снейк-кейс для однообразия
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^a-z0-9_]+", "", regex=True)
    )

    # выделим численные характеристики
    numeric_cols = [
        "year",
        "mileage",
        "engine_capacity",
        "power_hp",
        "fuel_consumption",
        "acceleration_100",
        "cost",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # удалим прям явные выбросы и дупликаты если есть
    df = df.drop_duplicates()
    df = df[df["cost"].notna()]  # цена обязательна
    df = df[df["year"].between(1980, 2025)]

    return df.reset_index(drop=True)


def quick_eda(df: pd.DataFrame) -> None:
    # выведем основные стат показатели
    display(df.head())
    display(df.describe(include="all", percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T)
    for col in ["engine_type", "transmission", "drive"]:
        display(df[col].value_counts().to_frame(name="counts"))


def plot_distributions(df: pd.DataFrame) -> None:
    # нарисуем график расрпределения цен
    plt.figure(figsize=(8, 5))
    sns.histplot(df["cost"], bins=40, kde=True, edgecolor="black")
    plt.title("Распределение цен объявлений (Avito)")
    plt.xlabel("Цена, ₽")
    plt.ylabel("Количество объявлений")
    plt.tight_layout()
    plt.show()


def plot_price_vs_year(df: pd.DataFrame) -> None:
    # график расрпределения цен по годам (предполагаем, что это основной фактор)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="year",
        y="cost",
        hue="transmission",
        alpha=0.6,
    )
    plt.title("Цена ~ Год выпуска (цвет: тип коробки)")
    plt.ylabel("Цена, ₽")
    plt.tight_layout()
    plt.show()


def plot_price_by_drive(df: pd.DataFrame) -> None:
    # график расрпределения цен по типу привода
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="drive", y="cost")
    plt.title("Распределение цен по типу привода")
    plt.ylabel("Цена, ₽")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    # график корреляции между числовыми признаками, чтобы найти неявные зависимости и лучше понять как признаки влияют друг на друга и на цену
    numeric_cols = df.select_dtypes(include="number").columns
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
    )
    plt.title("Корреляция числовых признаков")
    plt.tight_layout()
    plt.show()


def baseline_regression(df: pd.DataFrame) -> pd.Series:
    # сделаем базовую регрессию, чтобы понять как цена зависит от других признаков и выведем коэффициенты сдвига и наклона
    df = df[df["cost"] > 0]

    # Выбираем признаки и логарифмируем целевую переменную
    features = ["year", "mileage", "engine_capacity", "power_hp"]
    X = df[features]
    y = np.log(df["cost"])  # логарифмируем цену

    # Делим на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Обучаем модель
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Получаем предсказания в логарифмах
    y_pred_log = model.predict(X_test)

    # Переводим предсказания и реальные значения обратно в рубли
    y_pred = np.exp(y_pred_log)
    y_true = np.exp(y_test)

    # Метрики на исходной шкале
    print("\nModel quality on hold‑out set:")
    print(f"  MAE : {mean_absolute_error(y_true, y_pred):,.0f} р")
    print(f"  R^2  : {r2_score(y_true, y_pred):.3f}")

    # Коэффициенты регрессии
    coeffs = pd.Series(model.coef_, index=features)
    coeffs.name = "slope (log model)"
    coeffs = coeffs.sort_values(ascending=False)
    display(coeffs.to_frame())
    print("bias =", model.intercept_)

    return coeffs


if __name__ == "__main__":
    df_cars = load_and_prepare(Path("../data/avito_bmw_cars.csv"))

    # print("...........Quick EDA...........")
    quick_eda(df_cars)

    # print("...........Distributions...........")
    plot_distributions(df_cars)

    # print("...........Price vs Year...........")
    plot_price_vs_year(df_cars)

    # print("...........Price by Drive...........")
    plot_price_by_drive(df_cars)

    # print("...........Correlation Map...........")
    plot_correlation_heatmap(df_cars)

    # print("...........Baseline reg...........")
    baseline_regression(df_cars)
