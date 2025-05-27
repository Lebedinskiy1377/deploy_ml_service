import logging

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from demand_predictor import DemandPredictor
from db import engine


def optimize_price(
        data: pd.DataFrame,
        predictor: DemandPredictor,
        lambda_param: float = 0.5,
        target_margin: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Оптимизирует цену для каждого SKU, максимизируя функцию полезности f(gmv, margin),
    и генерирует визуализацию зависимости спроса, GMV и функции полезности от цены.

    Args:
        data (pd.DataFrame): Обработанные данные с признаками для каждого SKU.
        predictor (DemandPredictor): Модель для прогнозирования спроса.
        lambda_param (float): Коэффициент штрафа за отклонение от целевой маржи.
        target_margin (float): Целевая маржа (например, 0.3 для 30%).

    Returns:
        List[Dict[str, Any]]: Список словарей с оптимальными ценами, метриками и ссылками на графики.
    """
    results: List[Dict[str, Any]] = []
    prices: pd.DataFrame = pd.read_sql_query("SELECT * FROM prices", engine)

    data = pd.merge(data, prices, on='SKU', how='left')

    data['base_demand'] = predictor.predict(data[['SKU', 'week_num', 'year', 'discount', 'fincode', 'ui1_code',
         'ui2_code', 'ui3_code', 'vendor', 'brand_code', 'week_num_expiration',
         'year_expiration', 'week_num_creation', 'year_creation', 'day',
         'month', 'weekday', 'price']])

    for idx, row in data.iterrows():
        sku: np.int64 = np.int64(row['SKU'])
        week_num: np.uint32 = np.uint32(row['week_num'])
        year: np.int32 = np.int32(row['year'])
        discount: np.float64 = np.float64(row['discount'])
        fincode: str = str(row['fincode'])
        ui1_code: str = str(row['ui1_code'])
        ui2_code: str = str(row['ui2_code'])
        ui3_code: str = str(row['ui3_code'])
        vendor: str = str(row['vendor'])
        brand_code: str = str(row['brand_code'])
        week_num_expiration: np.uint32 = np.uint32(row['week_num_expiration'])
        year_expiration: np.int32 = np.int32(row['year_expiration'])
        week_num_creation: np.uint32 = np.uint32(row['week_num_creation'])
        year_creation: np.int32 = np.int32(row['year_creation'])
        day: np.int32 = np.int32(row['day'])
        month: np.int32 = np.int32(row['month'])
        weekday: np.int32 = np.int32(row['weekday'])
        base_price: np.float64 = np.float64(row['price'])
        cost: np.float64 = np.float64(row['cost'])
        base_demand: np.float64 = np.float64(row['base_demand'])

        row_data = pd.DataFrame({
            'SKU': [sku],
            'week_num': [week_num],
            'year': [year],
            'discount': [discount],
            'fincode': [fincode],
            'ui1_code': [ui1_code],
            'ui2_code': [ui2_code],
            'ui3_code': [ui3_code],
            'vendor': [vendor],
            'brand_code': [brand_code],
            'week_num_expiration': [week_num_expiration],
            'year_expiration': [year_expiration],
            'week_num_creation': [week_num_creation],
            'year_creation': [year_creation],
            'day': [day],
            'month': [month],
            'weekday': [weekday],
            'price': [base_price]
        })

        row_data['SKU'] = row_data['SKU'].astype(np.int64)
        row_data['week_num'] = row_data['week_num'].astype(np.uint32)
        row_data['year'] = row_data['year'].astype(np.int32)
        row_data['discount'] = row_data['discount'].astype(np.float64)
        row_data['fincode'] = row_data['fincode'].astype('category')
        row_data['ui1_code'] = row_data['ui1_code'].astype('category')
        row_data['ui2_code'] = row_data['ui2_code'].astype('category')
        row_data['ui3_code'] = row_data['ui3_code'].astype('category')
        row_data['vendor'] = row_data['vendor'].astype('category')
        row_data['brand_code'] = row_data['brand_code'].astype('category')
        row_data['week_num_expiration'] = row_data['week_num_expiration'].astype(np.uint32)
        row_data['year_expiration'] = row_data['year_expiration'].astype(np.int32)
        row_data['week_num_creation'] = row_data['week_num_creation'].astype(np.uint32)
        row_data['year_creation'] = row_data['year_creation'].astype(np.int32)
        row_data['day'] = row_data['day'].astype(np.int32)
        row_data['month'] = row_data['month'].astype(np.int32)
        row_data['weekday'] = row_data['weekday'].astype(np.int32)
        row_data['price'] = row_data['price'].astype(np.float64)

        price_candidates: np.ndarray = np.linspace(base_price * 0.7, base_price * 1.3, 30)

        demands: np.ndarray = predictor.adjust_demand_with_price(row_data, price_candidates)
        print(f'prices: {price_candidates}, demands: {demands}')

        scores: np.ndarray = np.zeros_like(price_candidates)
        margins: np.ndarray = np.zeros_like(price_candidates)

        for i, (price, demand) in enumerate(zip(price_candidates, demands)):
            margin: np.float64 = (price - cost) / price if price > 0 else 0.0
            margins[i] = margin
            penalty: np.float64 = lambda_param * max(0, target_margin - margin)
            scores[i] = price * demand * (1 - penalty)

        best_idx: int = np.argmax(scores)
        best_price: np.float64 = price_candidates[best_idx]
        best_demand: np.float64 = demands[best_idx]
        best_score: np.float64 = scores[best_idx]

        results.append({
            'SKU': int(sku),
            'optimal_price': float(best_price),
            'expected_demand': float(best_demand),
            'gmv': float(best_price * best_demand),
            'margin': float((best_price - cost) / best_price if best_price > 0 else 0.0),
            'score': float(best_score),
            'base_demand': base_demand,
        })

    return pd.DataFrame(results).merge(prices, on='SKU', how='left')
