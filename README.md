# Прогнозирование выбора банка клиентом

ML-проект на банковских данных: анализ признаков, построение пайплайна предобработки и сравнение моделей бинарной классификации. Модель оценивает вероятность того, что клиент выберет банк и откроет счёт.

**Лучший сохранённый результат — ROC-AUC 0,7637 ± 0,0031 для CatBoost на стратифицированной 5-fold кросс-валидации.**

## Задача и данные

Данные относятся к [соревнованию на Kaggle](https://www.kaggle.com/competitions/bank-issues-042022).

| Набор | Количество строк | Назначение |
|---|---:|---|
| `train.csv` | 119 518 | Обучение и кросс-валидация; содержит целевую переменную |
| `test.csv` | 51 222 | Получение вероятностей для клиентов без известных меток |
| `sampleSubmission.csv` | 51 222 | Пример формата файла для отправки |

В обучающей выборке 21 070 положительных примеров — около **17,63%**. Для сохранения соотношения классов используется стратификация.

Входные данные содержат идентификатор клиента и 13 признаков:

- **Характеристики клиента:** `gender`, `age`, `marital_status`, `job_position`, `education`, `living_region`.
- **Параметры кредитного предложения:** `credit_sum`, `credit_month`, `tariff_id`.
- **Финансовые характеристики:** `score_shk`, `monthly_income`, `credit_count`, `overdue_credit_count`.

Целевая переменная — **`open_account_flg`**. Идентификатор `client_id` сохраняется в итоговом файле, но исключается из признаков финальным `ColumnTransformer`.

## Подход

### Анализ и исследование признаков

В [основном ноутбуке](feature_engineering.ipynb) исследуются пропуски, распределения, корреляции и выбросы. Сравниваются методы масштабирования, рассматриваются полиномиальные признаки и отбор признаков с помощью `SequentialFeatureSelector` и `SelectFromModel`.

### Финальная предобработка

| Признаки | Преобразования |
|---|---|
| `age` | Заполнение медианой, индикатор пропуска при наличии пропусков на обучении, ограничение выбросов по IQR, `StandardScaler` |
| `credit_sum`, `credit_month`, `monthly_income`, `credit_count` | Заполнение медианой, индикаторы пропусков при наличии пропусков на обучении, преобразование Yeo–Johnson |
| `score_shk`, `overdue_credit_count` | Заполнение медианой, индикаторы пропусков при наличии пропусков на обучении, `StandardScaler` |
| `tariff_id`, `gender`, `marital_status`, `job_position`, `education` | Заполнение наиболее частым значением, `OneHotEncoder` с обработкой неизвестных категорий |
| `living_region` | Очистка названий регулярными выражениями, объединение категорий с частотой менее 100 в «ДРУГОЙ», `BinaryEncoder` |

Для обработки регионов и выбросов реализованы собственные трансформеры: `LivingRegionCleaner`, `RareCategoryGrouper` и `IQRWinsorizer`.

Предобработка и классификатор объединены в `Pipeline`. При кросс-валидации обучаемые преобразования заново подгоняются на тренировочной части каждого фолда; валидационная часть используется только для оценки. При этом решения о составе предобработки приняты по исследованию исходной обучающей выборки — отдельная внешняя проверка этих решений не проводилась.

### Обучение и оценка

- Сравниваются **Logistic Regression, LightGBM, XGBoost и CatBoost**.
- Применяется `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Основная метрика — **ROC-AUC**. Для логистической регрессии дополнительно рассчитывается `average_precision_score`.
- В Logistic Regression и LightGBM используется `class_weight="balanced"`.
- Модель с максимальным средним ROC-AUC обучается на всей размеченной выборке и формирует `submission.csv`.

## Результаты

Метрики взяты из сохранённых выводов ноутбука. Значение после ± — стандартное отклонение по пяти фолдам.

| Модель | Средний ROC-AUC | Стандартное отклонение |
|---|---:|---:|
| Logistic Regression | 0,7163 | 0,0033 |
| LightGBM | 0,7628 | 0,0023 |
| XGBoost | 0,7635 | 0,0026 |
| **CatBoost** | **0,7637** | **0,0031** |

CatBoost превосходит линейный baseline примерно на **0,0474 ROC-AUC**. Результаты трёх бустингов близки; небольшая разница средних сама по себе не подтверждает статистически значимого превосходства CatBoost.

Для Logistic Regression средний Average Precision составляет **0,3551 ± 0,0038**. В выводе ноутбука эта метрика подписана как PR-AUC, но вычисляется именно `average_precision_score`.

Результат на закрытой тестовой выборке Kaggle в репозитории не зафиксирован. Указанные метрики относятся к локальной кросс-валидации.

## Стек

**Python, NumPy, Pandas, SciPy, Scikit-learn, Category Encoders, LightGBM, XGBoost, CatBoost, Matplotlib, Seaborn, Jupyter.**

## Структура репозитория

| Файл или папка | Содержимое |
|---|---|
| [feature_engineering.ipynb](feature_engineering.ipynb) | Основной ноутбук: EDA, эксперименты с признаками, финальный пайплайн, сравнение моделей и предсказания |
| `bank-issues-data/` | Обучающая и тестовая выборки, пример submission |
| `bank-issues-data.zip` | Архив исходных данных |
| `submission.csv` | Сохранённые вероятности для тестовых клиентов |
| `places.txt` | Вспомогательный список названий регионов |
| `*.png` | Графики распределений, корреляций, выбросов и масштабирования |
| `catboost_info/` | Логи обучения CatBoost |
| `imputation.ipynb`, `regulars.ipynb` | Пустые файлы; для запуска не используются |

## Запуск

### 1. Подготовить окружение

```bash
git clone https://github.com/DaryaGudkova/credit_score.git
cd credit_score
python -m venv .venv
```

Активировать окружение в Linux/macOS:

```bash
source .venv/bin/activate
```

В Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Установить библиотеки и открыть ноутбук:

```bash
python -m pip install numpy pandas scipy "scikit-learn>=1.2" category_encoders lightgbm xgboost catboost matplotlib seaborn jupyterlab
python -m jupyterlab feature_engineering.ipynb
```

### 2. Проверить данные

```bash
python -m zipfile -e bank-issues-data.zip .
```

### 3. Запустить финальное сравнение моделей

1. Перезапустить ядро и выполнить в новой ячейке:

   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt

   from sklearn.model_selection import train_test_split, StratifiedKFold
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import roc_auc_score, average_precision_score
   ```

2. Последовательно выполнить код от раздела **«Полный пайплайн для предобработки данных»** до конца ноутбука.
3. После завершения получить обновлённый `submission.csv` в корне проекта.

В файле результата две колонки: `client_id` и `open_account_flg`. Во второй записана вероятность положительного класса.
