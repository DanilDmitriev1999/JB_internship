# JB_internship
Это мое решение тестового задания от JetBrains. Задача заключалась в определение токсичности комментария.
Все эксперименты доступны на [Comet.ml](https://www.comet.ml/danildmitriev1999/jetbrainsinternship?shareable=vn9P5E9MO890e9IJlH2cSK1Pg)

# Данные
Данные были взять из библиотеки `Hugging Face Dataset`. В данных были только обучающие данные с преобладающим 
позитивных (нейтральных) комментариев. Для получения данных для теста и валидации использовал `train_test_split` 
с фиксированным сидом (сиды фиксировал с помощью `seed_everything(294)` из `pytorch_lightning`).

## Подготовка данных
Данные из библиотеки уже примерно готовы к обучению, персоны были заменены на *user*, единственное, я удалил знаки 
препинания и user.

# Метрики
В задание просили *accuracy*, но также я вычислял *f-score*, так как данные несбалансированны.

# Подходы
Так как данные несбалансированны я применял подход UnderSampling (уменьшил в 3 раза данные преобладающего класс).
В качестве лосса использовал Local Loss (сам лосс находится в папке loss). Он показал лучше, чем BCE.

Использовал 3 типа предобученных моделей:
1. DistilRoBERTa - то что требовалось, только быстрее обучается
2. RoBERTa - требовалось в задание.
3. DeBERTa - новая интересная модель с измененным механизмом внимания.

## Классификация пулинга из модели
Классический подход. Получал пулинг из БЕРТподомной модели и проводил его через линейный слой.
Модель с таким подходом находится в `models/Roberta.py`.

## Классификация контактинации 3 скрытых слое
Необходимость данного подхода вызвало использование `DeBERTa`. В текущей реализации в *Hugging Face* нельзя вернуть 
пулинг модели и прошлось контактировать скрытые состояние модели. В начале я использовал 3 последних слоя, но
в статье *Revealing the Dark Secrets of BERT* проводили эксперименты с распределением внимания в задачах GLUE и авторы
показали, что внимание сосредоточено в центральных слоях. Так и я контактировал *6, 7, 8* слой в RoBERTa и в DeBERTa, а 
в DistilRoBERTa - 3, 4, 5. Именно данных подход дал наибольшую метрику, как *accuracy*, так и *F-score*.

## Прочие подходы
Я также добавлял слой biGRU в модель, но прироста это не дало. Такие примеры есть в Comet.ml.

## Leaderboard
| Model Name                          | F-score | Accuracy | Experiment name in Comet.ml                                               |
|-------------------------------------|---------|----------|---------------------------------------------------------------------------|
| Deberta-base (6, 7, 8 layers)       | **0.7856**  | **0.9721**   | Focal Loss deberta-base (5 epoch; 6, 7, 8 layers): UnderSampling          |
| DistilRoberta-base-pooling          | 0.7673  | 0.9679   | Focal Loss distilroberta-base UnderSampling                               |
| Deberta-base (last 3 layers)        | 0.7526  | 0.9680   | Focal Loss deberta-base (5 epoch, last 3 layers): Vanila Model            |
| RoBERTa-base (6, 7, 8 layers)       | 0.7414  | 0.9663   | Focal Loss (6, 7, 8 layers) roberta-base UnderSampling                    |
| DistilRoberta-base (3, 4, 6 layers) | 0.7395  | 0.9655   | Focal Loss (3, 4, 5) distilroberta-base UnderSampling                     |
| Deberta-base (last 3 layers)        | 0.7184  | 0.9596   | Focal Loss deberta-base (2 epoch, last 3 layers): Vanila Model            |
| Baseline DistilRoberta              | 0.6386  | 0.9422   | Focal Loss distilroberta: Vanila Prepare Data + Vanila distilroberta-base |

`Deberta-base (6, 7, 8 layers)` - доступна для скачивания. [Link](https://www.comet.ml/api/rest/v2/registry-model/item/download?workspaceName=danildmitriev1999&modelName=deberta-jb&version=1.0.0)
Ее результаты:

classification_report

              precision    recall  f1-score   support

           0       0.99      0.98      0.99      5945
           1       0.79      0.81      0.80       448

    accuracy                           0.97      6393
    macro avg       0.89      0.90      0.89      6393
    weighted avg       0.97      0.97      0.97      6393

confusion_matrix

[[5851 94]

[ 84 364]]

