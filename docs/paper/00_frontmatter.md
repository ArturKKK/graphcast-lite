УДК 551.509.313:004.032.26

# Мультимасштабная графовая нейросетевая модель регионального прогноза погоды с усвоением данных наблюдений

**А.С. Табаков**^1,2^, **А.В. Пененко**^1^

^1^Институт вычислительной математики и математической геофизики СО РАН, г. Новосибирск, Россия;
^2^Новосибирский государственный университет, г. Новосибирск, Россия

{{ЗАПОЛНИТЬ: адрес электронной почты автора-корреспондента}}

Представлена региональная нейросетевая модель прогноза погоды над югом Красноярского края, построенная на графовом представлении атмосферного состояния. Глобальная расчётная сетка с шагом 0.703° и региональная вставка с шагом 0.25° объединены в единый граф из 133 279 узлов, что устраняет необходимость задания внешних граничных условий: согласование разрешений на стыке достигается механизмом обмена сообщениями в процессе обучения. Модель содержит 5.9 млн обучаемых параметров, обучение и выпуск прогноза выполняются на одном графическом ускорителе. Оценка на независимой выборке из 1607 сроков (все сезоны) показала среднеквадратическую ошибку прогноза приземной температуры 1,84 °C на суточном сроке при успешности 69 % относительно инерционного прогноза. В контрольном эксперименте с равным бюджетом обучения установлено, что ошибка на проверочной выборке при одношаговом прогнозе является ненадёжным критерием отбора модели для многошагового прогноза. Исследовано усвоение наблюдений методами релаксации и оптимальной интерполяции внутри авторегрессионного цикла: непрерывное усвоение ограничивает рост ошибки при развёртке до 168 ч, тогда как «память» модели о единичной коррекции составляет порядка суток. Схема доведена до оперативного контура, работающего по глобальным анализам GDAS с обновлением каждые шесть часов.

**Ключевые слова:** численный прогноз погоды, графовая нейронная сеть, региональный прогноз, мультимасштабная сетка, усвоение данных, оптимальная интерполяция, авторегрессионный прогноз, машинное обучение

# A multiscale graph neural network for regional weather forecasting with data assimilation

**A.S. Tabakov**^1,2^, **A.V. Penenko**^1^

^1^Institute of Computational Mathematics and Mathematical Geophysics SB RAS, Novosibirsk, Russia;
^2^Novosibirsk State University, Novosibirsk, Russia

{{ЗАПОЛНИТЬ: e-mail}}

A regional neural network weather forecasting model for the south of Krasnoyarsk Krai based on a graph representation of the atmospheric state is presented. A global mesh with a 0.703° spacing and a regional insert with a 0.25° spacing are combined into a single graph of 133,279 nodes, which removes the need for external lateral boundary conditions: the resolutions are reconciled at the interface by the message passing mechanism during training. The model has 5.9 million trainable parameters; both training and forecast production run on a single GPU. Evaluation on an independent sample of 1607 initial times covering all seasons yields a root-mean-square error of the 2-m temperature forecast of 1.84 °C at the 24-h lead time with a skill score of 69 % relative to persistence. A controlled experiment with an equal training budget shows that the single-step validation error is an unreliable criterion for selecting a model intended for multi-step forecasting. Observation assimilation by relaxation (nudging) and optimal interpolation within the autoregressive cycle is examined: continuous assimilation limits error growth over a 168-h rollout, whereas the model's memory of a single correction lasts about one day. The scheme has been brought to an operational loop driven by global GDAS analyses updated every six hours.

**Keywords:** numerical weather prediction, graph neural network, regional forecasting, multiscale mesh, data assimilation, optimal interpolation, autoregressive forecast, machine learning

<!--
ЗАМЕЧАНИЯ ПО ОФОРМЛЕНИЮ (сверено с выпуском № 1 (395), 2025):

1. Аннотация даётся БЕЗ заголовка «Аннотация» — сразу абзацем после e-mail.
2. Разделы статьи НЕ нумеруются: «Введение», «Данные и методы», …, «Выводы».
3. Индексы аффилиаций — надстрочные цифры сразу после фамилии, без пробела.
4. DOI и строку «Поступила …» проставляет редакция; в рукописи их нет.
5. УДК: 551.509.313 — численные методы прогноза погоды; 004.032.26 — нейронные сети.
   Проверить у научного руководителя, принят ли в институте иной код.
6. Объём: до 20 стр. включая рисунки, таблицы и списки литературы (TNR 12, интервал 1.5).
7. Два списка: «Список литературы» (русские источники, затем латиница) и «References»
   (транслитерация русских + перевод названия в квадратных скобках + пометка [in Russ.]).
8. Аннотация ~200 слов — соответствует объёму аннотаций в выпуске.

ЧТО НУЖНО ОТ АВТОРОВ:
- адрес электронной почты автора-корреспондента (в журнале указывается один);
- согласование порядка авторов и точных названий аффилиаций;
- ORCID (журнал их не печатает в статье, но требует в заявке);
- заявка по форме журнала (zayvka.doc) и акт экспертизы о возможности открытого опубликования.
-->
