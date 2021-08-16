import pandas as pd
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings


warnings.filterwarnings("ignore")


demo = pd.read_spss('demo2017 (1).SAV')
#в разных выборках названия тех же столбцов отличаются регистрами
#сделаем так, чтобы они совпадали
demo = demo.rename(columns={'N_UHC' : 'n_uhc', 'N_UPC' : 'n_upc', 'AGE' : 'age', 'SEX' : 'sex'})
demo.to_csv('demo.csv', sep='\t', index=False)

health = pd.read_spss('health2017 (1).sav')
health.to_csv('health.csv', sep='\t', index=False)


income = pd.read_spss('income2017 (1).SAV')
income.to_csv('income.csv', sep='\t', index=False)

#объединим сначала первые две так, чтобы не было одинаковых столбцов
buf = demo.merge(health, how='outer', left_on=['n_uhc', 'n_upc'], right_on=['n_uhc', 'n_upc'], suffixes=('', '_y'))
buf = buf.drop(buf.filter(regex='_y$').columns.tolist(), axis=1)
#и так же следим за различиями в регистре
buf = buf.rename(columns={'YEAR' : 'year'})
buf.to_csv('buf.csv', sep='\t', index=False)
income = income.rename(columns={'N_UHC' : 'n_uhc', 'RESID' :'resid'})
#объединяем с третьей ту, что получилась ранее
df = buf.merge(income, how='outer', left_on=['n_uhc'], right_on=['n_uhc'], suffixes=('', '_y'))
df = df.drop(df.filter(regex='_y$').columns.tolist(), axis=1)
df.to_csv('df.csv', sep='\t', index=False)

#формируем подвыборку
dataset = df.sample(n=2000, random_state=20001227)
dataset.reset_index(drop=True, inplace=True)
#в соответствии с файлом описания переменных меняем тип на дискретный там, где это надо
dataset = dataset.astype({'n_uhc' : 'Int64', 'n_upc' : 'Int64', 'year' : 'Int64', 'age' : 'Int64', 'nummonth' : 'Int64',\
    'weight' : 'Int64', 'height' : 'Int64', 'HSIZE' : 'Int64', 'ch0_5' : 'Int64', 'ch6_12' : 'Int64', 'ch13_17' : 'Int64',\
    'elder' : 'Int64' , 'HH_BLINT' : 'Int64', 'HH_INT1' : 'Int64', 'HH_INT2' : 'Int64', 'HH_INT3' : 'Int64', 'HH_INT4' : 'Int64'})
dataset.dtypes.to_csv('dtypes.txt', sep='\t', index=False)
dataset.to_csv('dataset.csv', sep='\t', index=False)

#округляем веса
dataset['Yweight'] = np.floor(dataset['Yweight'])
dataset = dataset.astype({'Yweight' : 'int64'})


print("Описательная статистика")
print()
print("Все доходы")
print("mean", dataset['totalinc'].mean())
print("median", dataset['totalinc'].median())
print("mode", dataset['totalinc'].mode().values)
print("std", dataset['totalinc'].std())
print("skewness", dataset['totalinc'].skew())
print("kurtosis", dataset['totalinc'].kurtosis())
dataset['totalinc'].plot(kind="hist", title='Total income')
print()
plt.show()

print("Все расходы")
print("mean", dataset['totalexp'].mean())
print("median", dataset['totalexp'].median())
print("mode", dataset['totalexp'].mode().values)
print("std", dataset['totalexp'].std())
print("skewness", dataset['totalexp'].skew())
print("kurtosis", dataset['totalexp'].kurtosis())
dataset['totalexp'].plot(kind="hist", title='Total expense')
print()
plt.show()


print("Зарплата")
print("mean", dataset['inc_1'].mean())
print("median", dataset['inc_1'].median())
print("mode", dataset['inc_1'].mode().values)
print("std", dataset['inc_1'].std())
print("skewness", dataset['inc_1'].skew())
print("kurtosis", dataset['inc_1'].kurtosis())
dataset['inc_1'].plot(kind="hist", title='Wages')
print()
plt.show()


print("Поступления от продажи личного и домашнего имущества")
print("mean", dataset['inc_9'].mean())
print("median", dataset['inc_9'].median())
print("mode", dataset['inc_9'].mode().values)
print("std", dataset['inc_9'].std())
print("skewness", dataset['inc_9'].skew())
print("kurtosis", dataset['inc_9'].kurtosis())
dataset['inc_6'].plot(kind="hist", title='inc_9')
print()
plt.show()


print("Расходы на здравоохранение")
print("mean", dataset['exp_9'].mean())
print("median", dataset['exp_9'].median())
print("mode", dataset['exp_9'].mode().values)
print("std", dataset['exp_9'].std())
print("skewness", dataset['exp_9'].skew())
print("kurtosis", dataset['exp_9'].kurtosis())
dataset['exp_9'].plot(kind="hist", title='exp_9')
print()
plt.show()


print("Вес")
print("mean", dataset['weight'].mean())
print("median", dataset['weight'].median())
print("mode", dataset['weight'].mode().values)
print("std", dataset['weight'].std())
print("skewness", dataset['weight'].skew())
print("kurtosis", dataset['weight'].kurtosis())
dataset['weight'].plot(kind="hist", title='Weight')
print()
plt.show()


print("Рост")
print("mean", dataset['height'].mean())
print("median", dataset['height'].median())
print("mode", dataset['height'].mode().values)
print("std", dataset['height'].std())
print("skewness", dataset['height'].skew())
print("kurtosis", dataset['height'].kurtosis())
dataset['height'].plot(kind="hist", title='Height')
print()
plt.show()


print("Возраст")
print("mean", dataset['age'].mean())
print("median", dataset['age'].median())
print("mode", dataset['age'].mode().values)
print("std", dataset['age'].std())
print("skewness", dataset['age'].skew())
print("kurtosis", dataset['age'].kurtosis())
dataset['age'].plot(kind="hist", title='Height')
print()
plt.show()

mass_index = dataset['weight'] / (dataset['height']**2)
saved_money = dataset['totalinc'] - dataset['totalexp']

print("Индекс массы тела")
print("mean", mass_index.mean())
print("median", mass_index.median())
print("mode", mass_index.mode().values)
print("std", mass_index.std())
print("skewness", mass_index.skew())
print("kurtosis", mass_index.kurtosis())
mass_index.plot(kind="hist", title='Mass index')
print()
plt.show()

print("Сбережения")
print("mean", saved_money.mean())
print("median", saved_money.median())
print("mode", saved_money.mode().values)
print("std", saved_money.std())
print("skewness", saved_money.skew())
print("kurtosis", saved_money.kurtosis())
saved_money.plot(kind="hist", title='Saved money')
print()
plt.show()

#столбчатые диаграммы для дискретных переменных
sns.histplot(x=dataset['sex'])
plt.show()
fig, ax = plt.subplots(figsize=(30, 4))
sns.histplot(x=dataset['region'], ax=ax)
plt.show()
fig, ax = plt.subplots(figsize=(30, 4))
sns.histplot(x=dataset['HTYPE'].dropna(), ax = ax, hue=dataset['HTYPE'].dropna())
plt.show()
sns.histplot(x=dataset['HSIZE'].dropna())
plt.show()
fig, ax = plt.subplots(figsize=(30, 4))
sns.histplot(x=dataset['Educat'], ax = ax, hue=dataset['Educat'])
plt.show()
sns.histplot(x=dataset['healthev'].dropna())
plt.show()

#до этого мы не создавали взвешенную выборку с дублированными по весам наблюдениями
#слишком большой расход времени и памяти
#проделать это с одной переменной, а не всем датафреймом - куда приятнее
weighted_inc_1 = pd.Series([])
for i in range(len(dataset['Yweight'])):
    weighted_inc_1 = weighted_inc_1.append(pd.Series(np.repeat(dataset['inc_1'][i], dataset['Yweight'][i])))


print("Взвешенная зарплата")
print("mean", weighted_inc_1.mean())
print("median", weighted_inc_1.median())
print("mode", weighted_inc_1.mode().values)
print("std", weighted_inc_1.std())
print("skewness", weighted_inc_1.skew())
print("kurtosis", weighted_inc_1.kurtosis())
weighted_inc_1.plot(kind="hist", title='Weighted wages')
print()
plt.show()

#то же самое для сбережений
weighted_saved_money = pd.Series([])
for i in range(len(dataset['Yweight'])):
    weighted_saved_money = weighted_saved_money.append(pd.Series(np.repeat(saved_money[i], dataset['Yweight'][i])))


print("Взвешенные сбережения")
print("mean", weighted_saved_money.mean())
print("median", weighted_saved_money.median())
print("mode", weighted_saved_money.mode().values)
print("std", weighted_saved_money.std())
print("skewness", weighted_saved_money.skew())
print("kurtosis", weighted_saved_money.kurtosis())
weighted_saved_money.plot(kind="hist")
print()
print()
plt.show()


#аналогичный анализ логарифмированных переменных
#однако не все переменные можно логарифмировать
log_totalinc = pd.Series(np.log(dataset['totalinc']))
log_totalexp = pd.Series(np.log(dataset['totalexp']))
log_weight = pd.Series(np.log(dataset['weight']))
log_height = pd.Series(np.log(dataset['height']))

print("Статистика логарифмированных величин")
print()
print("Все доходы")
print("mean", log_totalinc.mean())
print("median", log_totalinc.median())
print("mode", log_totalinc.mode().values)
print("std", log_totalinc.std())
print("skewness", log_totalinc.skew())
print("kurtosis",log_totalinc.kurtosis())
log_totalinc.plot(kind="hist", title='Total income')
print()
plt.show()

print("Все расходы")
print("mean", log_totalexp.mean())
print("median", log_totalexp.median())
print("mode", log_totalexp.mode().values)
print("std", log_totalexp.std())
print("skewness", log_totalexp.skew())
print("kurtosis", log_totalexp.kurtosis())
log_totalexp.plot(kind="hist", title='Total expense')
print()
plt.show()



print("Вес")
print("mean", log_weight.mean())
print("median", log_weight.median())
print("mode", log_weight.mode().values)
print("std", log_weight.std())
print("skewness", log_weight.skew())
print("kurtosis", log_weight.kurtosis())
log_weight.plot(kind="hist", title='Weight')
print()
plt.show()


print("Рост")
print("mean", log_height.mean())
print("median", log_height.median())
print("mode", log_height.mode().values)
print("std", log_height.std())
print("skewness", log_height.skew())
print("kurtosis", log_height.kurtosis())
log_height.plot(kind="hist", title='Height')
print()
plt.show()



#аналогичный анализ стандартизированных величин
sd_totalinc = (dataset['totalinc'] - dataset['totalinc'].mean())/dataset['totalinc'].std()
sd_totalexp = (dataset['totalexp'] - dataset['totalexp'].mean())/dataset['totalexp'].std()
sd_inc_1 = (dataset['inc_1'] - dataset['inc_1'].mean())/dataset['inc_1'].std()
sd_inc_9 = (dataset['inc_9'] - dataset['inc_9'].mean())/dataset['inc_9'].std()
sd_exp_9 = (dataset['exp_9'] - dataset['exp_9'].mean())/dataset['exp_9'].std()
sd_weight = (dataset['weight'] - dataset['weight'].mean())/dataset['weight'].std()
sd_height = (dataset['height'] - dataset['height'].mean())/dataset['height'].std()
sd_age = (dataset['age'] - dataset['age'].mean())/dataset['age'].std()

sd_totalinc.plot(kind="hist", title='Std Total income')
plt.show()
sd_totalexp.plot(kind="hist", title='Std Total expense')
plt.show()
sd_inc_1.plot(kind="hist", title='Std Wages')
plt.show()
sd_inc_9.plot(kind="hist", title='Std inc_9')
plt.show()
sd_exp_9.plot(kind="hist", title='Std exp_9')
plt.show()
sd_weight.plot(kind="hist", title='Std Weight')
plt.show()
sd_height.plot(kind="hist", title='Std Height')
plt.show()
sd_age.plot(kind="hist", title='Std Age')
plt.show()

# тест показывает, что номинальная зарплата не совпадает с реальной за 2017 год
#это ни разу не удивительно
print("Тест Стьюдента для зарплаты")
print(ss.ttest_1samp(dataset['inc_1'], 815.25, nan_policy='omit'))
print(ss.ttest_1samp(weighted_inc_1, 815.25, nan_policy='omit'))
print()


males = dataset.loc[dataset['sex'] == 'Male']
females = dataset.loc[dataset['sex'] == 'Female']

#исходя из теста, средние зарплаты мужчин и женщин различаются существенно
#использовано две версии теста на случай неравенства дисперсий
print("Тест Стьюдента для зарплаты по признаку пола")
print(ss.ttest_ind(males['inc_1'], females['inc_1'], equal_var=True, nan_policy='omit'))
print(ss.ttest_ind(males['inc_1'], females['inc_1'], equal_var=False, nan_policy='omit'))
print()
#тест говорит, что дисперсии равны
print("Тест Ливиня для проверки равенства дисперсий")
print(ss.levene(males['inc_1'].dropna(), females['inc_1'].dropna()))
print()

grodno = dataset.loc[dataset['region'] == 'Grodno oblast']
mogilev = dataset.loc[dataset['region'] == 'Mogilev oblast']

#для inc_9 отличия имеются, а для exp_9 нет
print("Тесты Стьюдента и Ливиня для Гродно и Могилева")
print()
print("По переменной inc_9")
print(ss.ttest_ind(grodno['inc_9'], mogilev['inc_9'], equal_var=True, nan_policy='omit'))
print(ss.ttest_ind(grodno['inc_9'], mogilev['inc_9'], equal_var=False, nan_policy='omit'))
print(ss.levene(grodno['inc_9'].dropna(), mogilev['inc_9'].dropna()))
print()
print("По переменной exp_9")
print(ss.ttest_ind(grodno['exp_9'], mogilev['exp_9'], equal_var=True, nan_policy='omit'))
print(ss.ttest_ind(grodno['exp_9'], mogilev['exp_9'], equal_var=False, nan_policy='omit'))
print(ss.levene(grodno['exp_9'].dropna(), mogilev['exp_9'].dropna()))
print()

#создаем категориальные переменные на основе количественных
#для возраста
age = pd.Series([])
for i in range(len(dataset['age'])):
    if dataset['age'][i] >= 18 and dataset['age'][i] < 25:
        age = age.append(pd.Series(['18-24']), ignore_index=True)
    elif dataset['age'][i] >= 25 and dataset['age'][i] < 35:
        age = age.append(pd.Series(['25-34']), ignore_index=True)
    elif dataset['age'][i] >= 35 and dataset['age'][i] < 45:
        age = age.append(pd.Series(['35-44']), ignore_index=True)
    elif dataset['age'][i] >= 45 and dataset['age'][i] < 55:
        age = age.append(pd.Series(['45-54']), ignore_index=True)
    elif dataset['age'][i] >= 55 and dataset['age'][i] < 65:
        age = age.append(pd.Series(['55-64']), ignore_index=True)
    else:
        age = age.append(pd.Series(np.nan))

#и зарплаты
wages = pd.Series([])
for i in range(len(dataset['inc_1'])):
    if dataset['inc_1'][i] > 0 and dataset['inc_1'][i] < 400:
        wages = wages.append(pd.Series(['0-400']), ignore_index=True)
    elif dataset['inc_1'][i] >= 400 and dataset['inc_1'][i] < 500:
        wages = wages.append(pd.Series(['400-500']), ignore_index=True)
    elif dataset['inc_1'][i] >= 500 and dataset['inc_1'][i] < 700:
        wages = wages.append(pd.Series(['500-700']), ignore_index=True)
    elif dataset['inc_1'][i] >= 700 and dataset['inc_1'][i] < 1000:
        wages = wages.append(pd.Series(['700-1000']), ignore_index=True)
    elif dataset['inc_1'][i] >= 1000:
        wages = wages.append(pd.Series(['>1000']), ignore_index=True)
    elif abs(dataset['inc_1'][i]) < 0.00000001 or np.isnan(dataset['inc_1'][i]):
        wages = wages.append(pd.Series(np.nan))

#добавляем созданные переменные в качестве столбцов
dataset['cat_wages'] = wages.copy()
dataset['cat_age'] = age.copy()

#уникальный номер для каждого набюдения понадобится в следующем задании
dataset['id'] = range(1, 2001)

#иллюстрируем таблицы сопряженности с помощью тепловых карт
data = dataset.groupby(['sex', 'cat_wages'], as_index=False)['id'].count()
data = data.rename(columns={'id' : 'count'})

#эта штука еще понадобится дальше
wages_frequencies = data['count'].copy()

pivot = data.pivot_table(values='count',index='sex',columns='cat_wages', aggfunc=lambda x : x)
sns.heatmap(data=pivot, annot=True, fmt='d')
plt.show()

data = dataset.groupby(['cat_age', 'cat_wages'], as_index=False)['id'].count()
data = data.rename(columns={'id' : 'count'})
pivot = data.pivot_table(values='count',index='cat_age',columns='cat_wages', aggfunc=lambda x : x)
sns.heatmap(data=pivot, annot=True)
plt.show()

data = dataset.groupby(['Educat', 'cat_wages'], as_index=False)['id'].count()
data = data.rename(columns={'id' : 'count'})
pivot = data.pivot_table(values='count',index='Educat',columns='cat_wages', aggfunc=lambda x : x)
fig, ax = plt.subplots(figsize=(20, 4))
sns.heatmap(data=pivot, annot=True, fmt='d', ax=ax)
plt.show()

data = dataset.groupby(['Educat', 'sport'], as_index=False)['id'].count()
data = data.rename(columns={'id' : 'count'})
pivot = data.pivot_table(values='count',index='Educat',columns='sport', aggfunc=lambda x : x)
fig, ax = plt.subplots(figsize=(20, 7))
sns.heatmap(data=pivot, annot=True, fmt='f', ax=ax)
plt.show()

data = dataset.groupby(['sex', 'smoker'], as_index=False)['id'].count()
data = data.rename(columns={'id' : 'count'})
pivot = data.pivot_table(values='count',index='sex',columns='smoker', aggfunc=lambda x : x)
sns.heatmap(data=pivot, annot=True, fmt='d')
plt.show()

#для хи-квадрат теста нужны частоты (количества наблюдений в каждом интервале)
male_wages_frequencies = pd.Series([])
female_wages_frequencies = pd.Series([])
for i in range(len(wages_frequencies)):
    if i < len(wages_frequencies)/2:
        female_wages_frequencies = female_wages_frequencies.append(pd.Series([wages_frequencies[i]]), ignore_index=True)
    else:
        male_wages_frequencies = male_wages_frequencies.append(pd.Series([wages_frequencies[i]]), ignore_index=True)

#таки различия есть
print("Хи-квадрат тест для зарплат у мужчин и женщин")
print(ss.chisquare(male_wages_frequencies, female_wages_frequencies))
print()

dat = dataset.groupby(['sex', 'Educat'], as_index=False)['id'].count()
dat = dat.rename(columns={'id' : 'count'})
education_frequencies = dat['count'].copy()


male_education_frequencies = pd.Series([])
female_education_frequencies = pd.Series([])
for i in range(len(education_frequencies)):
    if i < len(education_frequencies)/2:
        female_education_frequencies = female_education_frequencies.append(pd.Series([education_frequencies[i]]), ignore_index=True)
    else:
        male_education_frequencies = male_education_frequencies.append(pd.Series([education_frequencies[i]]), ignore_index=True)

#тут тоже
print("Хи-квадрат тест для образования у мужчин и женщин")
print(ss.chisquare(male_education_frequencies, female_education_frequencies))
print()

data.drop([0, 3], inplace=True)
table = np.array([[data['count'][1], data['count'][2]], [data['count'][4], data['count'][5]]])

#и кто же чаще курит...
print("Хи-квадрат тест для проверки взаимной зависимости пола и привычки к курению")
print(ss.chi2_contingency(table))
print()



good = dataset.loc[dataset['healthev'] == 'Good']['inc_1'].dropna()
bad = dataset.loc[dataset['healthev'] == 'Bad']['inc_1'].dropna()
so_so = dataset.loc[dataset['healthev'] == 'Not very good, but not bad']['inc_1'].dropna()

#на случай неравенства дисперсий ниже, кроме теста Фишера, также выполняется тест Краскелла-Уоллиса

#зависимость довольно сильная
print("Однофакторный дисперсионный анализ для зарплаты и оценки здоровья")
print(ss.levene(good, bad, so_so))
print(ss.f_oneway(good, bad, so_so))
print(ss.kruskal(good, bad, so_so))
print()


minsk = dataset.loc[dataset['region'] == 'Minsk city']['inc_1'].dropna()
minsk_obl = dataset.loc[dataset['region'] == 'Minsk oblast']['inc_1'].dropna()
grodno_obl = dataset.loc[dataset['region'] == 'Grodno oblast']['inc_1'].dropna()
brest_obl = dataset.loc[dataset['region'] == 'Brest oblast']['inc_1'].dropna()
gomel_obl = dataset.loc[dataset['region'] == 'Gomel oblast']['inc_1'].dropna()
vitebsk_obl = dataset.loc[dataset['region'] == 'Vitebsk oblast']['inc_1'].dropna()
mogilev_obl = dataset.loc[dataset['region'] == 'Mogilev oblast']['inc_1'].dropna()

#зависимость тоже есть
print("Однофакторный дисперсионный анализ для зарплаты и региона")
print("С Минском")
print(ss.levene(minsk, minsk_obl, grodno_obl, brest_obl, gomel_obl, vitebsk_obl, mogilev_obl))
print(ss.f_oneway(minsk, minsk_obl, grodno_obl, brest_obl, gomel_obl, vitebsk_obl, mogilev_obl))
print(ss.kruskal(minsk, minsk_obl, grodno_obl, brest_obl, gomel_obl, vitebsk_obl, mogilev_obl))
print()

#а тут нет
#вывод такой что вне Минска по стране платят одинаково плохо
print("И без него")
print(ss.levene(minsk_obl, grodno_obl, brest_obl, gomel_obl, vitebsk_obl, mogilev_obl))
print(ss.f_oneway(minsk_obl, grodno_obl, brest_obl, gomel_obl, vitebsk_obl, mogilev_obl))
print(ss.kruskal(minsk_obl, grodno_obl, brest_obl, gomel_obl, vitebsk_obl, mogilev_obl))
print()


#у Гродно и Могилева лучший p-value
features = ['cashinc', 'InKind', 'Privlg']
X = dataset[dataset['region'] == 'Grodno oblast'][features]
Y = dataset[dataset['region'] == 'Mogilev oblast'][features]

print("Тест Хоттелинга")
print(pg.multivariate_ttest(X, Y))
print()




features = ['inc_9', 'cashinc', 'InKind', 'Privlg', 'totalinc', 'exp_9', 'totalexp']
corr = dataset[features].corr()
print("Корреляционная матрица")
print(corr)
print()

X = dataset[dataset['region'] == 'Grodno oblast'][features].corr()
Y = dataset[dataset['region'] == 'Mogilev oblast'][features].corr()
print("С наблюдениями только из Гродно")
print()
print(X)
print()
print("С наблюдениями только из Могилева")
print(Y)
print()

#будем считать сильной корреляцию больше 0,9 по модулю
#лучше всего коррелируют общий доход, общий денежный доход и общий расход
#частный коэффициент корреляции показывает влияние двух переменных друг на друга без учета некоторой третьей
r_inc_exp = (corr['totalinc']['totalexp'] - corr['totalinc']['cashinc']*corr['totalexp']['cashinc'])/ \
    (((1 - corr['totalinc']['cashinc']**2)*(1 - corr['totalexp']['cashinc']**2))**0.5)

r_inc_cash = (corr['totalinc']['cashinc'] - corr['totalinc']['totalexp']*corr['totalexp']['cashinc'])/ \
    (((1 - corr['totalinc']['totalexp']**2)*(1 - corr['totalexp']['cashinc']**2))**0.5)

r_exp_cash = (corr['totalexp']['cashinc'] - corr['totalinc']['totalexp']*corr['totalinc']['cashinc'])/ \
    (((1 - corr['totalinc']['totalexp']**2)*(1 - corr['totalinc']['cashinc']**2))**0.5)

print("Частные коэффициенты корреляции")
print('totalinc and totalexp without cashinc', r_inc_exp)
print('totalinc and cashinc without totalexp', r_inc_cash)
print('totalexp and cashinc without totalinc', r_exp_cash)
print()



#///
dataset.dropna(inplace=True)
features = ['inc_9', 'InKind', 'Privlg']
X = dataset[features]
y = dataset['exp_9']


X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print("Регрессия для exp_9: общая информация")
print()
print(est2.summary())
print()
print()

features = ['inc_9', 'InKind', 'Privlg', 'exp_9']
X = dataset[features]
y = dataset['totalexp']

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print("Регрессия для totalexp: общая информация")
print()
print(est2.summary())
print()
print()


print("Многофакторный дисперсионный анализ")
print("С использованием различных способов вычисления суммы квадратов")
print(pg.anova(dv='inc_1', between=['cat_age', 'sex', 'Educat'], data=dataset, ss_type=1))
print(pg.anova(dv='inc_1', between=['cat_age', 'sex', 'Educat'], data=dataset, ss_type=2))
print(pg.anova(dv='inc_1', between=['cat_age', 'sex', 'Educat'], data=dataset, ss_type=3))


