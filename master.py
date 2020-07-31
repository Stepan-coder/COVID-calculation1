import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# Так как x(t)=x(0)*e^{kt}, мы можем взять логарифмы, чтобы получить ln x(t)=ln x(0) + kt. Это означает, что
# для поиска ln x(0) и k вы можете найти наименьшие квадраты для данных {(t,ln x(t))}. Это скажет вам, что
# ln x(t) = b + at, так что k=a и x(0)=e^b

def read_from_csv(file):
    return pd.read_csv(file, encoding='utf-8', delimiter=",")


def write_to_csv(file, data, titles):
    df = pd.DataFrame(data, columns=titles)
    df.to_csv(file, index=False, sep=',', encoding='utf-8')


def increase(my_list):
    resalt = [0]
    for ml in range(1, len(my_list), 1):
        delta = my_list[ml] - my_list[ml - 1] if (my_list[ml] - my_list[ml - 1]) > 0 else 0
        resalt.append(delta)
    return resalt


def get_trend(my_list):
    a = 0
    b = 0
    alpha = 0
    if len(my_list) > 1:
        sumX = sum([r for r in range(1, len(my_list) + 1)])
        sumY = sum(my_list)
        sumX2 = sum([r * r for r in range(1, len(my_list) + 1)])
        sumXY = sum([r * my_list[r - 1] for r in range(1, len(my_list) + 1)])
        # a * sumX2 + b * sumX = sumXY
        # a * sumX + b * len(my_list) = sumY
        # b = (sumY - a * sumX)/len(my_list)
        # a * sumX2 + (sumY - a * sumX)/len(my_list) * sumX = sumXY
        # a * sumX2 *len(my_list) + (sumY - a * sumX) * sumX = sumXY * len(my_list)
        # a * sumX2 *len(my_list) + sumY * sumX - a * sumX * sumX = sumXY * len(my_list)
        # a * sumX2 *len(my_list) - a * sumX * sumX = sumXY * len(my_list) - sumY * sumX
        # a*(sumX2 *len(my_list) - sumX * sumX) = sumXY * len(my_list) - sumY * sumX
        a = (sumXY * len(my_list) - sumY * sumX) / (sumX2 * len(my_list) - sumX * sumX)
        b = (sumY - a * sumX) / len(my_list)
        alpha = math.degrees(math.atan(a))
    return a, b, alpha


def normalase(my_list):
    maximal = max(my_list)
    if maximal == 0:
        return [0 for i in my_list]
    else:
        return [i/maximal for i in my_list]


class Covid():
    def __init__(self, country, date, confirmed, deaths, recovered):
        self.country = country
        self.date = date
        self.confirmed = confirmed
        self.deaths = deaths
        self.recovered = recovered


# Подготавливаем названия всех регионов
def prepear_regions_names():
    regions_names = []
    for file in os.listdir(files_folder):
        new_data = list(pd.read_csv(files_folder + "\\" + file, encoding='utf-8', delimiter=",")['Country/Region'])
        regions_names += [nd.replace("*", "").replace(" ", "") for nd in new_data]
    return set(regions_names)


# Читаем все файлики
def prepear_info(regions_names):
    records = []
    for file in os.listdir(files_folder):
        date = file.split(".")[0].split("-")
        date[0], date[1] = date[1], date[0]
        day_confirmed = {}
        day_deaths = {}
        day_recovered = {}
        for rn in regions_names:
            day_confirmed[rn] = 0
            day_deaths[rn] = 0
            day_recovered[rn] = 0
        df = pd.read_csv(files_folder + "\\" + file, encoding='utf-8', delimiter=",")
        for d in range(len(df)):
            day_confirmed[df.at[d, 'Country/Region'].replace("*", "").replace(" ", "")] += df.at[d, 'Confirmed']
            day_deaths[df.at[d, 'Country/Region'].replace("*", "").replace(" ", "")] += df.at[d, 'Deaths']
            day_recovered[df.at[d, 'Country/Region'].replace("*", "").replace(" ", "")] += df.at[d, 'Recovered']
        for k in regions_names:
            records.append(Covid(k, date, day_confirmed[k], day_deaths[k], day_recovered[k]))
    return records


# сортируем записи по дням
def sort_records(records):
    sorted_records = {}
    for rl in records:
        if rl.date[2] in sorted_records:  # год
            if rl.date[1] in sorted_records[rl.date[2]]:  # месяц
                if rl.date[0] in sorted_records[rl.date[2]][rl.date[1]]:  # день
                    sorted_records[rl.date[2]][rl.date[1]][rl.date[0]].append(rl)
                else:
                    sorted_records[rl.date[2]][rl.date[1]][rl.date[0]] = [rl]
            else:
                sorted_records[rl.date[2]][rl.date[1]] = {rl.date[0]: [rl]}
        else:
            sorted_records[rl.date[2]] = {rl.date[1]: {rl.date[0]: [rl]}}
    return sorted_records


# Собираем все данные по категориям
def assembly_data(sorted_records):
    days = []
    day_confirmed = {}
    day_deaths = {}
    day_recovered = {}
    for sr_year in sorted(list(sorted_records.keys())):
        for sr_month in sorted(list(sorted_records[sr_year].keys())):
            for sr_day in sorted(list(sorted_records[sr_year][sr_month].keys())):
                days.append(str(sr_day) + "/" + str(sr_month) + "/" + str(sr_year))
                all_confirmed = 0
                all_deaths = 0
                all_recovered = 0
                for i in sorted_records[sr_year][sr_month][sr_day]:
                    confirmed = i.confirmed if not math.isnan(i.confirmed) else 0
                    deaths = i.deaths if not math.isnan(i.deaths) else 0
                    recovered = i.recovered if not math.isnan(i.recovered) else 0
                    all_confirmed += confirmed
                    all_deaths += deaths
                    all_recovered += recovered
                    if i.country not in day_confirmed:
                        day_confirmed[i.country] = [confirmed]
                        day_deaths[i.country] = [deaths]
                        day_recovered[i.country] = [recovered]
                    else:
                        day_confirmed[i.country].append(confirmed)
                        day_deaths[i.country].append(deaths)
                        day_recovered[i.country].append(recovered)

                if "all" not in day_confirmed:
                    day_confirmed["all"] = [all_confirmed]
                    day_deaths["all"] = [all_deaths]
                    day_recovered["all"] = [all_recovered]
                else:
                    day_confirmed["all"].append(all_confirmed)
                    day_deaths["all"].append(all_deaths)
                    day_recovered["all"].append(all_recovered)
    return days, day_confirmed, day_deaths, day_recovered


# создание индивидуальных графиков
def individual_schedule(days, day_confirmed, day_deaths, day_recovered):
    now_count = 0
    for c in day_confirmed:
        c = str(c).replace("*", "").replace(" ", "")

        a_day_confirmed = get_trend(normalase(day_confirmed[c][-7:]))[0]
        b_day_confirmed = get_trend(normalase(day_confirmed[c][-7:]))[1]
        x_t_day_confirmed = round(math.exp(b_day_confirmed)*math.exp(a_day_confirmed), 4)

        a_day_deaths = get_trend(normalase(day_deaths[c][-7:]))[0]
        b_day_deaths = get_trend(normalase(day_deaths[c][-7:]))[1]
        x_t_day_deaths = round(math.exp(b_day_deaths)*math.exp(a_day_deaths), 4)

        a_day_recovered = get_trend(normalase(day_recovered[c][-7:]))[0]
        b_day_recovered = get_trend(normalase(day_recovered[c][-7:]))[1]
        x_t_day_recovered = round(math.exp(b_day_recovered)*math.exp(a_day_recovered), 4)

        increase_day_confirmed = increase(day_confirmed[c])
        increase_day_deaths = increase(day_deaths[c])
        increase_day_recovered = increase(day_recovered[c])

        str_day_confirmed = 'Confirmed, ' + str(day_confirmed[c][-1])
        str_day_deaths = 'Deaths, ' + str(day_deaths[c][-1])
        str_day_recovered = 'Recovered, ' + str(day_recovered[c][-1])

        str_increase_day_confirmed = 'Increase Confirmed, ' + str(increase_day_confirmed[-1])
        str_increase_day_deaths = 'Increase Deaths, ' + str(increase_day_deaths[-1])
        str_increase_day_recovered = 'Increase Recovered, ' + str(increase_day_recovered[-1])

        title = "COVID - 19 : " + str(c) + " ( c=" + str(x_t_day_confirmed) + ", d=" + \
                str(x_t_day_deaths) + ", r=" + str(x_t_day_recovered) + " )"
        print(str(now_count) + "/" + str(len(regions_names) + 1), title)
        fig = plt.figure(figsize=(54, 18))
        plt.subplot(211)
        plt.title(title, fontsize=26)
        plt.plot(days[:len(day_confirmed[c])], day_confirmed[c], linestyle="-", color='red')
        plt.plot(days[:len(day_deaths[c])], day_deaths[c], linestyle="-", color='black')
        plt.plot(days[:len(day_recovered[c])], day_recovered[c], linestyle="-", color='green')
        plt.xticks(days)
        plt.xticks(rotation=40)
        plt.ylabel('Count')
        plt.grid(True)
        plt.legend([str_day_confirmed, str_day_deaths, str_day_recovered], loc='upper left')
        plt.subplot(212)
        plt.plot(days[:len(increase_day_confirmed)], increase_day_confirmed, linestyle="--", color='red')
        plt.plot(days[:len(increase_day_deaths)], increase_day_deaths, linestyle="--", color='black')
        plt.plot(days[:len(increase_day_recovered)], increase_day_recovered, linestyle="--", color='green')
        plt.xticks(days)
        plt.xticks(rotation=40)
        plt.ylabel('Increase Count')
        plt.xlabel('Days')
        plt.grid(True)
        plt.legend([str_increase_day_confirmed, str_increase_day_deaths, str_increase_day_recovered], loc='upper left')
        # plt.show()
        fig.savefig("countries" + "\\" + c +".png", dpi=fig.dpi)
        plt.close()
        now_count += 1


# составляем общий список, так сказать некий топ
def top_confirmed(days, day_confirmed, day_deaths, day_recovered):
    top = []
    for rn in regions_names:
        top.append([rn, day_confirmed[rn][-1], day_deaths[rn][-1], day_recovered[rn][-1]])
    sorted_conf = sorted(top, key=lambda x: x[1], reverse=True)[:15]
    fig = plt.figure(figsize=(36, 18))
    c_legend = []
    for st in sorted_conf:
        plt.plot(days[:len(day_confirmed[st[0]])], day_confirmed[st[0]], linestyle="-")
        c_legend.append(str(st[0]) + "," + str(st[1]))
    plt.xticks(days)
    plt.xticks(rotation=40)
    plt.ylabel('Count')
    plt.grid(True)
    plt.legend(c_legend, loc='upper left')
    # plt.show()
    fig.savefig("Top Confirmed.png", dpi=fig.dpi)
    plt.close()


def top_deaths(days, day_confirmed, day_deaths, day_recovered):
    top = []
    for rn in regions_names:
        top.append([rn, day_confirmed[rn][-1], day_deaths[rn][-1], day_recovered[rn][-1]])
    sorted_death = sorted(top, key=lambda x: x[2], reverse=True)[:15]
    fig = plt.figure(figsize=(36, 18))
    d_legend = []
    for st in sorted_death:
        plt.plot(days[:len(day_deaths[st[0]])], day_deaths[st[0]], linestyle="-")
        d_legend.append(str(st[0]) + "," + str(st[2]))
    plt.xticks(days)
    plt.xticks(rotation=40)
    plt.ylabel('Count')
    plt.grid(True)
    plt.legend(d_legend, loc='upper left')
    # plt.show()
    fig.savefig("Top Deaths.png", dpi=fig.dpi)
    plt.close()


def top_recovered(days, day_confirmed, day_deaths, day_recovered):
    top = []
    for rn in regions_names:
        top.append([rn, day_confirmed[rn][-1], day_deaths[rn][-1], day_recovered[rn][-1]])
    sorted_rec = sorted(top, key=lambda x: x[3], reverse=True)[:15]
    fig = plt.figure(figsize=(36, 18))
    r_legend = []
    for st in sorted_rec:
        plt.plot(days[:len(day_recovered[st[0]])], day_recovered[st[0]], linestyle="-")
        r_legend.append(str(st[0]) + "," + str(st[3]))
    plt.xticks(days)
    plt.xticks(rotation=40)
    plt.ylabel('Count')
    plt.grid(True)
    plt.legend(r_legend, loc='upper left')
    # plt.show()
    fig.savefig("Top Recovered.png", dpi=fig.dpi)
    plt.close()


files_folder = "daily_data"
titles = ['Country/Region', 'Confirmed', 'Deaths', 'Recovered']  # Убираем дату, т.к. она есть в заголовоке файла

regions_names = prepear_regions_names()
records = prepear_info(regions_names)
sorted_records = sort_records(records)
days, day_confirmed, day_deaths, day_recovered = assembly_data(sorted_records)
individual_schedule(days, day_confirmed, day_deaths, day_recovered)
top_confirmed(days, day_confirmed, day_deaths, day_recovered)
top_deaths(days, day_confirmed, day_deaths, day_recovered)
top_recovered(days, day_confirmed, day_deaths, day_recovered)
print(day_recovered["all"][-1]/day_confirmed["all"][-1], day_deaths["all"][-1]/day_confirmed["all"][-1])
