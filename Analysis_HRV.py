# Загрузка функций для анализа


import warnings

warnings.filterwarnings('ignore')


from tokens import tok
token = tok()

import json
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from hrvanalysis import get_frequency_domain_features
from hrvanalysis import get_poincare_plot_features
from hrvanalysis import get_sampen
from scipy.stats import kurtosis
from scipy.stats import skew
import io
import matplotlib.pyplot as plt
import seaborn as sns
import telebot
from telebot import types
import numpy as np
import pandas as pd
import pickle
from threading import Thread, Lock
from time import sleep
import time
import os
import datetime

folder = 'data'
path_json = 'data/json'
path_test = 'data/test'
path_text = 'data/text'

data = {}

listen_file = False

if not os.path.isdir(folder):
    os.mkdir(folder)
if not os.path.isdir(path_json):
    os.mkdir(path_json)
if not os.path.isdir(path_test):
    os.mkdir(path_test)
if not os.path.isdir(path_text):
    os.mkdir(path_text)


json_file = 'data.json'
path_json += ('/' + json_file)

def load_data():

    try:
        with open(path_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

    except:
        print('Ошибка базы данных!')
        #data = {"pictures": {}}
        with open(path_json, "w", encoding='utf-8') as f:
            f.write(json.dumps(data, indent=4))
    finally:
        f.close()

    return data

data = load_data()

def save_data(data):

    with open(path_json, "w", encoding='utf-8') as f:
        f.write(json.dumps(data, indent=4))

    f.close()


def get_figure_table(table):

    plt.figure()
    plt.xlabel("Дата")
    plt.ylabel("Время, мин")
    plt.figure(figsize=(10, 10))
    
    #sns.set(rc={'figure.figsize': (10, 10)})

    sns.set(context='paper', font_scale = 1, style='dark', font='sans-serif')


    
    sns.barplot(x=table[0], y=table[2],color='gray')
    sns.barplot(x=table[0], y=table[1],color='red')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return buf

def load_lk(id):

    global data
    
    data = load_data()

    print(data)
    try:
        print('Загружен профиль пользователя: ' + str(id))
        print(data[str(id)])

        dates = data[str(id)].keys()
        stressed = []
        alled = []


        now = datetime.datetime.now(tz=timezone)  
        nowf = now.strftime("%d.%m.%Y")

        datess = []

        for i in range(7):
            delta = datetime.timedelta(days=i)
            dd = now - delta
            datess.append(dd.strftime("%d.%m.%Y"))
        print(datess)

        for k in reversed(datess):
            try:
                stressed.append(data[str(id)][k]['stress'])
            except:
                stressed.append(0)
            try:
                alled.append(data[str(id)][k]['all'])
            except:
                alled.append(0)

            #stressed.append(data[str(id)][k]['stress'])
            #alled.append(data[str(id)][k]['all'])
        datess.reverse()
        print(datess)
        print(stressed)
        print(alled)

        stre = sum(stressed)
        alle = sum(alled)
        text = '👉 Вот данные о стрессе, за последнюю неделю:\n\n'
        if (stre / alle > 0.5):
            text += '❗️ Ваш уровень стресса более 50%. Возможно на этой неделе было, что-то важное. Тем не менее при продолжении тендеции, убедительно просим Вас обратиться к специалисту!\n\n'
        text += 'Доля стресса: ' + str(round((stre / alle)*100,2)) + ' %'
        text += '\nВсего измерений: ' + str(alle) + ' минут'
        bot.send_photo(id, photo=get_figure_table([datess, stressed, alled]), caption=text, reply_markup=keyboard_main)

    except:
        print('Профиль не найден!')
        texts = 'Данные не найдены!'
        bot.send_message(id, text=texts)


file = "data/adaboostpickled.tmp"
with open(file, "rb") as infile:
    pickled_model = pickle.load(infile)

offset = datetime.timedelta(hours=3)
timezone = datetime.timezone(offset, name='MSK')

bot = telebot.TeleBot(token);

welcome_file = path_text + '/welcome_text.txt'

keyboard_main = types.InlineKeyboardMarkup();
key_lk= types.InlineKeyboardButton(text='Личный кабинет', callback_data='lk');
keyboard_main.add(key_lk);
key_demo = types.InlineKeyboardButton(text='Демонстрационный режим', callback_data='yes');
keyboard_main.add(key_demo);
key_data= types.InlineKeyboardButton(text='Работа с данными', callback_data='data');
keyboard_main.add(key_data);
key_no= types.InlineKeyboardButton(text='Другое', callback_data='no');
keyboard_main.add(key_no);


keyboard_demo_list = types.InlineKeyboardMarkup();
key_var1= types.InlineKeyboardButton(text='1.1.txt', callback_data='var1');
keyboard_demo_list.add(key_var1);
key_var2 = types.InlineKeyboardButton(text='1.2.txt', callback_data='var2');
keyboard_demo_list.add(key_var2);
key_var3= types.InlineKeyboardButton(text='2.1.txt', callback_data='var3');
keyboard_demo_list.add(key_var3);
key_var4= types.InlineKeyboardButton(text='Другое', callback_data='var4');
keyboard_demo_list.add(key_var4);


keyboard_data = types.InlineKeyboardMarkup();
key_load = types.InlineKeyboardButton(text='Загрузить данные из файла', callback_data='load');
keyboard_data.add(key_load);
key_realtime = types.InlineKeyboardButton(text='Отслеживание в реальном времени', callback_data='realtime');
keyboard_data.add(key_realtime);

keyboard_files = types.InlineKeyboardMarkup();
key_txt = types.InlineKeyboardButton(text='.TXT', callback_data='txt');
keyboard_files.add(key_txt);
key_csv = types.InlineKeyboardButton(text='.CSV', callback_data='csv');
keyboard_files.add(key_csv);

one_minute_intervals = []


from typing import List, Tuple


one_minute_intervals = []


def get_time_domain_features(nn_intervals: List[float], pnni_as_percent: bool = True) -> dict:
    diff_nni = np.diff(nn_intervals)
    length_int = len(nn_intervals) - 1 if pnni_as_percent else len(nn_intervals)

    # Basic statistics
    mean_nni = np.mean(nn_intervals)
    median_nni = np.median(nn_intervals)
    range_nni = max(nn_intervals) - min(nn_intervals)

    sdsd = np.std(diff_nni)
    rmssd = np.sqrt(np.mean(diff_nni ** 2))

    nni_50 = sum(np.abs(diff_nni) > 50)
    pnni_50 = 100 * nni_50 / length_int
    nni_25 = sum(np.abs(diff_nni) > 20)
    pnni_25 = 100 * nni_25 / length_int

    # Feature found on github and not in documentation
    cvsd = rmssd / mean_nni

    # Features only for long term recordings
    sdnn = np.std(nn_intervals, ddof=1)  # ddof = 1 : unbiased estimator => divide std by n-1
    cvnni = sdnn / mean_nni

    # Heart Rate equivalent features
    heart_rate_list = np.divide(60000, nn_intervals)
    mean_hr = np.mean(heart_rate_list)
    min_hr = min(heart_rate_list)
    max_hr = max(heart_rate_list)
    std_hr = np.std(heart_rate_list)

    time_domain_features = {
        'mean_nni': mean_nni,
        'sdnn': sdnn,
        'sdsd': sdsd,
        'nni_50': nni_50,
        'pnni_50': pnni_50,
        'nni_25': nni_25,
        'pnni_25': pnni_25,
        'rmssd': rmssd,
        'median_nni': median_nni,
        'range_nni': range_nni,
        'cvsd': cvsd,
        'cvnni': cvnni,
        'mean_hr': mean_hr,
        "max_hr": max_hr,
        "min_hr": min_hr,
        "std_hr": std_hr,
    }

    return time_domain_features





def load_from_file(file):
    global one_minute_intervals
    global interval
    if file == '':
        pa=path_test+'/test.txt'
        with open(pa, encoding="utf-8") as f:
            test = path_test +'/' + f.readline()
        f.close()
    else:
        test = file

    data_list = list(np.loadtxt(test))
    
    # Define the time interval in milliseconds (1 minute = 60000 milliseconds)
    interval = 60000
    # Initialize variables
    one_minute_intervals = []
    current_interval = []
    current_sum = 0

    # Iterate through the data
    for value in data_list:

        if current_sum+value < interval:
            current_sum += value
            current_interval.append(value)

        else:
            one_minute_intervals.append(current_interval)
            current_interval = []
            #print(current_sum)
            current_sum = 0
            
    return(test)


def add_to_profile(id, file):
    global data
    global one_minute_intervals
    global interval
    data = load_data()

    try:
        print(data[str(id)])
    except:
        data[str(id)] = {}
    now = datetime.datetime.now(tz=timezone)  
    nowf = now.strftime("%d.%m.%Y")

    load_from_file(file)

    max = 0
    min = 999
    RRmax = []
    RRmin = []
    RRStress = []
    summer = 0

    stress_vector = []



    for count, interval in enumerate(one_minute_intervals):
        
        print(len(interval))
        
        RR = remove_outliers(rr_intervals=interval, low_rri=300, high_rri=2000)

        status_stress = check_stress(RR)

        if len(interval) > max:
            max = len(interval)
            RRmax = RR
        if len(interval) < min:
            min = len(interval)
            RRmin = RR

        if status_stress == 1:
            RRStress = RR


        
        stress_vector.append(int(status_stress))
        print(status_stress)
        summer += status_stress
        
    text = 'Данные загружены.\nМаксимальный пульс: ' + str(max) + '\nМинимальный пульс: ' + str(min) + '\nВремя в стрессе: ' + str(summer) + '/' + str(len(one_minute_intervals)+1) + ' минут'
    bot.send_photo(id, photo=get_figure([RRStress, RRmin, RRmax]), caption=text, reply_markup=keyboard_main)

    print(data)
    try:
        print(data[str(id)][str(nowf)])
    except:
        data[str(id)][str(nowf)] = {}

    try:
        data[str(id)][str(nowf)]['stress'] += int(summer)
    except:
        data[str(id)][str(nowf)]['stress'] = int(summer)

    try:
        data[str(id)][str(nowf)]['all'] += int(len(one_minute_intervals)+1)
    except:
        data[str(id)][str(nowf)]['all'] = int(len(one_minute_intervals)+1)

    try:
        data[str(id)][str(nowf)]['vect'] += stress_vector
    except:
        data[str(id)][str(nowf)]['vect'] = stress_vector

    
    print(data)
    save_data(data)

def normalization_in_minute(vector):
    new_vector = []
    new_vector.append(vector[0]/1000)
    i = 1
    for element in vector[1:]:
        new_vector.append(new_vector[i-1]+(element/1000))
        i += 1
    return new_vector


def check_stress(RR):
    rr = []
    for i in range(1, len(RR)):
        rr.append(2*(RR[i]-RR[i-1])/(RR[i]+RR[i-1]))
    time_1 = get_time_domain_features(RR)
    time_rr = get_time_domain_features(rr)
    freq_1 = get_frequency_domain_features(RR)
    pointcare_1 = get_poincare_plot_features(RR)
    sampen=get_sampen(RR)
    df = pd.DataFrame([(time_1['mean_nni'], time_1['median_nni'], time_1['sdnn'], time_1['rmssd'], time_1['sdsd'], time_1['sdnn']/time_1['rmssd'], time_1['mean_hr'], time_1['pnni_25'], time_1['pnni_50'], pointcare_1['sd1'], pointcare_1['sd2'], kurtosis(RR), skew(RR),
                        time_rr['mean_nni'], time_rr['median_nni'], time_rr['sdnn'], time_rr['rmssd'], time_rr['sdsd'], time_rr['sdnn']/time_rr['rmssd'], kurtosis(rr), skew(rr),
                        freq_1['vlf'], freq_1['vlf']/freq_1['total_power']*100, freq_1['lf'], freq_1['lf']/freq_1['total_power']*100, freq_1['lfnu'], freq_1['hf'], freq_1['hf']/freq_1['total_power']*100, freq_1['hfnu'], freq_1['total_power'], freq_1['lf_hf_ratio'], 1/freq_1['lf_hf_ratio'],
                        sampen['sampen'])],
                        columns=['MEAN_RR', 'MEDIAN_RR', 'SDRR', 'RMSSD', 'SDSD', 'SDRR_RMSSD', 'HR', 'pNN25', 'pNN50', 'SD1', 'SD2', 'KURT', 'SKEW',
                                    'MEAN_REL_RR', 'MEDIAN_REL_RR', 'SDRR_REL_RR', 'RMSSD_REL_RR', 'SDSD_REL_RR', 'SDRR_RMSSD_REL_RR', 'KURT_REL_RR', 'SKEW_REL_RR',
                                    'VLF', 'VLF_PCT', 'LF', 'LF_PCT', 'LF_NU', 'HF', 'HF_PCT', 'HF_NU', 'TP', 'LF_HF', 'HF_LF',
                                    'sampen'])
    df_end = df.loc[:, ['KURT', 'VLF', 'MEAN_REL_RR', 'SDSD_REL_RR', 'KURT_REL_RR', 'TP', 'MEDIAN_REL_RR', 'pNN25', 'LF', 'SD2', 'RMSSD_REL_RR', 'SDRR']]

    return pickled_model.predict(df_end)[0]

def get_figure(table):

    plt.figure()
    plt.xlabel("Время, с")
    plt.ylabel("Длительность импульса, мс")
    plt.figure(figsize=(10, 10))
    
    #sns.set(rc={'figure.figsize': (10, 10)})

    sns.set(context='paper', font_scale = 2, style='dark', font='sans-serif')
    RRStress = table[0]
    for RR in table[1:]:
        sns.lineplot(x=normalization_in_minute(RR), y=RR,color='gray')
    if (RRStress != []):
        sns.lineplot(x=normalization_in_minute(RRStress), y=RRStress,color='red')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return buf



def test(id):

    global one_minute_intervals
    global interval
    #print(one_minute_intervals)
    max = 0
    min = 999
    RRmax = []
    RRmin = []
    RRStress = []
    summer = 0
    for count, interval in enumerate(one_minute_intervals):
        
        print(len(interval))
        
        RR = remove_outliers(rr_intervals=interval, low_rri=300, high_rri=2000)

        if len(interval) > max:
            max = len(interval)
            RRmax = RR
        if len(interval) < min:
            min = len(interval)
            RRmin = RR

        
        status_stress = check_stress(RR)

        print(status_stress)
        if status_stress == 1:
            #print('В течении %d минут объект находится в стрессе' % (count+1))
            RRStress = RR
        else:

            ...
            #print('В течении %d минут объект не находится в стрессе' % (count+1))
            #for i in interval:
             # print(round(i))
        #print('\n')

        summer += status_stress

        #print(summer)
        #pickled_model.predict(df_end)
        #sleep(1)
    



    text = 'Тест завершен.\nМаксимальный пульс: ' + str(max) + '\nМинимальный пульс: ' + str(min) + '\nВремя в стрессе: ' + str(summer) + '/' + str(len(one_minute_intervals)+1) + ' минут'
    bot.send_photo(id, photo=get_figure([RRStress, RRmin, RRmax]), caption=text, reply_markup=keyboard_main)


@bot.message_handler(content_types=['document'])
def handle_docs_photo(message):
    global listen_file
    if listen_file == True:
        try:
            chat_id = message.chat.id

            file_info = bot.get_file(message.document.file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            if not os.path.isdir('data/downloads/'+ str(chat_id) + '/'):
                os.mkdir('data/downloads/'+ str(chat_id) + '/')
            src = 'data/downloads/' + str(chat_id) + '/' + message.document.file_name;
            with open(src, 'wb') as new_file:
                new_file.write(downloaded_file)

            bot.reply_to(message, "Файл успешно загружен!")

            add_to_profile(chat_id, src)

            listen_file = False
        except Exception as e:
            bot.reply_to(message, e)


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if (message.text == "/help") or (message.text == "/start"):
        with open(welcome_file, encoding="utf-8") as f:
            texts = f.read()
        f.close()
        
        bot.send_message(message.from_user.id, text=texts, reply_markup=keyboard_main)

@bot.callback_query_handler(func=lambda call: True)
def callback_worker(call):

    global listen_file

    if call.data == "yes":
        texts = 'Выберите тестовые данные: '
        bot.send_message(call.message.chat.id, text=texts, reply_markup=keyboard_demo_list)
    elif call.data == "var1":
        tst = load_from_file(path_test+'/1.1.txt')
        bot.send_message(call.message.chat.id, 'Запускаю демонстрационный режим... (' + tst + ')');
        t1 = Thread(target=test(call.message.chat.id))
        t1.start()
        t1.join()
    elif call.data == "var2":
        tst = load_from_file(path_test+'/1.2.txt')
        bot.send_message(call.message.chat.id, 'Запускаю демонстрационный режим... (' + tst + ')');
        t1 = Thread(target=test(call.message.chat.id))
        t1.start()
        t1.join()
    elif call.data == "var3":
        tst = load_from_file(path_test+'/2.1.txt')
        bot.send_message(call.message.chat.id, 'Запускаю демонстрационный режим... (' + tst + ')');
        t1 = Thread(target=test(call.message.chat.id))
        t1.start()
        t1.join()
    elif call.data == "var4":
        tst = load_from_file('')
        bot.send_message(call.message.chat.id, 'Запускаю демонстрационный режим... (' + tst + ')');
        t1 = Thread(target=test(call.message.chat.id))
        t1.start()
        t1.join()
    elif call.data == "lk":

        load_lk(call.message.chat.id)

    elif call.data == "data":
        texts = 'Выберите способ передачи данных:'
        bot.send_message(call.message.chat.id, text=texts, reply_markup=keyboard_data)

    elif call.data == "load":
        texts = 'Выберите тип файла:'
        bot.send_message(call.message.chat.id, text=texts, reply_markup=keyboard_files)
        
    elif call.data == "txt":
        texts = 'Ожидаю файл.'
        bot.send_message(call.message.chat.id, text=texts)
        listen_file = True

    elif (call.data == "no") or (call.data == "realtime") or (call.data == "csv"):
        texts = 'Функция временно недоступна.'
        bot.send_message(call.message.chat.id, text=texts)


bot.polling(none_stop=True, interval=0)