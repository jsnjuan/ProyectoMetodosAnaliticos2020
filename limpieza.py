# -*- coding: utf-8 -*-

import pandas as pd
import re
import unicodedata
import pickle

rt_proy = r'~/Desktop/MCD/MA/ProyectoFinal'

rt = (r'~/Desktop/MCD/MA/Taskmaster-master/TM-1-2019')

rt_tdt = rt + r'/train-dev-test'

dev = pd.read_csv(rt_tdt + r"/dev.csv",header=None)
test = pd.read_csv(rt_tdt + r"/test.csv",header=None)
train = pd.read_csv(rt_tdt + r"/train.csv",header=None)

self_dialogs = pd.read_json(rt + r"/self-dialogs.json")

# Hagamos un conteo para conocer los diálogos que tenemos para cada tipo de 
# orden

self_dialogs['instruction_id'].value_counts()

# Out[12]: 
# pizza-ordering-2      1211
# auto-repair-appt-1    1161
# coffee-ordering-1      735
# restaurant-table-1     704
# uber-lyft-1            646
# movie-tickets-1        642
# coffee-ordering-2      641
# restaurant-table-2     494
# uber-lyft-2            452
# movie-tickets-2        377
# pizza-ordering-1       257
# movie-tickets-3        195
# restaurant-table-3     102
# movie-finder            54
# movie-ticket-1          37
# Name: instruction_id, dtype: int64

# Elegimos únicamente una instrucción, en este caso, pizza-ordering-2
self_dialogs = self_dialogs[self_dialogs['instruction_id']=='pizza-ordering-2']

# vemos el tamaño de cada conversación
self_dialogs['len_utterance'] = self_dialogs['utterances'].map(len)

self_dialogs['len_utterance'].describe()

# Nos quedamos únicamente con la conversación del texto, quitando las anotaciones
# y todo lo demás

def clean_uterrance(utterances):
    return [ut['text'].lower() for ut in utterances]

self_dialogs['utterances'] = self_dialogs['utterances'].map(clean_uterrance)

self_dialogs.set_index('conversation_id', inplace=True)

#========================  INICIA: LIMPIEZA MANUAL ============================
# derivado de una revisión específca a nivel de caracteres y palabras, es 
# necesario hacer algunos cambios manualmente, como los que se indican a 
# continuación
conversation_id = 'dlg-d4cadb74-be2c-49b3-9fb3-08e952e36ebb' 
#conversation_id = self_dialogs.index == 'dlg-d4cadb74-be2c-49b3-9fb3-08e952e36ebb' 

self_dialogs.at[conversation_id, 'utterances'] = [self_dialogs.at[conversation_id, 'utterances'][i] for i in range(20)]

conversation_id = 'dlg-b12ff9a7-adfa-4cdd-8cd7-a1e84526c59a'
self_dialogs.at[conversation_id, 'utterances'] = [x.replace('`', '') 
                                                   for x in self_dialogs.loc[conversation_id, 'utterances']]

conversation_id = 'dlg-fa97878f-320d-4008-af75-31b160e397f2'
self_dialogs.at[conversation_id, 'utterances'] = [x.replace('`', '') 
                                                   for x in self_dialogs.loc[conversation_id, 'utterances']]


conversation_id = 'dlg-103ba46c-964d-4934-b10f-015d259eb863'
self_dialogs.at[conversation_id, 'utterances'] = [x.replace('”', '') 
                                                   for x in self_dialogs.loc[conversation_id, 'utterances']]

conversation_id = 'dlg-aa325470-8eed-4a87-8be6-8bf3a2d1b55c'
self_dialogs.at[conversation_id, 'utterances'] = [x.replace('”', '"') 
                                                   for x in self_dialogs.loc[conversation_id, 'utterances']]

conversation_id = 'dlg-f8377ab3-f272-4803-b16b-2666bf032199'
self_dialogs.at[conversation_id, 'utterances'] = [x.replace('”', '"') 
                                                   for x in self_dialogs.loc[conversation_id, 'utterances']]

# Algunas palabras mal escritas

# Palabras mal escritas
    #favrote, favorite id: dlg-798bace5-cd19-45b4-8a85-21cb21d284b3
    #favortie, favorite id: dlg-798bace5-cd19-45b4-8a85-21cb21d284b3
    #sald, salad id: dlg-798bace5-cd19-45b4-8a85-21cb21d284b3
    
conversation_id = 'dlg-798bace5-cd19-45b4-8a85-21cb21d284b3'
self_dialogs.at[conversation_id, 'utterances'] = [x.replace('favrote', 'favorite') 
                                                   for x in self_dialogs.loc[conversation_id, 'utterances']]


conversation_id = 'dlg-798bace5-cd19-45b4-8a85-21cb21d284b3'
self_dialogs.at[conversation_id, 'utterances'] = [x.replace('favortie', 'favorite') 
                                                   for x in self_dialogs.loc[conversation_id, 'utterances']]

self_dialogs.at[conversation_id, 'utterances'] = [x.replace('ald', 'salad') 
                                                   for x in self_dialogs.loc[conversation_id, 'utterances']]

#========================  TERMINA: LIMPIEZA MANUAL ===========================

# Generaremos una relación de oraciones y ids, a fin de identificar conversaiones
# con oraciones de las que hay que revisar los caracteres a fondo.

ls_sentences  = [(json_id, utt)  for json_id, dialog in zip(self_dialogs.index, 
                           self_dialogs['utterances']) for utt in dialog]

# Creamos todos los pares de textos, la variable independiente y la respuesta
# que daremos al modelo, recordemos que se condiciona a todo el texto previo, 
# lo que haremos es dada una conversación [a, b, c, d, e] vamos a generar los
# siguientes pares: (a, b), (a+b, c), (a+b+c, d), (a+b+c+d, e)


ls_tuplas = []

for ls in self_dialogs['utterances']:

    utt_prev = None 

    for utt in ls:
        if utt_prev:
            ls_tuplas.append( (utt_prev, utt) )
            utt_prev = utt_prev + ' ' + utt
        else:
            utt_prev = utt

# Veamos el total de pares que ajustará nuestro modelo para pedir pizza
print(len(ls_tuplas))

# print(len(ls_tuplas))
# 25643

# Limpiaremos cada cadena de texto. 


#========================  INICIA: LIMPIEZA AUTOMATICA ========================
# Converts the unicode file to ascii
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

def preprocess_str(w):  
    # aquí esta corección es para los tamaños de las pizzas, esto considera
    # todos los tamaños posibles
    w = re.sub('58"', ' 58 inches ', w)
    w = re.sub('18"', ' 18 inches ', w)
    w = re.sub('20"', ' 20 inches ', w)
    w = re.sub('12"', ' 12 inches ', w)
    w = re.sub('30"', ' 30 inches ', w)
    w = re.sub('14"', ' 14 inches ', w)
    w = re.sub('10"', ' 10 inches ', w)
    w = re.sub('26"', ' 26 inches ', w)
    w = re.sub('1"', '1 inch ', w)
    w = re.sub('22"', ' 22 inches ', w)
    w = re.sub('16"', ' 16 inches ', w)
    w = re.sub('8"', '8 inches', w)
    
    # Corregir primero los siguientes typos identificados:
    w = re.sub('that;s', "that's", w)
    w = re.sub('i;d', "i'd", w)
    w = re.sub('you;d', "you'd", w)
    w = re.sub('a;;', "all", w)
    w = re.sub('thankyou', "thank you", w)
    w = re.sub('whatcha', "whatch out", w)
    # reeemplazamos 'ñ' por 'n'
    w = re.sub('ñ', 'n', w)
    w = re.sub('’', "'", w)
    w = re.sub(' & ', " and ", w)
    # El patrón de porcentaje de descuento o de propina
    w = re.sub(r'\d{1, 2}%', "\1 percent", w)
     
    # corregimos los siguientes errores:
    w = re.sub(r"reciept", "receipt", w)
    w = re.sub(r"xtreme", "extreme", w)
    w = re.sub(r"spinich", "spinach", w)
    w = re.sub(r"pinapple", "pineapple", w)
    w = re.sub(r"hawaian", "hawaiian", w)
    w = re.sub(r"hawiann", "hawaiian", w)
    w = re.sub(r"mozzerella", "mozzarella", w)
    w = re.sub(r"parmesean", "parmesan", w)
    w = re.sub(r"topings", "toppings", w)
    w = re.sub(r"tomotoes", "tomatoes", w)
    w = re.sub(r"mintues", "minutes", w)
    w = re.sub(r"mn", "minutes", w)
    w = re.sub(r"specality", "speciality", w)
    w = re.sub(r"margetarita", "margarita", w)
    w = re.sub(r"minuets", "minutes", w)
    w = re.sub(r"toppin", "topping", w)
    w = re.sub(r"mushoom", "mushroom", w)
    w = re.sub(r"instea", "instead", w)
    w = re.sub(r"parmasan", "parmesan", w)
    w = re.sub(r"paridise", "paradise", w)
    w = re.sub(r"pizzzas", "pizzas", w)
    w = re.sub(r"margeritte", "margarita", w)
    w = re.sub(r"deliever", "deliver", w)
    w = re.sub(r"pinapples", "pineapples", w)
    w = re.sub(r"baccon", "bacon", w)
    w = re.sub(r"meduium", "medium", w)
    w = re.sub(r"pleasw", "please", w)
    w = re.sub(r"thanx", "thanks", w)
    w = re.sub(r"esle", "else", w)
    w = re.sub(r"ogringal", "original", w)
    w = re.sub(r"toping", "topping", w)
    w = re.sub(r"margharita", "margarita", w)
    w = re.sub(r"restuarant", "restaurant", w)
    w = re.sub(r"coversation", "conversation", w)
    w = re.sub(r"whaaat", "what", w)
    w = re.sub(r"carmalized", "caramelized", w)
    w = re.sub(r"gratiuty", "gratuity", w)
    w = re.sub(r"recieve", "receive", w)
    w = re.sub(r"definately", "definitely", w)
    w = re.sub(r"anyting", "anything", w)
    w = re.sub(r"ssalad", "salad", w)
    w = re.sub(r"garlice", "garlic", w)
    w = re.sub(r"pepperoini", "pepperoni", w)
    w = re.sub(r"pepperonin", "pepperoni", w)
    w = re.sub(r"pepperonin", "pepperoni", w)
    w = re.sub(r"jalopeno", "jalapeno", w)    
        
    w = unicode_to_ascii(w.strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
      
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^A-Za-z0-9?!.¿,\/'\"+=:;]", " ", w)

    # Clean the text
    w = re.sub(r"what's", "what is ", w)
    w = re.sub(r"\'s", " ", w)
    w = re.sub(r"\'s", " ", w)
    w = re.sub(r"\'ve", " have ", w)
    w = re.sub(r"can't", "cannot ", w)
    w = re.sub(r"n't", " not ", w)
    w = re.sub(r"i'm", "i am ", w)
    w = re.sub(r"\'re", " are ", w)
    w = re.sub(r"\'d", " would ", w)
    w = re.sub(r"\'ll", " will ", w)
    w = re.sub(r",", " ", w)
    w = re.sub(r"\.", " ", w)
    w = re.sub(r"!", " ! ", w)
    w = re.sub(r"\/", " ", w)
    w = re.sub(r"\^", " ^ ", w)
    w = re.sub(r"\+", " + ", w)
    w = re.sub(r"\-", " - ", w)
    w = re.sub(r"\=", " = ", w)
    w = re.sub(r"'", " ", w)
    w = re.sub(r":", " : ", w)
    w = re.sub(r" e g ", " eg ", w)
    w = re.sub(r" b g ", " bg ", w)
    w = re.sub(r"\0s", "0", w)
    w = re.sub(r" 9 11 ", "911", w)
    w = re.sub(r" fod", "food", w)
    w = w.strip()
    w = ' '.join(x for x in w.split(' ') if x!='')
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

ls_tuplas_limpio = [(preprocess_str(a), preprocess_str(b)) for a, b in ls_tuplas]

#========================  TERMINA: LIMPIEZA AUTOMATICA =======================


# Conteo de palabras
# Antes, haremos un conteo de caracteres, para ver si existen algunos a los que
# debamos poner especial cuidado, 
dc_words = {}

for a, b in ls_tuplas_limpio:
    for l in a.split(' '):
        dc_words[l] = dc_words.get(l, 0) + 1
    for l in b.split(' '):
        dc_words[l] = dc_words.get(l, 0) + 1

# Pondremos el diccionaro generado en un dataframe para hacer un conteo
df_words = pd.DataFrame.from_dict(dc_words, orient='index')

df_words.sort_values(by=0, inplace = True, ascending=False)

# De una revisión a df_words es posible encontrar palabras equivocadas.
# Cuando la frecuencia comienza a ser menor, digamos de 100 para abajo,
# empiezan a aparecer palabras un poco más especializadas, o bien,typos
# por ejemplo, 
# reciept -> receipt 
# xtreme -> extreme
# spinich -> spinach
# pinapple -> pineapple
# hawaian -> hawaiian
# hawiann -> hawaiian
# mozzerella -> mozzarella
# parmesean -> parmesan
# topings -> toppings
# tomotoes -> tomatoes
# mintues -> minutes
# mn -> minutes
# specality -> speciality
# margetarita -> margarita
# minuets -> minutes
# toppin -> topping
# mushoom -> mushroom
# instea -> instead
# parmasan -> parmesan
# paridise -> paradise
# pizzzas -> pizzas
# margeritte -> margarita
# deliever -> deliver
# pinapples -> pineapples
# baccon -> bacon
# meduium -> medium
# pleasw -> please
# thanx -> thanks
# esle -> else
# ogringal -> original
# toping -> topping
# margharita -> margarita
# restuarant -> restaurant
# coversation -> conversation
# whaaat -> what
# carmalized -> caramelized
# gratiuty -> gratuity
# recieve -> receive
# definately -> definitely
# anyting -> anything
# ssalad -> salad
# garlice -> garlic
# pepperoini -> pepperoni
# pepperonin -> pepperoni
# jalopeno -> jalapeno

print(df_words.index)

# Hasta aquí ya tenemos nuestro conjunto de oraciones de entrada y de salida
# lo guardaremos en un archivo de formato pickle para trabajar con este
# conjunto en google colab

with open(rt_proy + r'\ls_tuplas.pkl', 'wb') as f:
    pickle.dump(ls_tuplas_limpio, f)


