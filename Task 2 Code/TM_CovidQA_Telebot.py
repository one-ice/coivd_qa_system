# !/usr/bin/env python
# -*- coding: utf-8 -*-
# This program is dedicated to the public domain under the CC0 license.
"""
Simple Bot to reply to Telegram messages.First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""
import logging
import random
import os
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

import numpy as np
import pandas as pd

import requests
import json
from requests.packages.urllib3.exceptions import InsecureRequestWarning

from transformers import AutoTokenizer,pipeline
from transformers import AutoModelForQuestionAnswering

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.INFO)
logger = logging.getLogger(__name__)

model_dir = "/home/cx/Downloads/covidAsk-master/deep_learning_model"
                    
# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def covidAsk(query):
    params = {'query': query, 'strat': 'dense_first'}
    res = requests.get('https://covidask.korea.ac.kr/api', params=params, verify=False)
    outs = json.loads(res.text)
    return outs
    
def get_response(model_name, query, context):
    # Load model & tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #model.cuda()
    # build model
    #nlp = pipeline('question-answering', model=model, tokenizer=tokenizer, device=0)
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
 
    # get predictions
    inputs = {
            'question': query,
            'context':context
        }
      
    predict = nlp(inputs)['answer']
    return predict    

def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi ask me anything about COVID-19!')    
 
def help(update, context):
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')

def echo(update, context):
    """Echo the user message."""
    user_input = str(update.message.text)
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

    query = user_input
    results = covidAsk(query)
    context = results['ret'][0]['context']  
    
    return_ans = get_response(model_dir, query, context)
    
    
    # INSERT ANY FUNCTION HERE

    update.message.reply_text(return_ans)
    
    
   
def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)
    
def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater("5119346310:AAG5m33bX6jJ892i3l2ZVtHbyR9QzMl-xLc", use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, echo))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot

    # local code
    updater.start_polling() 

    # deployed code
    # updater.start_webhook(listen="0.0.0.0",
    #                       port=int(PORT),
    #                       url_path=SECRET_TOKEN,
    #                       webhook_url='https://ezfinbot.herokuapp.com/' + SECRET_TOKEN)


    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
