__author__ = 'thor'
# -*- coding: utf-8 -*-
import requests

currency_map = {
    '$': "USD",
    '₤': "GBP",
    '£': "GBP",
    '€': "EUR",
    '₤': "GBP",
    '£': "GBP",
    '€': "EUR",
    'dollar': "USD",
    'dollars': "USD",
    'euro': "EUR",
    'euros': "EUR",
    'pound': "GBP",
    'pounds': "GBP"
}


def exchange_rate_from_to(from_currency, to_currency):
    return requests.get(
        'http://rate-exchange.appspot.com/currency?from={from_currency}&to={to_currency}&q=1'
        .format(from_currency=currency_code(from_currency), to_currency=currency_code(to_currency))).json()['rate']


def euros_from(from_currency):
    return requests.get(
        'http://rate-exchange.appspot.com/currency?from={from_currency}&to=EUR&q=1'
        .format(from_currency=currency_code(from_currency))).json()['rate']


def euros_to(from_currency):
    return requests.get(
        'http://rate-exchange.appspot.com/currency?from=EUR&to={to_currency}&q=1'
        .format(to_currency=currency_code(from_currency))).json()['rate']


def to_euros_rates(from_currencies=['USD', 'GBP']):
    return {currency: euros_from(currency) for currency in from_currencies}


def currency_code(currency_spec):
    try:
        return currency_map[currency_spec]
    except Exception:
        return currency_spec


