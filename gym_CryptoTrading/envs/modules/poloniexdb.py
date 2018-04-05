from __future__ import division
import sqlite3
import time
import os
import numpy as np
from poloniex import Poloniex


def makeDB():
    ''' This function takes no arguments.

        This function creates a new database for the Poloniex Public API data in a preexisting subfolder
        named 'databases' and names it 'poloniex.db'.
        If a database already named like this exists in the database folder, it returns True.
        If the database is created successfully returns True.
    '''
    os.chdir('databases')
    if os.path.isfile('./poloniex.db'):
        os.chdir('..')
        print("Database already exists.")
        print("Continuing...")
        return True
    else:
        conn = sqlite3.connect('poloniex.db')
        c = conn.cursor()
        try:
            c.execute('''CREATE TABLE
                         IF NOT EXISTS currencyPairs(
                             currencyPair TEXT PRIMARY KEY,
                             active INTEGER NOT NULL,
                             lastUpdate REAL NOT NULL,
                             lastDate REAL NOT NULL
                             );
                         ''')

            c.execute('''CREATE TABLE 
                         IF NOT EXISTS data( 
                             currencyPair TEXT NOT NULL,                        
                             date REAL NOT NULL,
                             high REAL NOT NULL,
                             low REAL NOT NULL,
                             open REAL NOT NULL,
                             close REAL NOT NULL,
                             volume REAL NOT NULL,
                             quoteVolume REAL NOT NULL,
                             weightedAverage REAL NOT NULL,
                             FOREIGN KEY(currencyPair) REFERENCES currencyPairs(currencyPair),
                             CONSTRAINT unique_date UNIQUE (currencyPair, date)
                            );
                        ''')
            conn.commit()
            print("Database successfully created!")
            return True
        except:
            os.remove('poloniex.db')
            print("Error at poloniexdb.py: Problem creating the database.")
            raise
        finally:
            conn.close()
            os.chdir('..')


def addPairToDB(currencyPair):
    '''
    -----------------------
        ARGUMENTS

        - currencyPair: String representing a Poloniex currency pair, e.g. "BTC_ETH".

    -----------------------

    This function adds a new pair to the database, downloading and storing all data avaliable for the pair
    at the moment.
    If the database does not exists yet, it returns 0.
    If more than 150 calls are done to the API and no data is received, it returns 2.
    If an entry with the same pair of currencies already exists on the database or the pair
    is added correctly, it returns True.
    '''
    assert type(currencyPair) == str
    os.chdir('databases')
    if os.path.isfile('./poloniex.db'):
        conn = sqlite3.connect('poloniex.db')
        c = conn.cursor()
        try:
            c.execute("SELECT EXISTS(SELECT 1 FROM currencyPairs WHERE currencyPair= ? LIMIT 1)", [currencyPair])
            alreadyExists = c.fetchall()[0][0]
            if not alreadyExists:
                polo = Poloniex()
                lastUpdate = time.time()
                updated = False
                trials = 0
                while updated == False and trials < 150:
                    rawData = polo.returnChartData(currencyPair, 300, start = 1.0, end = lastUpdate)
                    if float(rawData[0]['close']) != 0.0:
                        lastDate = float(rawData[-1]['date'])
                        c.execute("INSERT INTO currencyPairs VALUES (?,?,?,?)",
                                  [currencyPair, 0, lastUpdate, lastDate])
                        for i in range(len(rawData)):
                            c.execute("INSERT INTO data VALUES (?,?,?,?,?,?,?,?,?)",
                                      [currencyPair,
                                       float(rawData[i]['date']),
                                       float(rawData[i]['high']),
                                       float(rawData[i]['low']),
                                       float(rawData[i]['open']),
                                       float(rawData[i]['close']),
                                       float(rawData[i]['volume']),
                                       float(rawData[i]['quoteVolume']),
                                       float(rawData[i]['weightedAverage'])])

                        conn.commit()
                        updated = True
                    else:
                        trials += 1
                        time.sleep(1)
                if updated == False:
                    print('Timeout while adding pair to database. Too many failed calls.')
                    print("Continuing...")
                    return 2
                else:
                    print("Pair sucessfully added to the database!")
                    return True
            else:
                print("Pair already in the database.")
                return True
        except:
            print("Error at poloniexdb.py: Unable to add pair to database.")
            raise
        finally:
            conn.close()
            os.chdir('..')
    else:
        os.chdir('..')
        print("The database you are trying to access does not exist.")
        return 0


def updatePair(currencyPair):
    '''
    -----------------------
        ARGUMENTS

        - currencyPair: String representing a Poloniex currency pair, e.g. "BTC_ETH".

    -----------------------

    This function updates the data of a pair already stored on the database.
    If the database does not exists yet, it returns 0.
    If more than 150 calls are done to the API and no data is received, it returns 2.
    If the pair is not in the database, it returns 3.
    If the database is already up to date or successfully updated, it returns True.
    '''
    assert type(currencyPair) == str
    os.chdir('databases')
    if os.path.isfile('./poloniex.db'):
        conn = sqlite3.connect('poloniex.db')
        c = conn.cursor()
        try:
            c.execute("SELECT EXISTS(SELECT 1 FROM currencyPairs WHERE currencyPair= ? LIMIT 1)", [currencyPair])
            pairExists = c.fetchall()[0][0]
            if pairExists:
                polo = Poloniex()
                lastUpdate = time.time()
                updated = False
                trials = 0
                c.execute("SELECT lastDate FROM currencyPairs WHERE currencyPair = ?", [currencyPair])
                oldLastDate = c.fetchall()[0][0]
                if lastUpdate - oldLastDate > 300:
                    while updated == False and trials < 150:
                        rawData = polo.returnChartData(currencyPair, 300, start=oldLastDate + 0.1, end=lastUpdate)
                        if float(rawData[0]['close']) != 0.0:
                            lastDate = float(rawData[-1]['date'])
                            c.execute('''UPDATE currencyPairs SET lastUpdate = ?, lastDate = ? 
                            WHERE currencyPair= ?''', [lastUpdate, lastDate, currencyPair])
                            for i in range(len(rawData)):
                                c.execute("INSERT INTO data VALUES (?,?,?,?,?,?,?,?,?)",
                                          [currencyPair,
                                           float(rawData[i]['date']),
                                           float(rawData[i]['high']),
                                           float(rawData[i]['low']),
                                           float(rawData[i]['open']),
                                           float(rawData[i]['close']),
                                           float(rawData[i]['volume']),
                                           float(rawData[i]['quoteVolume']),
                                           float(rawData[i]['weightedAverage'])])
                            conn.commit()
                            updated = True
                        else:
                            trials += 1
                            time.sleep(1)
                    if updated == False:
                        print('Timeout while adding pair to database. Too many failed calls.')
                        print("Continuing...")
                        return 2
                    else:
                        print("Pair successfully updated.")
                        return True
                else:
                    print('Pair already up to date.')
                    return True
            else:
                print("Pair is not in the database.")
                return 3
        except:
            print("Error at poloniexdb.py: Unable to update pair.")
            raise
        finally:
            conn.close()
            os.chdir('..')
    else:
        os.chdir('..')
        print("The database you are trying to access does not exist.")
        return 0


def getData(currencyPair, period, variables, start=0.0, end='last', steps=None):
    '''
    -----------------------
        ARGUMENTS

        - currencyPair: String representing a Poloniex currency pair, e.g. "BTC_ETH".
        - period: Integer representing the size of the candles returned. Must be a natural multiple of 300.
        - variables: List of strings representing variables from the poloniex API to be returned, e.g. ["open", "low"].
        - start: UNIX timestamp or None.
        - end: UNIX timestamp or None, default value being the most recently stored timestamp.
        - steps: UNIX int or None

    -----------------------

    This function returns the data of a pair already stored on the database given some parameters as a dictionary with
    arrays.
    If the database does not exists yet, it returns 0.
    '''
    assert type(currencyPair) == str
    assert type(period) == int and period % 300 == 0 and period > 0
    for var in variables:
        assert var in ['date', 'high', 'low', 'open', 'close', 'volume', 'quoteVolume', 'weightedAverage']
    assert type(start) == float or start == None
    assert end == 'last' or type(end) == float or end == None
    assert type(steps) == int or steps == None
    assert len([var for var in [start, end, steps] if var == None]) == 1
    os.chdir('databases')
    if os.path.isfile('./poloniex.db'):
        conn = sqlite3.connect('poloniex.db')
        c = conn.cursor()
        try:
            c.execute("SELECT lastDate FROM currencyPairs WHERE currencyPair = ?", [currencyPair])
            lastDate = c.fetchall()[0][0]
            if end == 'last':
                end = lastDate
            elif type(end) == float:
                assert end <= lastDate and end >= 0.0
            elif end == None:
                end = start + period * steps
            if type(start) == float:
                assert start <= end
            elif start == None:
                start = end - period * steps
            assert start >= 0.0
            c.execute('SELECT ' + ' ,'.join(variables) +
                      ' FROM data WHERE currencyPair = ? AND date > ? AND date <= ?', [currencyPair, start, end])
            rawData = c.fetchall()

            if 'weightedAverage' in variables:
                c.execute('SELECT volume FROM data WHERE currencyPair = ? AND date > ? AND date <= ?',
                          [currencyPair, start, end])
                volumes = c.fetchall()
                c.execute('SELECT open FROM data WHERE currencyPair = ? AND date > ? AND date <= ?',
                          [currencyPair, start, end])
                openings = c.fetchall()

            subSamples = period // 300
            samples = len(rawData) // subSamples
            Data = {}
            for v in range(len(variables)):
                var = variables[v]
                if var == 'date':
                    Data['date'] = np.array([rawData[i * subSamples][v] for i in range(samples)])
                elif var == 'high':
                    Data['high'] = np.zeros(samples)
                    for i in range(samples):
                        Data['high'][i] = max([rawData[i * subSamples + j][v] for j in range(subSamples)])
                elif var == 'low':
                    Data['low'] = np.zeros(samples)
                    for i in range(samples):
                        Data['low'][i] = min([rawData[i * subSamples + j][v] for j in range(subSamples)])
                elif var == 'open':
                    Data['open'] = np.array([rawData[i * subSamples][v] for i in range(samples)])
                elif var == 'close':
                    Data['close'] = np.array([rawData[i * subSamples][v] for i in range(samples)])
                elif var == 'volume':
                    Data['volume'] = np.zeros(samples)
                    for i in range(samples):
                        Data['volume'][i] = sum([rawData[i * subSamples + j][v] for j in range(subSamples)])
                elif var == 'quoteVolume':
                    Data['quoteVolume'] = np.zeros(samples)
                    for i in range(samples):
                        Data['quoteVolume'][i] = sum([rawData[i * subSamples + j][v] for j in range(subSamples)])
                elif var == 'weightedAverage':
                    Data['weightedAverage'] = np.zeros(samples)
                    for i in range(samples):
                        auxAvg = 0.0
                        sumVols = 0.0
                        for j in range(subSamples):
                            auxAvg = auxAvg + rawData[i * subSamples + j][v] * volumes[i * subSamples + j][0]
                            sumVols = sumVols + volumes[i * subSamples + j][0]
                        if sumVols == 0.0:
                            Data['weightedAverage'][i] = openings[i * subSamples][0]
                        else:
                            Data['weightedAverage'][i] = auxAvg / sumVols
            return Data
        except:
            print("Error at poloniexdb.py: Unable to get data.")
            raise
        finally:
            conn.close()
            os.chdir('..')
    else:
        os.chdir('..')
        print("The database you are trying to access does not exist.")
        return 0