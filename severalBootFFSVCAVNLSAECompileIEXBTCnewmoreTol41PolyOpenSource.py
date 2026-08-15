#/usr/bin/python2.7 -u 
# originally developed for py 2.7, auto converted to 3.11 then comment out unworking grid stuff in plots

# dev Feb 19 2014

# start to update for bootstrap trial Aug19 2014

# set up for MVAR app usage Jul 14 2015

# add some crypto to match mcarlorisk scripts jun 3 2018

# May 26 2025 update for polygon.io for MVAR
 
# Aug 2025 commentary and tuneups for stand alone
# needs Python3.11 not higher yet due to unresolved issues
# we have some comments here about plotting bugs with py3.11
# but it is more likely due to subtle changes in the plot library than python
# so that is just a shorthand notation

# several inputs from list

import numpy
import scipy.optimize
import os
import time
from datetime import datetime as dt
from datetime import date

import matplotlib.backends.backend_pdf

# example like javascript bind
from functools import partial
#def f(a,b,c):
#    print a,b,c
#bound_f = partial(f,1)
#bound_f(2,3)

from sklearn.metrics import r2_score

from subprocess import call
from functools import reduce

# does not work:  from matplotlib.backends.backend_pdf import PdfPages

# matplotlib config directory
os.environ['MPLCONFIGDIR'] = "."

# plotting imports
import matplotlib

##### TRY PdfPages instead:  
matplotlib.use('Agg') # to plot to file instead of screen

from pylab import *
#end plotting

from sklearn import linear_model
# no good on py3
#from sklearn import cross_validation
import sklearn

import json
import urllib.request, urllib.parse, urllib.error
import sys
import csv
import random
import copy
import math
import socket

import signal
import time
from threading import Thread

# leave False to dump out json output file as normal
# may run faster if this is True
skipio = False

tolIndexToTol = 0.005

def getScriptPath():
    print("sys.argv[0] = ", sys.argv[0])
    return os.path.dirname(__file__) # os.path.dirname(os.path.realpath(sys.argv[0]))

# should have an empty file called running in the run directory to run this 
# when this file is deleted, this run will be killed
# an easy way to kill this remotely w/o having process kill permissions
def monitor(threadnum):
  while True:
    time.sleep(2)    
    print("monitor...")
    if not os.path.exists("running"):
      print("monitor detected no running file...exit")
      # if the running file does not exist, wack this process
      # this should solve the case of long-running annealing jobs not getting killed fast upon user request
      mypid = os.getpid()    
      os.kill(mypid,signal.SIGKILL)

# for dot plotting
# return x and y arrays
def makeTradeRecommenedArrays(backtestForecastMXoutsideOfZTol):
  xdot = []
  ydot = []
  for i in range(0, len(backtestForecastMXoutsideOfZTol)):
    if backtestForecastMXoutsideOfZTol[i]:
      xdot.append(i)
      ydot.append(0.5)  

  return xdot, ydot

def adjR2(R2, n, p):
  return 1.0-(1.0-R2)*float(n-1.0)/float(n-p) 

def mydim(a):
      if not type(a) == list:
        return []
      return [len(a)] + mydim(a[0])

# to give a hint to the calling application what version is being run but not needed now
# print("BTC ETHM XRP BCH$ EOS$ ADA$ LTC$ ZEC and more new source tol")

if True:
  monitorThread = Thread(target=monitor, args=(1,))
  monitorThread.daemon = True
  monitorThread.start()

mypid = os.getpid()
#host = socket.gethostbyname(socket.gethostname())
host = 'any'

with open("runpid", "w") as outfile:
  outfile.write(str(mypid))
  outfile.write(" ")
  outfile.write(host)

debug = False

# output dict

jout = {}

jout["status"] = "starting job with polygon.io access"
jout["sys.version"] = sys.version

annealToUse = 'BFGS'
# note:  alignInPython must be used unless the compiled program that does file timestamp alignment is available

alignInPython = True # off py 2.7 server, align in python direct
if sys.version.startswith('2.7.3'):
  annealToUse = 'Anneal'   
  alignInPython = False # on py 2.7 server, can use compiled version

if skipio == False:
  with open("status", "w") as outfile:
    json.dump(jout, outfile)

# for LARS type regression
# uselogit = 0
# uselars = True 

# for KNearest classifier
# uselogit = 1
# uselars  = False 

# note:  cmd line input args should use 0 1 instead of False True

# coded as False True for code simplicity here

uselogit = 1 # various levels to be supported in the future?
uselars = False # uselogit should be zero also

useopen = True 

windowsize = 200 
ntrials = 9 # will run 1 more since starts at zero 
neighbors = 20
coolrate = 0 # fast 1 = slow 2 = (not anneal at all, minabs diff), 3 = BFGS (not anneal)
noboot=1 # if 1 skip boot run nominal
larsalpha=100
knnvarcutoff=50 # percent times 10
volen=21 # volatility length
scramblesens=0 # use scramble method for sens check
dyncutoff=0 # 1 dynamic cutoff experiment 0 = fixed cutoff
lassolarsbic=0 # use aic old default
shareCount=1.0
costPerTrade=10.0 # dollars
allowShorting=1
riskFreeRate=3.0
daysWithheld=0
reuseMergedRaw=0 # if 1, symbol list must be the same as when it was previously run with 0 (0 = full download)
pullDelay=15 # seconds between pulls to avoid overloading the API if you use polygon.io free plan. 

polyiokey = 'x'

# for forecasting, default
# for backtest, these are computed

m1ZTol=0
m2ZTol=0
m3ZTol=0

model1minabs=0

# not a cmd line arg yet
diffvol=1 # 0 do not daily difference the volatility, 1 daily difference the volatility

normalize=0

print((sys.version))

def factorial(n):
    if n < 2: return 1
    return reduce(lambda x, y: x*y, range(2, int(n)+1))

# BINOMDIST exact cumulative

def prob(s, p, n):
    #print("computing prob for ", s, p, n)
    #print s, p, n

    x = 1.0 - p

    a = n - s
    b = s + 1.0

    c = a + b - 1.0

    prob = 0

    for j in range(int(a), int(c + 1)):
        prob += factorial(c) / (factorial(j)*factorial(c-j)) \
                * x**j * (1 - x)**(c-j)

    #print("probresult is ", prob)

    return prob

probtest = prob(3,0.5,4)

#print("probtest") 
#print(probtest)

#scipy.optimize.show_options('minimize', 'Anneal')

#sys.exit(0)

# SPY should be first for now (or at least output target)

theargs = sys.argv

nargs = len(sys.argv) 

if nargs < 2:
  print("Usage:  several option=val ... sym1 sym2 sym3 ...")  
  print("option:  uselogit=[1|0]")
  print("option:  uselars=[1|0]")
  print("option:  useopen=[1|0]") 
  print("option:  coolrate=[0|1] 0 is fast 1 is slow (need to add more info here for 2 and 3)")
  print("option:  windowsize=200 points to include in regression")
  print("option:  ntrials=10 number of back test days")
  print("option:  neighbors=20 number of points in knearest calc if used")  
  print("option:  epsilon1000=2000 multiquadric param x 1000")
  print("option:  exponent=2.0 multiquadric exponent")
  print("option:  noboot=1 if set to 1, don't bootstrap, run nominal")
  print("option:  larsalpha=100  [ 0 to 100 ] div by 100 internal ... not used currently")
  print("option:  knnvarcutoff=50 percent cutoff times 10 for knn variable weedout default 50 which is 5%")
  print("option:  normalize=0 if 1, normalize table and predicted point before running KNN classifer")
  print("option:  volen=21 number of days to include in volatility calc") 
  print("option:  scramblesens=0 if 1 use scramble data method for variable sens estimate / KNN model")
  print("option:  dyncutoff=0 if 1 use dynamic cutoff method for variable sens estimate, ignore knnvarcutoff")
  print("option:  residExp=1.0 for min abs ^ residExp solve")
  print("option:  shareCount=1.0")
  print("option:  costPerTrade=10.0")
  print("option:  allowShorting=1")
  print("option:  daysWithheld=0")
  print("option:  polyiokey=undefined your polygon.io data access key for stock data, not needed for crypto only studies")
  print("option:  reuseMergedRaw=0 if 1 re-use data pulled from prior run of this file, symbol list must be same as prior run if 1")
  print("option:  pullDelay=15 (integer) seconds between datapulls to avoid rate limits on polgyon.io free plan, set to 0 for as fast as possible")

  print("more options may be available, see related iOS app")
  sys.exit(0)

# process arguments 
# not checked for scary values in general
theargs.pop(0)

origargs = copy.deepcopy(theargs)

# this is not showing 

print("---")
print("origargs =", origargs)
print("---")

allargs = copy.deepcopy(theargs)

for arg in theargs:
  print("checking arg ", arg)
  found = arg.find("=")

  if True:
    if found == -1: # all done w/ settings
      print("at arg ", arg, " done with settings")
      #origargs.pop(0)
      break
    else:
       keyval = arg.split("=")
       print("removing ", origargs[0], " from origargs")
       origargs.pop(0)

       if keyval[0] == 'polyiokey':
         locals()[keyval[0]] = keyval[1]
       else:
         #locals()[keyval[0]] = int(keyval[1])

         # should check for validity
         if '.' in keyval[1]:
           locals()[keyval[0]] = float(keyval[1])  
         else:
           locals()[keyval[0]] =   int(keyval[1])

if "epsilon1000" in locals():
  print("epsilon defined on cmd line as ", epsilon1000)
  epsilon = epsilon1000/1000.0
else:
  print("use default epsilon of 2.0")
  epsilon = 2.0

if "exponent" in locals():
  print("exponent defined on cmd line as ", exponent)
else:
  print("use default exponent of 2.0")
  exponent = 2.0

if "residExp" in locals():
  print("residExp defined on cmd line as ", residExp)
else:
  residExp = 1.0
  print("use default residExp of 1.0")

print("after arg processing:")
print("uselogit     = ", uselogit)
print("uselars      = ", uselars)
print("lassolarsbic = ", lassolarsbic)
print("useopen      = ", useopen)
print("windowsize   = ", windowsize)
print("ntrials      = ", ntrials)
print("neighbors    = ", neighbors)
print("coolrate     = ", coolrate)
print("epsilon      = ", epsilon)
print("noboot       = ", noboot)

print("m1ZTol       = ", m1ZTol)
print("m2ZTol       = ", m2ZTol)
print("m3ZTol       = ", m3ZTol)

print("shareCount   = ", shareCount)
print("costPerTrade = ", costPerTrade)
print("riskFreeRate = ", riskFreeRate)

# comment out for security if output is saved in a log file or otherwise
print("polyiokey  = ", polyiokey)

print("pullDelay = ", pullDelay)
 
# we are using the same var names for input (for a forecast)
# as during computation so to avoid confusion, copy these

in_m1ZTol = m1ZTol
in_m2ZTol = m2ZTol
in_m3ZTol = m3ZTol

if uselars == 1:
  uselars = True
else:
  uselars = False

if useopen == 1:
  useopen = True
else:
  useopen = False

if useopen == 0:
  useopen = False

print("final value of useopen = ", useopen)

if useopen == True and ntrials == -1:
  print("useopen = 1 and ntrials = -1 (pure forecast) is not supported yet")
  print("current opening day data not available yet")
  sys.exit(0)

#sys.exit(0)

print("the remaining args = ", origargs)

# orig test case
symbols = ["F", "XOM", "USO", "AA", "X", "DOW"]

symbols = origargs 

combined=""
for x in allargs:
  combined = combined + " " + x

print("combined = ", combined)

jout["params"] = combined

nsymbols = len(symbols)

def rhook(a, b, c):
  print("reporthook " + str(a) + " " + str(b) + " " + str(c))

#
# loop through symbols and download from proper source API if we don't set the reuseMergedRaw file flag on command line
# otherwise if reuseMergedRaw == 1, just read in mergedraw.csv left from prior run
# and put it into the mergedraw array (1 array element per file line, return separated)
#

if reuseMergedRaw == 0:
  for symbol in symbols:
    print("downloading ", symbol)

    # orig data sources now molto obsoleto

    fname = symbol + ".csv" 
    pname = "http://ichart.finance.yahoo.com/table.csv?s=" + symbol  
    ## google 
    pname = "http://www.google.com/finance/historical?q=" + symbol + "&startdate=Jan%201%2C%201977&output=csv"

    ## iex
    # should url encode the symbol
    # ah how sad iex is dead, look how it went thru 2 iterations of API types

    pname = "https://api.iextrading.com/1.0/stock/" + symbol + "/chart/5y" 
    pname = "https://cloud.iexapis.com/stable/stock/" + symbol + "/chart/5y?token=pk_af42e8548aff4b47a37dd41ff5e25e48"

    #print pname 

    ## polygon.io

    # from ObjC
    #iexString = [NSString stringWithFormat:@"https://api.polygon.io/v2/aggs/ticker/%@/range/1/day/2005-01-01/%d-%02d-%02d?sort=asc&apiKey=%@",
    #                            tsym, year, month, day, /*self.polygonIoKeyField.text*/polygonIoKey];

    xtoday = date.today()
    xformatted_date = xtoday.strftime("%Y-%m-%d")

    pname = "https://api.polygon.io/v2/aggs/ticker/" + symbol + "/range/1/day/2005-01-01/" + xformatted_date + "?sort=asc&apiKey=" + polyiokey

    ## from obj C ref
    ## https://min-api.cryptocompare.com/data/histoday?fsym=%@&tsym=USD&limit=2000
    
    ucasesymbol = symbol.upper()

    # some crypto symbols overlap with stock market symbols (e.g. ETH)
    # so we have special versions of these on left column here 
    # to indicate that we want the crypto coin and not the stock market symbol

    symlookup = {
      "BTC"  : "BTC",
      "ETHM" : "ETH",
      "XRP"  : "XRP",
      "BCH$" : "BCH",
      "EOS$" : "EOS",
      "ADA$" : "ADA", 
      "LTC$" : "LTC", 
      "ZEC"  : "ZEC",
      "XLM"  : "XLM", 
      "IOTA" : "IOTA",
      "TRX$"  : "TRX",
      "NEO$"  : "NEO",
      "XMR"   : "XMR",
      "DASH"  : "DASH",
      "USDT"  : "USDT",
      "XEM"   : "XEM",
      "VEN"   : "VEN",
      "BNB"   : "BNB", 
      "ETC"   : "ETC",
      "ONT"   : "ONT",
      "QTUM"  : "QTUM",
      "OMG"   : "OMG",
      "BCN"   : "BCN",
      "DOGE"  : "DOGE",
      "MIOTA" : "MIOTA",

      "SOL$"   : "SOL",
      "DOT"   : "DOT",
      "MATIC" : "MATIC",
      "TRX$"   : "TRX",
      "AVAX"  : "AVAX",
      "WBTC"  : "WBTC",
      "LEO$"   : "LEO",
      "UNI"   : "UNI",
      "FTT"   : "FTT",
      "CRO"   : "CRO",
      "LINK$"  : "LINK",
      "NEAR$"  : "NEAR",
      "ATOM$"  : "ATOM",
      "ALGO"  : "ALGO",
      "APE"   : "APE",

      "FLOW"  : "FLOW",
      "VET$"  : "VET",
      "FIL"   : "FIL",
      "ICP"   : "ICP",
      "MANA"  : "MANA",
      "SAND$" : "SAND",
      "XTZ"   : "XTZ",
      "HBAR"  : "HBAR",
      "AAVE"  : "AAVE",
      "THETA" : "THETA",
      "QNT"   : "QNT"
    
    }

    trypname = "" 
    isCrypto = False

    try: 
      trypname = 'https://min-api.cryptocompare.com/data/histoday?fsym=' + symlookup[ucasesymbol] + "&tsym=USD&limit=2000"
      isCrypto = True
    except:
      print("no crypto symbol found, using polygon.io feed")

    if trypname != "":
      pname = trypname
      print("pulling from: " + pname)
  
    # have to delay for polygon.io to avoid hitting rate limits on free plans for larger number of stock symbols pulled
    # if you have less than 5 symbols it may work w/o a delay
    # also it may work for crypto w/o a delay but this is just a sledge hammer patch for now, no subtlety

    # can comment this out if you have a paid plan w/ better rate limits

    # we keep the iex.json output file name even though it is polygon now
    
    time.sleep(pullDelay)
    call(["curl", "-k", "-o", "iex.json", pname]) # use curl instead

    # iex code here should translate JSON to csv
    iexout = "Date,Open,High,Low,Close,Volume\n" 

    with open("iex.json", 'r') as jsonIn:
      content =  jsonIn.read()
      try:
        iex_json = json.loads(content)
      except Exception as e: # will get here on bad symbol or data feed error
        print("bad symbol or data feed error")
        iex_json = []

        jout["status"] = "bad symbol or data feed error\n\nYou may have typed a symbol wrong, please check then re-run."
        
        with open("status", "w") as outfile:
          json.dump(jout, outfile)

        sys.exit(0)
        
      if isCrypto :
        iex_json = iex_json["Data"] # price data is one level down
      else : 
        try:
          iex_json = iex_json["results"] # polygon
        except Exception as eResults:
          print("bad response from polygon.io")
          iex_json = []

          jout["status"] = "bad response from polygon.io\n\ncheck your polygon.io key and/or stock symbol corrrectness"
          with open("status", "w") as outfile:
            json.dump(jout, outfile)

          sys.exit(0)
          
      print(str(len(iex_json)) + " data points")
      kiexRange = reversed(range(len(iex_json))) # iex order

      ## new crypto source do it the same as iex
      ## if symbol.upper() == "BTC" or symbol.upper() == "ETHM" or symbol.upper() == "XRP" or symbol.upper() == "BCH$" or symbol.upper() == "EOS$" or symbol.upper() == "ADA$" or symbol.upper() == "LTC$" : 
      ##      kiexRange = (xrange(len(iex_json))) # btc order
      
      for kiex in kiexRange: 
        #print kiex
        diex = iex_json[kiex] # dictionary for 1 day
        # convert dict to csv
        try:
          # probably can do this more elegantly from above symbol overlap conversion table
          # iex
          if (symbol.upper() != "BTC" and 
              symbol.upper() != "ETHM" and 
              symbol.upper() != "XRP" and 
              symbol.upper() != "BCH$" and 
              symbol.upper() != "EOS$" and 
              symbol.upper() != "ADA$" and 
              symbol.upper() != "LTC$" and 
              symbol.upper() != "ZEC" and 
              symbol.upper() != "XLM" and 
              symbol.upper() != "IOTA" and 
              symbol.upper() != "TRX$" and 
              symbol.upper() != "NEO$" and 
              symbol.upper() != "XMR" and 
              symbol.upper() != "DASH" and 
              symbol.upper() != "USDT" and 
              symbol.upper() != "XEM" and 
              symbol.upper() != "VEN" and 
              symbol.upper() != "BNB" and 
              symbol.upper() != "ETC" and 
              symbol.upper() != "ONT" and 
              symbol.upper() != "QTUM" and 
              symbol.upper() != "OMG" and 
              symbol.upper() != "BCN" and 
              symbol.upper() != "DOGE" and 
              symbol.upper() != "MIOTA" and 

              symbol.upper() != "SOL$" and 
              symbol.upper() != "DOT" and 
              symbol.upper() != "MATIC" and 
              symbol.upper() != "TRX$" and
              symbol.upper() != "AVAX" and
              symbol.upper() != "WBTC" and
              symbol.upper() != "LEO$" and
              symbol.upper() != "UNI" and
              symbol.upper() != "FTT" and
              symbol.upper() != "CRO" and
              symbol.upper() != "LINK$" and
              symbol.upper() != "NEAR$" and
              symbol.upper() != "ATOM$" and
              symbol.upper() != "ALGO" and
              symbol.upper() != "APE" and
  
              symbol.upper() != "FLOW" and
              symbol.upper() != "VET$" and
              symbol.upper() != "FIL" and
              symbol.upper() != "ICP" and
              symbol.upper() != "MANA" and
              symbol.upper() != "SAND$" and
              symbol.upper() != "XTZ" and
              symbol.upper() != "HBAR" and
              symbol.upper() != "AAVE" and
              symbol.upper() != "THETA" and
              symbol.upper() != "QNT" ) :

            #iexline = str(diex['date']) + ',' + str(diex['open']) + ',' + str(diex['high']) + ',' + str(diex['low']) + ',' + str(diex['close']) + ',' + str(diex['volume']) + '\n'

            # polygon but keep variables for iex naming convention
              
            from datetime import datetime

            xtime_interval = diex['t'] / 1000
            xdate = datetime.fromtimestamp(xtime_interval)
            xdate_string = xdate.strftime('%Y-%m-%d')

            iexline = xdate_string + ',' + str(diex['o']) + ',' + str(diex['h']) + ',' + str(diex['l']) + ',' + str(diex['c']) + ',' + str(diex['v']) + '\n'

          else:
            #print diex
            # process BTC/ETHM/XRP/BCH$/etc data from slickcharts.com (old data source)
            # diex['time'] all remainder the same
            # jun23 2018 mod for new data source not slickcharts
            # print "time = " + str(diex['time'])
            BTCdatePrelim = dt.fromtimestamp(diex['time']-3600*5)
            BTCdate = BTCdatePrelim.strftime('%Y-%m-%d') # may have been wrong for slickcharts, didnt have the bump date like in objC

            iexline = BTCdate + ',' + str(diex['open']) + ',' + str(diex['high']) + ',' + str(diex['low']) + ',' + str(diex['close']) + ',' + str(diex['volumeto']) + '\n'  
          iexout = iexout + iexline
        except Exception as e:
          print(e)
          print("caution, error trying to parse at kiex = ", kiex, " symbol = ", symbol)

      text_file = open(fname, "w")
      text_file.write(iexout)
      text_file.close()

    with open(fname, 'r') as content_file:
      content = content_file.read()
    
    if content.find("404 Not Found") >= 0:
      print("error:  unable to download history file for " + symbol)
      print("stopping")
      jout["status"] = "error:  unable to download history file for " + symbol
      jout["alldone"] = "1";
      with open("status", "w") as outfile:
        json.dump(jout, outfile)
      sys.exit(0)
    
    #print b 

  # new way, over loop

  # need to have a list of the max number of possible CSV files init'd

  allraw = [0]*101

  ncsv = 0

  for symbol in symbols:
    fname = symbol + ".csv"
    print("reading ", fname)
    thisraw = []

    reader = csv.reader(open(fname))

    for row in reader:
      thisraw.append(row)  

    print(len(thisraw), " lines read")
    allraw[ncsv] = copy.deepcopy(thisraw) # does this copy it?  now it does

    ncsv += 1

  #print allraw

  mergedraw = []

  spyraw = allraw[0] # not necessarily spyraw, just the first symbol

  print("aligning dates")

  if alignInPython: # orig way to align in python (possibly easier to do w/ pandas but this script was written without pandas)

    print("aligning dates in python")
    for spyrow in spyraw:
      thedate = spyrow[0]

      if debug:
        print("for date ", thedate, "\r", end=' ')

      jout["status"] = "aligning data at " + thedate

      if skipio == False:
        with open("status", "w") as outfile:
          json.dump(jout, outfile)

      if not os.path.exists("running"):
        print("user terminated process")
        jout["status"] = "user terminated process"
        jout["alldone"] = "1"    
        with open("status", "w") as outfile:
          json.dump(jout, outfile)

        sys.exit(0)

      found = 0
      
      # not necessarily gldraw, just the next symbol 

      icsv = 0

      for symbol in symbols:

        gldraw = allraw[icsv] # here is where to index

        if icsv == 0:
          #print "don't merge 1st reference csv"
          icsv += 1
          continue
      
        #print "merging ", symbol

        #print "gldraw is:"
        #print gldraw
    
        for gldrow in gldraw:
          if thedate == gldrow[0]:
            if icsv == 1:
              mergedrow = spyrow + gldrow
              #print "first merge row len = ", len(mergedrow)
            else:
              mergedrow = mergedrow + gldrow
              #print "next merge row len = ", len(mergedrow)

            # orig loc:  mergedraw.append(mergedrow)
            found = 1
            break

        # move away from here for multifile
        #mergedraw.append(mergedrow)

        #if found == 0:
        #  print "gldrow not found for ", thedate
        
        icsv += 1

      # move to here for multifile
      # 7 columns per csv from yahoo finance (6 google)
      if len(mergedrow) == (6*nsymbols): 
        #print "appending after row ", len(mergedraw)
        mergedraw.append(mergedrow)
      else:
        dum = 1
        #print thedate, " skipping merge, line too short"

  else: # compiled (C) date align (.cpp really)

    print("running compiled align code")

    jout["status"] = "aligning data"

    with open("status", "w") as outfile:
      json.dump(jout, outfile)

    scriptPath = getScriptPath()

    print("scriptPath = ", scriptPath)

    call_list = [scriptPath + "/align"]

    # need to append the CSV file names with csv suffix

    for symbol in symbols:
      csvsymbol = symbol + ".csv"
      call_list.append(csvsymbol)

    call(call_list) # should write mergedrawfast.csv

    #print "beforereadin, mergedraw len = ", len(mergedraw)
    mergereader = csv.reader(open("mergedrawfast.csv"))

    mergedraw = []

    for row in mergereader:
      newrow = []
      for rowitem in row:
        if rowitem == "-":
          rowitem = 0   
        newrow.append(rowitem)
      mergedraw.append(newrow)

    print(len(mergedraw), " lines read from mergedrawnew.csv")

  # delete days withheld from top of file (recent days withheld)

  if daysWithheld > 0:
    print("deleting", daysWithheld, "most recent days from mergedraw")
    del mergedraw[1:1+daysWithheld]

  myfile = open("mergedraw.csv", 'w')
  wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)

  for thelist in mergedraw:
    wr.writerow(thelist)

  myfile.close()

else: # use precomputed mergedraw.csv file for re-entry at top level of script (e.g. re-run for model tuning without having to re-pull data)
  mergedraw = []
  myfile = open("mergedraw.csv", 'r')
  wr = csv.reader(myfile, quoting=csv.QUOTE_ALL)

  for row in wr:
    mergedraw.append(row)

  myfile.close()

#sys.exit(0)

#print mergedraw

print("merged raw count = ", len(mergedraw))      

#sys.exit(0)

if len(mergedraw) <= 2:
   jout["status"] = "error getting raw stock data, check your stock symbols, no commas allowed between symbols"
   with open("status", "w") as outfile:
     json.dump(jout, outfile)
   sys.exit(0) 

#for r in mergedraw:
#  print r

# 0    1    2    3   4     5      6
# Date,Open,High,Low,Close,Volume,Adj Close  
# 7    8    9    10  11    12     13
# etc every 7

# the pattern is 1,4 = open close, the +7 for each additional 

thedate = 0
spyopen = 1
spyclose = 4
gldopen = 8
gldclose = 11

#regtable = [["date", "spyc-o", "spyclose", "spyclose1", "spyclose2", 
#                    "spyopen",  "spyopen1",  "spyopen2", 
#                                "gldclose1", "gldclose2", 
#                    "gldopen",  "gldopen1",  "gldopen2"]]

# experiment, add trailing volatility 1vol to each symbol (1 day prior)
# vol computed from close

#regtable = [["date", "spyc-o", "spyclose", "spyclose1", "spyclose2", 
#                    "spyopen",  "spyopen1",  "spyopen2", "spyclose1vol", 
#                                "gldclose1", "gldclose2", 
#                    "gldopen",  "gldopen1",  "gldopen2", "gldclose1vol"]]

# probably should construct this dynamically

regtable = []
regtableo = [] # include current day open prices and close prices for future computation

headerline = ["date"]
headerlineo = ["date"]

txt  = symbols[0] + "c-o"
txtc = symbols[0] + "close"

headerline.append(txt)
headerline.append(txtc)

headerlineo.append(txt)
headerlineo.append(txtc)

print("headerline = ", headerline)
print("headerlineo = ", headerlineo)

firstsym = 1

for symbol in symbols:
  print(symbol)
  txt0 = symbol + "close"  # not used for first symbol ... not used in ordinary regression table either
  txtA = symbol + "close1"
  txtB = symbol + "close2"
  txtC = symbol + "open"
  txtD = symbol + "open1"
  txtE = symbol + "open2"
  txtF = symbol + "close1vol"
  headerline.append(txtA)
  headerline.append(txtB)
  if useopen:
    headerline.append(txtC)
  headerline.append(txtD)
  headerline.append(txtE)
  headerline.append(txtF)

  if firstsym == 0:
    headerlineo.append(txt0) # including todays close price for later future calculation
  headerlineo.append(txtA)   # including todays open price
  headerlineo.append(txtB)
  headerlineo.append(txtC)
  headerlineo.append(txtD)
  headerlineo.append(txtE)
  headerlineo.append(txtF)
  firstsym = 0

print("full headerline = ", headerline)
print("full headerlineo = ", headerlineo)

#sys.exit(0)
 
regtable.append(headerline)
regtableo.append(headerlineo)

# regression table cols

# not used for generic multi symbol case ... this was the initial hard coded setup
# forecast SPYclose - open as a function of spy & gold lags (also optionally open prices)

SPYCO = 1
SPYC  = 2
SPYC1 = 3
SPYC2 = 4
SPYO  = 5
SPYO1 = 6
SPYO2 = 7
GLDC1 = 8
GLDC2 = 9
GLDO  = 10
GLDO1 = 11
GLDO2 = 12

#print "mergedraw row zero ", mergedraw[0]

#print "len mergedraw = ", len(mergedraw)

#print "range ", range(1,len(mergedraw)-2)

# orig way to make the reg table hardcoded

# see above, in original merged rows... 
# the pattern is 1,4 = open close, the +7 for each additional symbol

print("creating regression table")

# this is for regline only not reglineo (with today's open prices in it)

donotdifference  = [] # column number, we may not want to difference volatility
donotdifferenceo = [] # for reglineo column counts

for i in range(1, len(mergedraw) - 2):
  regline = [mergedraw[i][thedate]]
  reglineo = [mergedraw[i][thedate]] # with open prices (current day) ... needed for future forecast
 
  closeidx = 4
  openidx  = 1

  # orig
  tgtvalue = float(mergedraw[i][closeidx]) - float(mergedraw[i][openidx]) # close - open

  # experiment
  # tgtvalue = float(mergedraw[i][closeidx]) - float(mergedraw[i+1][closeidx]) # close - close1 
 
  tgtclose = float(mergedraw[i][closeidx]) # include this for checking but don't put it in the regression (or we can use it as an alternate solve target)

  regline.append(tgtvalue)
  regline.append(tgtclose)

  reglineo.append(tgtvalue)
  reglineo.append(tgtclose)

  firstsym = 1

  # now we loop over the symbols
  for symbol in symbols:
    try:
      c0 = float(mergedraw[i][closeidx]) # for reglineo only ... need current day's close price
    except:
      c0 = 0
    try:
      c1 = float(mergedraw[i+1][closeidx])
    except:
      c1 = 0
    try:
      c2 = float(mergedraw[i+2][closeidx])
    except:
      c2 = 0
    try:
      o0 = float(mergedraw[i][openidx])
    except:
      o0 = 0 
    try:
      o1 = float(mergedraw[i+1][openidx])
    except:
      o1 = 0
    try:
      o2 = float(mergedraw[i+2][openidx])
    except:
      o2 = 0

    v1 = float(mergedraw[i+1][closeidx]) # placeholder for volatility 1 day lagged but this is price 1 d lag

    # actually compute vol after the differencing is done?  or what?  differencing is generic for columns so maybe do it here first

    level1data = []

    # copy as float the right number of values

    # 10 day vol for now

    volatilitylength = volen # really 1+ due to range logic / definition

    if volatilitylength < 2:
      volatilitylength = 2

    for thislag in range(0, volatilitylength):
      row4vol = i+1+thislag 
      if row4vol < len(mergedraw): # dont roll off the bottom of the matrix
        lev = float(mergedraw[row4vol][closeidx]) # need to convert to float
        level1data.append(lev)

    # difference in place, note we go one less to avoid running past the end
    for thislag in range(0, len(level1data)-1):
      level1data[thislag] = level1data[thislag] - level1data[thislag+1] 

    # wack the last item which is a level not a diff to avoid confustion 
    level1data.pop()

    #print "before computing stdev for vol, list is ", level1data

    thestdev = numpy.std(level1data)

    #print "thestdev = ", thestdev

    v1 = thestdev # later we want to not difference this further, or...do we?   can leave it or not, we shall see
             
    regline.append(c1)
    regline.append(c2)
    if useopen:
      regline.append(o0)
    regline.append(o1)  
    regline.append(o2)
    regline.append(v1) # vol placeholder ... now actual volatility

    if i == 1: # first time only 
      coltonot = len(regline) - 1
      ### DIFFERENCE THE VOLATILITY FOR NOW 
      if diffvol == 0:
        donotdifference.append(coltonot)

    # add to the line for including today's opening prices

    if firstsym == 0:
      reglineo.append(c0) # add todays closing for all dependent symbols
    reglineo.append(c1)
    reglineo.append(c2)
    reglineo.append(o0)
    reglineo.append(o1)
    reglineo.append(o2)
    reglineo.append(v1) # vol placeholder

    if i == 1: # first time only 
      coltonot = len(reglineo) - 1
      ### DIFFERENCE THE VOLATILITY FOR NOW 
      if diffvol == 0:
        donotdifferenceo.append(coltonot)

    firstsym = 0

    # was 7, 6 is for google
    if useopen:
      bump = 6 
    else:
      bump = 6 
    closeidx += bump 
    openidx  += bump 
 
  regtable.append(regline)
  regtableo.append(reglineo)

print("donotdifference  = ", donotdifference)
print("donotdifferenceo = ", donotdifferenceo)

myfile = open("regtable.csv", 'w')
wr = csv.writer(myfile)

# move this until after the daily diffs are computed

if False:
  regtable_tmp = copy.deepcopy(regtable) # prep to wack the dates and headers if needed

  print("regtable sizes")
  print(len(regtable_tmp))
  print(len(regtable_tmp[0]))

  for i in range(0,len(regtable_tmp)):
    regtable_tmp[i][0] = 0 # wack the text date ... or convert to a number later

  for i in range(0,len(regtable_tmp[0])):
    regtable_tmp[0][i] = 0 
 
  regtable_fast = numpy.array(regtable_tmp, dtype=float)

  print(type(regtable_fast))

  print(regtable[:3][:3])
  print("---")
  print(regtable_fast[:3,:3])

# quoting=csv.QUOTE_ALL

# has a header here
print("debug regtable[0] = ", regtable[0])

for thelist in regtable:
  wr.writerow(thelist)

myfile.close()

myfile = open("regtableo.csv", 'w')
wr = csv.writer(myfile)

# quoting=csv.QUOTE_ALL

for thelist in regtableo:
  wr.writerow(thelist)

myfile.close()

#sys.exit(0)

#print "regtable"
#for i in range(0, 8):
#  print regtable[i]

#sys.exit(0)

# save levels version
# should this be a deep copy?

regtableL = regtable
regtableLo = regtableo

firstrow = regtableL[1]
firstrowo = regtableLo[1]

print("number of columns = ", len(firstrow))
print("number of columnso = ", len(firstrowo))

ncolsrt = len(firstrow)
ncolsrto = len(firstrowo)

#sys.exit(0)

# ah hah, regtable gets recomputed here without header and differenced
# we should put the forecasts on regtableL (levels)

regtable = []
regtableo = []

#print "len(regtableL) ", len(regtableL)

#print "range ", range(1, len(regtableL)-1)

# creating the differences
# note that regtable before has a header line; after differencing, no.
# because we are just appending rows and started with an empty list

print("computing daily differences")

# set the False: to avoid differencing for test

for i in range(1, len(regtableL) - 1):
  regline = [regtableL[i][0],
             regtableL[i][1]]

  reglineo = [regtableLo[i][0],
              regtableLo[i][1]]

  for j in range(2, ncolsrt): 
    #print "j = ", j
    # skip diff computation if this is a volatility column...trial
    if not (j in donotdifference):
      regline = regline + [(regtableL[i][j]-regtableL[i+1][j])]
    else:
      regline = regline + [regtableL[i][j]]
 
  # now do for the table that has the open values
  for j in range(2, ncolsrto): 
    #print "j = ", j
    if not (j in donotdifferenceo):
      reglineo = reglineo + [(regtableLo[i][j]-regtableLo[i+1][j])]
    else:
      reglineo = reglineo + [regtableLo[i][j]]

  regtable.append(regline)
  regtableo.append(reglineo)

myfile = open("regtableDiff.csv", 'w')
wr = csv.writer(myfile)

wr.writerow(regtableL[0]) # header line from Levels matrix

for thelist in regtable:
  wr.writerow(thelist)

myfile.close()

# now write full table with open columns
myfile = open("regtableDiffo.csv", 'w')
wr = csv.writer(myfile)

wr.writerow(regtableLo[0]) # header line from Levels matrix

for thelist in regtableo:
  wr.writerow(thelist)

myfile.close()

# need to construct the forward forecast row here just in case we are not backtesting
# this code is fraught with indexing peril
# is there an easier way?

#                                   fclose1 is fclose from "yesterday"
#                                   note that we are putting the data in col 4 but pulling from col 3
#                                      (zero indexed)

#                     c-o  close    fclose1         fclose2 


# full row

jout["forecast_day"] = "DAY_AFTER_"+regtable[0][0]

futureX = ["DAY_AFTER_"+regtable[0][0], "toforecast", "unknown"];

k = 2 # this is the first close column in the full "with opens" regression table
# this will become close1

print("futureX with volatility data")

for symbol in symbols:

 futureX = futureX + [regtableo[0][k]]   # now close1
 futureX = futureX + [regtableo[0][k+1]] # now close2 ... skip close2 or it will be close3

 futureX = futureX + [regtableo[0][k+3]] # now open1
 futureX = futureX + [regtableo[0][k+4]] # now open2 ... ah, now we are missing close1 which was close before for the next symbol
                                         # now regtableo should have the close values in the proper spot

 futureX = futureX + [regtableo[0][k+5]] # v1:  now vol (of close) 1

 # orig way w/o volatility k = k + 6
 k = k + 7 # 1 more variable v1 ... need to check this

futureXfull = futureX

print("futureXfull = ", futureXfull)
print("number of columns = ", len(futureXfull))

myfile = open("regtableDiffWithFuture.csv", 'w')
wr = csv.writer(myfile)

wr.writerow(regtableL[0]) # header line from Levels matrix

wr.writerow(futureXfull)

for thelist in regtable:
  wr.writerow(thelist)

myfile.close()

# here, copy to fast numpy array

# bootstrap should loop around here I think

# or for a 1 iteration boot

boottable = copy.deepcopy(regtable)

# orig way of bootstrap...whole table...too much boot
# should bootstrap just the regressed items for each windowed step
# does this have a header?
##for i in range(0,len(regtable)):
##  therand = random.randint(0, len(regtable)-1) # with replacement so can choose the same again
##  regtable[i] = boottable[therand]

# no header it seems
print("first line in regtable") 
print(regtable[0])

regtable_tmp = copy.deepcopy(regtable) # prep to wack the dates and headers if needed

print("regtable sizes")
print(len(regtable_tmp))
print(len(regtable_tmp[0]))

for i in range(0,len(regtable_tmp)):
  regtable_tmp[i][0] = 0 # wack the text date ... or convert to a number later

# don't wack the first row
#for i in range(0,len(regtable_tmp[0])):
#  regtable_tmp[0][i] = 0 
 
regtable_fast = numpy.array(regtable_tmp, dtype=float)

regtable_tmp = 0 # free mem

print(type(regtable_fast))

print(regtable[:4][:4])
print("---")
print(regtable_fast[:4,:4])

#for theline in regtable:
#  print theline

#sys.exit(0)

scale = 1.0

forecastrow = 1

# coeffs initial (start w/ constant)
# are there issues w/ limited array size?

A = [0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0]

best = A
bestobj = 0

# forecast row i using model A

t2 = 1 
c = 1 

print("start backtesting")

def distfunc(a):
  #print("distfunc input type", type(a), " size = ", a.shape, " exponent =", exponent)

  theshape = a.shape

  minval = 9999999
  maxval = -9999999

  if False: # not used yet
   for i in range(0,theshape[1]):
     for j in range(0,theshape[0]):
       if (a[j,i] < minval):
         minval = a[j,i]
       if (a[j,i] > maxval):
         maxval = a[j,i]

  for i in range(0,theshape[1]):
    for j in range(0,theshape[0]):
      ## define in cmd line processing:  epsilon = 2 
      #print "iter" 
      ### a[j,i] = pow(2.71828, -(a[j,i]*a[j,i]))
      #print 'dist a[',j,i,']',a[j,i]
      #if (a[j,i] == 0): 
      #  print "zero at j,i = ", j,i
      ## the real one mquad

      ### cmd line now 
      ### exponent = 2.0 # must be float

      a[j,i] = pow(1 + pow(epsilon*a[j,i], exponent), 1/exponent) # mquad

      #a[j,i] = (1 - numpy.cos(3.14159*a[j,i]/(maxval)))*epsilon

      #clip = 35
      #if a[j,i] > clip:
      #	a[j,i] = clip 

      #a[j,i] = pow(2.71828, -epsilon*(a[j,i]*a[j,i]))

      ### kind of good
      #a[j,i] = 1 + epsilon*abs(a[j,i])

      #a[0,i] = a[0,i]*a[0,i]*math.log(a[0,i])
      
  #print "minval maxval distfunc =", minval, maxval

  return a 

# also use j index, hmm

def distfuncSpeed(a):
  print("distfuncSpeed input type", type(a), " size = ", a.shape)

  theshape = a.shape

  endj = theshape[0]
  endi = theshape[1]

  ## IMP for i in range(0,theshape[1]):
    # IMPLICIT for j in range(0,theshape[0]):
      ### endj = theshape[0]
      ## define in cmd line processing:  epsilon = 2 
      #print "iter" 
  a[0:endj,0:endi] = pow(2.71828, -(a[0:endj,0:endi] * a[0:endj,0:endi]))
      ## a[j,i] = pow(1 + pow(epsilon*a[0,i], 2), 0.5) # mquad
      #a[0,i] = a[0,i]*a[0,i]*math.log(a[0,i])

  return a

# if i == -1 pull x data from futureXfull

# return correct and actual forecast in 2nd item of tuple
 
def forecast1(A, i):
  
   fit = 1*A[0]

   if i == -1:
     print("forecast1 for futureXfull")
     row = futureXfull
   else:
     row = regtable[i]

   print("rowsize ", len(row))
   print("A size ", len(A))

   # orig way
   if i >= 0:
     for j in range(1,ncolsrt-2):
       # need to start at 3 so j starts at 1
       #print "ij = ", i, " ", j
       fit = fit + A[j]*regtable[i][j+2]
       #print " forecast Xi = ", regtable[i][j+2]

     print("ap = ", regtable[i][SPYCO], " ", fit)

     if ((fit > 0) and (regtable[i][SPYCO] > 0)) or ((fit < 0) and (regtable[i][SPYCO] < 0)):
        correct = 1
        #print "GOOD"
     else:
        correct = 0
        #print "BAD"

   else: # futureXfull

     for j in range(1,ncolsrt-2):
       # need to start at 3 so j starts at 1
       #print "ij = ", i, " ", j
       fit = fit + A[j]*futureXfull[j+2]
       #print " forecast Xi = ", regtable[i][j+2]

     print("ap = ", futureXfull[SPYCO], " ", fit)

     # 1st clause check for string to fix crash during 1 day ahead forecasting in python 3.11
     if (futureXfull[SPYCO] == 'toforecast' or (fit > 0) and (futureXfull[SPYCO] > 0)) or ((fit < 0) and (futureXfull[SPYCO] < 0)):
        correct = 0 
        print("FORECAST NOT TESTED YET")
     else:
        correct = 0
        print("FORECAST NOT TESTED YET")

   return correct, fit

global objcount

# PRODUCTION MIN ABS DIFF

def theobjAbs(A):

  varMask = []

  for i in range(0, len(A)):
    varMask.append(1.0)
    
  return theobjAbsPreMask(varMask, A)

# need to bind varMask variable to send this to the optimizer

def theobjAbsPreMask(varMask,A):
  #return 1  # for speed

  localY = []
  localYHat = []

  global objcount

  objcount += 1

  #print "computing ", objcount, "\r",  

  fit = [0 for y in range(0, len(regtable)+1)] # mod

  #print fit

  ncorrect = 0

  absdiffsum = 0

  #print "rangetest ", range(forecastrow + 1, forecastrow + 1 + windowsize)

  for i in range(forecastrow + 1, forecastrow + 1 + windowsize):

    fit[i] = 1*A[0]*varMask[0]

    # both show 10 for 1 symbol predictor
    #print "varlen no const", len(A[1:ncolsrt-2])
    #print "A len", len(A)

    #print "datalen no const", regtable_fast[i, 3:ncolsrt].shape

    ASub = A[1:ncolsrt-2] # will not be used soon

    AX = copy.deepcopy(A) # so same indexing as A

    for j in range(0, len(AX)):
      if varMask[j] == 0:
        AX[j] = 0

    thedot = numpy.dot(AX[1:ncolsrt-2], regtable_fast[i, 3:ncolsrt])
    fit[i] += thedot

    #if ((fit[i] > 0) and (regtable_fast[i,SPYCO] > 0)) or ((fit[i] < 0) and (regtable_fast[i,SPYCO] < 0)):
    #   ncorrect += 1

    # verified this is not locked to a binary output yet
    # print "regtable_fast[i,SPYCO]", regtable_fast[i,SPYCO]

    # absdiff = pow(abs(fit[i] - regtable_fast[i,SPYCO]),3) #*(fit[i] - regtable_fast[i,SPYCO])

    # use squared diff to test versus analytic linear
    absdiff = pow(abs(fit[i] - regtable_fast[i,SPYCO]), residExp) # *abs(fit[i] - regtable_fast[i,SPYCO])

    localY.append(regtable_fast[i,SPYCO])
    localYHat.append(fit[i])

    absdiffsum += absdiff

  R2 = r2_score(localY, localYHat)
 
  nparams = numpy.count_nonzero(varMask)
 
  localAdjR2 = adjR2(R2, len(localY), nparams)

  #print "local abs diff R2 of fit =", r2_score(localY, localYHat), "Adjust R2 =", localAdjR2, "nparams =", nparams
  
  #print "absdiffsum = ", absdiffsum

  return absdiffsum

def theobj(A):
  varMask = []
  for i in range(0, len(A)):
    varMask.append(1.0)
 
  return theobjPreMask(varMask, A)

def theobjPreMask(varMask, A):
  #return 1  # for speed
  #print "start of obj function"

  print("varMask =", varMask)

  global objcount

  objcount += 1

  #print "computing ", objcount, "\r",  

  fit = [0 for y in range(0, len(regtable)+1)] # mod

  #print fit

  ncorrect = 0

  checksum = 0

  #print "rangetest ", range(forecastrow + 1, forecastrow + 1 + windowsize)

  for i in range(forecastrow + 1, forecastrow + 1 + windowsize):

    fit[i] = 1*A[0]*varMask[0]

    #nvar = 1

    #if False:

    AX = copy.deepcopy(A) # so same indexing as A

    for j in range(0, len(AX)):
      if varMask[j] == 0:
        AX[j] = 0

    thedot = numpy.dot(AX[1:ncolsrt-2], regtable_fast[i, 3:ncolsrt])
    fit[i] += thedot

    #else: 
    #  for j in range(1,ncolsrt-2):
    #    # need to start at 3 so j starts at 1
    #    #print "ij = ", i, " ", j
    #    #print type(A[j])
    #    #print type(regtable_fast[i, j+2])
    #    #TEST fit_i = fit[i]
    #    fit[i] = fit[i] + A[j]*regtable_fast[i, j+2]
    #    #TEST fit_i = fit_i + A[j]*regtable[i][j+2]

      #TEST print fit[i], " ", fit_i
    #  nvar += 1

    #print "nvar = ", nvar

    if ((fit[i] > 0) and (regtable_fast[i,SPYCO] > 0)) or ((fit[i] < 0) and (regtable_fast[i,SPYCO] < 0)):
       ncorrect += 1
    checksum += fit[i]

  #for q in A:
  #  #print "q = ", q
  #  if (q < -10 or q > 10):
  #    # penalize out of bounds
  #    ncorrect = ncorrect - int(abs(q))

  if True or debug:
    print("theobj ncorrect = ", ncorrect) #, "\r",

  return -ncorrect

#print "range 0 12 ", range(0,12)

for j in range(0, 18+1):
    A[j] = 0 


B0 = 0
B1 = 0
B2 = 0
B3 = 0
B4 = 0
B5 = 0
B6 = 0
B7 = 0
B8 = 0 
B9 = 0
B10 = 0
B11 = 0
B12 = 0

# may be too many?

# need to make this generic  ncols of regression table - date,output,closeofTarget (-3)

# 1st is constant

print("ncolsrt = ", ncolsrt)

Bfeed = [1] * (1 + ncolsrt-3)

print("Bfeed len = ", len(Bfeed))

larsAdjR2List = []
linearAdjR2List = []

# this is the input array to control the optimizer (how many variables)

B = numpy.array(Bfeed)

forecastcorrect = 0
forecastcorrect2 = 0
forecastcorrect3 = 0 # lars knn (model 3)

badrun = 0
badrun2 = 0
badrun3 = 0

# for ztol versions of bad run sequence
badrunZTol = 0
badrun2ZTol = 0
badrun3ZTol = 0

# not used yet
goodrun = 0
goodrun2 = 0
goodrun3 = 0

badrunsequence = []
badrunsequence2 = []
badrunsequence3 = []

badrunstair = []
badrunstair2 = []
badrunstair3 = []

# not computed inline with the above
# but have to wait until we get ztols that we like

badrunsequenceZTol = []
badrunsequence2ZTol = []
badrunsequence3ZTol = []

badrunstairZTol = []
badrunstair2ZTol = []
badrunstair3ZTol = []

backtestActual = []
backtestForecastM1 = []
backtestForecastM2 = []
backtestForecastM3 = []

nonZeroCoeffsPerBacktestM1 = []
nonZeroCoeffsPerBacktestM3 = []

# previously started at 1 because we thought we had a header row

# starting at neg 1 means forecast the future

def probmsg(p):
  if (p > 0.2):
    return "Poor" 
  elif (p <= 0.2 and p > 0.1):
    return "Marginal"
  elif (p <= 0.1 and p > 0.05):
    return "Fair" 
  elif (p <= 0.05 and p >= 0.01):
    return "Good"
  elif (p <= 0.01):
    return "Very Good"
 
startrow = 0

if (ntrials == -1):
  startrow = -1 

# check dist func

a0trial = numpy.array([[0,0.1],[0.2,0.3]])

print("orig distfunc = ")
print(distfunc(a0trial))

a0trial = numpy.array([[0,0.1],[0.2,0.3]])

print("new distfunc = ")
print(distfuncSpeed(a0trial))

### sys.exit(0)

sumOfRowTimes = 0
sumOfRowTimesCount = 0

for forecastrow in range(startrow,ntrials+1):

  startTimeOneRow = 0 # time.clock()

  if not os.path.exists("running"):
    print("user terminated process")
    sys.exit(0)

  print("")
  print("*** forecast row ", forecastrow) 

  d = {}
  d['disp'] = True
  d['maxiter'] = 1000000
  #d['schedule'] = 'boltzmann'

  # fast, otherwise use default slow
  if coolrate == 0: 
    d['dwell'] = 1
    ### d['maxiter'] = 1 # hack...don't do much here for speed
 
  d['lower'] = -10
  d['upper'] = 10
  objcount = 0

  print("Annealing/BFGS solver...")
  print("proper random seed")
  numpy.random.seed(100)

  if model1minabs == 1:
    theresult = scipy.optimize.minimize(theobjAbs, B, method='BFGS', options=d)
  else:
    optMethod = annealToUse
    if coolrate == 3:
      optMethod = 'BFGS'
    theresult = scipy.optimize.minimize(theobj, B, method=optMethod, options=d) # Anneal
 
  # remove for machine 1 theresult = scipy.optimize.OptimizeResult()
  # theresult.x = copy.deepcopy(B)

#  d['dwell'] = 100

#  print "Anneal 2"  
#  theresult = scipy.optimize.minimize(theobj, B, method='Anneal', options=d)

  #theresult = scipy.optimize.minimize(theobj, B, method='Nelder-Mead')
 
  subtable = [];
  y = [];
 
  for i in range(forecastrow + 1, forecastrow + 1 + windowsize):
    
    row = [1] # for the constant

    for j in range(3,ncolsrt):
      row = row + [regtable[i][j]]
      #print "len reg row = ", len(row)

    yrow = regtable[i][SPYCO] 

    subtable.append(row);
    y.append(yrow);

  # poor man's bootstrap here on subtable and y
  booty = copy.deepcopy(y);
  bootsubtable = copy.deepcopy(subtable);

  numpy.random.seed() # make sure each run is different for bootstrapping

  # dont do the bootstrap resample if the noboot flag is set to 1
  if noboot == 0:
    for iboot in range(0,len(subtable)):
      therand = random.randint(0, len(subtable)-1); # with replacement so can choose the same again
      subtable[iboot] = bootsubtable[therand];
      y[iboot] = booty[therand];

  solved = numpy.linalg.lstsq(subtable, y)

  print("linear solved:")
  print(solved[0])

  from sklearn.linear_model import LinearRegression 

  newlinmodel = LinearRegression(fit_intercept=True).fit(subtable, y)
  linR2 = newlinmodel.score(subtable,y)
  print("new linear score  R2 =", linR2)
  nparams = numpy.count_nonzero(newlinmodel.coef_)
  print("n = ", len(y), "p = ", nparams)
  linAdjR2Now = adjR2(linR2, len(y), nparams)
  print("new linear Adjust R2 =", linAdjR2Now)  # has a lead zero wot?
  linearAdjR2List.append(linAdjR2Now)

  print("new linear coef", newlinmodel.coef_)
 
  if False: 
    # bootstrap linear model for p-values

    bootR2List = []
    bootCoeffList = []
    bootInterceptList = []

    for iLinBoot in range(0,100):
      linbootSamples = numpy.random.randint(low=0, high=len(y), size=len(y))

      #print "linbootSamples", linbootSamples

      randomizedY = []
      randomizedSubtable = []

      for iReorder in range(0,len(linbootSamples)):
        iSample = linbootSamples[iReorder]
        randomizedY.append(y[iSample])
        #print "subtable[iSample]", subtable[iSample]
        randomizedSubtable.append(subtable[iSample])

      newlinmodel = LinearRegression(fit_intercept=True).fit(randomizedSubtable, randomizedY)
      linR2boot = newlinmodel.score(randomizedSubtable, randomizedY) 
      #print "linboot R2 =", linR2boot
      bootR2List.append(linR2boot)
      bootCoeffList.append(newlinmodel.coef_) # coef_ is itself a list
      bootInterceptList.append(newlinmodel.intercept_)

    meanBootR2 = numpy.mean(bootR2List)       
    meanBootCoeff = numpy.mean(bootCoeffList, axis=0) # mean of columns
    meanBootIntercept = numpy.mean(bootInterceptList)

    stdBootR2 = numpy.std(bootR2List)
    stdBootCoeff = numpy.std(bootCoeffList, axis=0) # std of columns
    stdBootIntercept = numpy.std(bootInterceptList)

    print("len(stdBootCoeff)", len(stdBootCoeff))
    print("stdBootCoeff", stdBootCoeff)

    tValuesCoeff = numpy.divide(meanBootCoeff, stdBootCoeff)
    tValuesIntercept = numpy.divide(meanBootIntercept, stdBootIntercept)

    print("subtable.shape", mydim(subtable))
  
    stShape = mydim(subtable)

    subtableRefit = copy.deepcopy(subtable)

    # maybe better now
    print("tValuesCoeff =", tValuesCoeff)
    print("tValuesIntercept =", tValuesIntercept)

    print("max abs tValue =", max(numpy.absolute(tValuesCoeff)))
 
    # zero out any columns of subtableRefit that have low t scores

    # keep the constant for now

    numNonZeroLinParams = 0

    tValCutoff = 1

    for icol in range(0, stShape[1]):
     shouldZero = abs(tValuesCoeff[icol]) < tValCutoff
     if not shouldZero:
       numNonZeroLinParams += 1

     for irow in range(0, stShape[0]):
        if shouldZero:
          subtableRefit[irow][icol] = 0

    print("numNonZeroLinParams", numNonZeroLinParams)        

    newlinmodelRefit = LinearRegression(fit_intercept=True).fit(subtableRefit, y)

    refitR2 = newlinmodelRefit.score(subtableRefit, y)

    print("refitR2", refitR2)

    refitAdjR2 = adjR2(refitR2, len(y), numNonZeroLinParams)
    print("refit Adjust R2 =", refitAdjR2) 

    print("refit coef_", newlinmodelRefit.coef_)
 
    #testList = [[0,5], [1,3]]
    #print "testList Mean =", numpy.mean(testList, axis=1)

  # end of experimental bootstrap for p-values (scores)
    
  if (forecastrow == -1):
    tmplist = []
    for xi in solved[0]: 
      tmplist.append(xi)

    jout["full_linear_model"] = tmplist

  corr2 = 0

  if forecastrow >= 0:
    corr2, fittedPred = forecast1(solved[0], forecastrow) 
    #print "fittedPred", fittedPred # 1 value
  else:
    print("evaluating last known point not future")
    # hmm...need to pull from the futureXfull data
    corr2, fittedPred = forecast1(solved[0], -1) # if -1 pull X data from futureXrow
    jout["forecast_ahead2"] = fittedPred

  backtestForecastM2.append(fittedPred)
 
  print("calling new function trial") 
 
  # tune alpha first for LassoLars
  bestAlpha = 0.2
  bestAdjR2Lars = -1 

  if uselogit == 0 and uselars:

     alphaTrial = 0.0

     # slow
     while False and alphaTrial < 1:

       clfLars = linear_model.LassoLars(alpha=alphaTrial, fit_intercept=True, max_iter=100000, verbose=True)

       clfLars.fit(subtable, y)

       larsR2 = clfLars.score(subtable,y)

       numNonZero = numpy.count_nonzero(clfLars.coef_)

       #print "numNonZero", numNonZero

       #print "score  (R2 of fit some Lars)", larsR2

       #print "n = ", len(y), "p = ", numNonZero

       AdjR2Lars = adjR2(larsR2, len(y), numNonZero)
 
       #print "Adjust (R2 of fit some Lars tuning...)", AdjR2Lars

       if AdjR2Lars > bestAdjR2Lars:
         bestAdjR2Lars = AdjR2Lars 
         bestAlpha = alphaTrial 
         #print "bestAdjR2Lars = ", bestAdjR2Lars, "bestAlpha = ", bestAlpha

       alphaTrial = alphaTrial + 0.015

     # print "found bestAdjR2Lars = ", bestAdjR2Lars, "at bestAlpha = ", bestAlpha
 
  #clfLars = linear_model.Lars(n_nonzero_coefs=8, fit_intercept=True)

  ### cmt out nov 3 2014  
  # this was prod aug 3 2022 no bic

  theCriterion = 'aic'
  if lassolarsbic == 1:
    theCriterion = 'bic'

  clfLars = linear_model.LassoLarsIC(verbose=True, fit_intercept=True, criterion=theCriterion)

  #clf = linear_model.Perceptron(n_iter=10) 
  #clfLars = linear_model.LarsCV(verbose=True,cv=20) does not run w/ cv arg or not

  tic1 = 0 # time.clock()

  # testing alpha based
  #clfLars = linear_model.LassoLars(alpha=bestAlpha, fit_intercept=True, max_iter=100000, verbose=True) 

  tic2 = 0 # time.clock()

  print("LassoLars time create", (tic2-tic1))

  #clfLars = linear_model.LassoCV(fit_intercept=True, max_iter=10000, verbose=True,cv=cross_validation.LeaveOneOut(10, indices=True))

  if uselogit == 0:

      from sklearn.neighbors import KNeighborsRegressor
      clf = KNeighborsRegressor(n_neighbors=neighbors, weights=distfunc) # use default distfunc for trial , weights=distfunc)

      if uselars:
        clf = clfLars

      tic3 = 0 # time.clock()
      clf.fit(subtable, y)
      tic4 = 0 # time.clock()

      print("LassoLars time fit", (tic4-tic3))

      if uselars:
        print("LARS model, subset of linear variables")
        print((clf.intercept_))
        print((clf.coef_))

        numNonZero = numpy.count_nonzero(clf.coef_)

        if (clf.intercept_ != 0.0):
          numNonZero += 1

        nonZeroCoeffsPerBacktestM3.append(numNonZero)

        #print "numNonZero Lars coefficients", numNonZero

        larsR2 = clf.score(subtable,y)

        print("score  (R2 of fit some Lars)", larsR2)

        print("n = ", len(y), "p = ", numNonZero)

        larsAdjR2Now = adjR2(larsR2, len(y), numNonZero)

        print("Adjust (R2 of fit current Lars)", adjR2(larsR2, len(y), numNonZero)) 

        #if (numNonZero > 1):
        larsAdjR2List.append(larsAdjR2Now)
 
        if (forecastrow == -1):
          tmplist = [clf.intercept_]
          thefirst = True 
          for xi in clf.coef_: 
            if not thefirst:  # always skip the 0 resulting from the constant column which was for full linear
              tmplist.append(xi)  
            thefirst = False 

          jout["lars_model"] = tmplist

      else:
        print(clf)
 
  else:
    #clf = linear_model.LogisticRegression()
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(n_neighbors=neighbors, weights=distfunc) #, weights=distfunc) # now mquad f'real?
 
    print("knclassifier")

    # for regression studies as part of classifier deployment
    from sklearn.neighbors import KNeighborsRegressor
    clfKNR = KNeighborsRegressor(n_neighbors=neighbors, weights=distfunc) # use default distfunc for trial , weights=distfunc)
    
    #print dir(clf)

    #print (clf.intercept_)
    #print (clf.coef_)
    #clf = linear_model.RandomizedLogisticRegression()
    #from sklearn.linear_model import Perceptron
    #clf = Perceptron(verbose=3)

  if uselogit == 1:
    ybinary = [];

    for ki in range(0,len(y)):
      if (y[ki] <= 0):
        ybinary.append(0)
      else:
        ybinary.append(1)

    print("subtable.shape", mydim(subtable))
    #print "subtable", subtable

    clf.fit(subtable, ybinary)

    # experimental
    if False:
      # check regression metrics only

      # 2nd is # vars
      theShape =  mydim(subtable)

      clfKNR.fit(subtable, y)

      knrR2 = clfKNR.score(subtable,y)

      knrAdjR2Now = adjR2(knrR2, len(y), theShape[1])

      print("full var count knrAdjR2Now =", knrAdjR2Now)
    
    # test 
    # ybinaryPredict = clf.predict(subtable)

    # only for future versions
    #print clf.get_params(True)

    startnvars = len(subtable[0])

    print("startnvars = ", startnvars)

    # figure column max min

    # across the columns
    max_subtable = [-99999]*startnvars
    min_subtable = [99999]*startnvars

    for k in range(0, len(subtable)): # down the rows
      for kcol in range(0, startnvars): # across the cols
        if subtable[k][kcol] > max_subtable[kcol]:
          max_subtable[kcol] = subtable[k][kcol]
        if subtable[k][kcol] < min_subtable[kcol]:
          min_subtable[kcol] = subtable[k][kcol]       
     
    print("max of columns of subtable = ", max_subtable)
    print("min of columns of subtable = ", min_subtable)

    normfactor_subtable = [1]*startnvars

    for kcol in range(0, startnvars):
      normfactor_subtable[kcol] = max_subtable[kcol] - min_subtable[kcol]
      if (normfactor_subtable[kcol] < 0.001):
        print("normalizer too small for column ", kcol, " use 1 instead")
        normfactor_subtable[kcol] = 1; 

    nonnorm_subtable = copy.deepcopy(subtable) # save for later in case we need it

    # normalize the cols, probably a faster way but get it done clearly now

    if normalize == 1:

      print("normalizing columns of subtable")

      for k in range(0, len(subtable)): # down the rows
       for kcol in range(0, startnvars): # across the cols        
         subtable[k][kcol] = subtable[k][kcol] / normfactor_subtable[kcol]

      didnormalize = 1
      # note:  we also need to normalize the data we send to the fwd prediction
    else:
      didnormalize = 0
      print("do not normalize columns of subtable, orig way")

    basesubtable = copy.deepcopy(subtable)

    yhat = clf.predict(subtable)
    correct = 0
    for k in range(0, len(ybinary)):
      if (ybinary[k] == yhat[k]):
        correct += 1 
    print("fit correct logit % = ", float(correct) / len(yhat))    

    origcorrect = float(correct) / len(yhat) 

    # now 1 by 1 thru the variables and zero them out...do a fit and check on each

    print("start variable sensitivity check")

    deltafitwithvarremoved = [None]*startnvars

    for removei in range(0, startnvars):
      subtablecheck = copy.deepcopy(basesubtable)
      for rowiter in range(0, len(subtablecheck)):
        subtablecheck[rowiter][removei] = 0
        # pull a random value from this column instead of zero
        if scramblesens == 1:
          therandfill = random.randint(0, len(basesubtable)-1);
          therandval = basesubtable[therandfill][removei]
          subtablecheck[rowiter][removei] = therandval

      yhatcheck = clf.predict(subtablecheck) 
      correct = 0
      for k in range(0, len(ybinary)):
         if (ybinary[k] == yhatcheck[k]):
           correct += 1 
      theScore = clf.score(subtablecheck, ybinary)
      print("remove var ", removei, " fit correct logit % = ", float(correct) / len(yhatcheck), " score =", theScore) 
      deltafitwithvarremoved[removei] = origcorrect - float(correct) / len(yhatcheck)

    # compute range

    rangemin = 99999
    rangemax = -99999

    for removei in range (0, startnvars):
      if deltafitwithvarremoved[removei] < rangemin:
        rangemin = deltafitwithvarremoved[removei]
      if deltafitwithvarremoved[removei] > rangemax:
        rangemax = deltafitwithvarremoved[removei]

    totalrange = rangemax - rangemin

    print("totalrange = ", totalrange)

    percentrange = [None]*startnvars

    variablemask = [1]*startnvars # assume we start with all variables

    # fix div by zero error if all variables show up equal...may be possible in normalized case?
    if (totalrange > 0):
      for removei in range (0, startnvars):      
        percentrange[removei] = (deltafitwithvarremoved[removei]-rangemin)/totalrange
        print("percentrange for variable ", removei, " = ", percentrange[removei])
        if percentrange[removei] < (knnvarcutoff / 1000.0) :
          print("removing above variable, below cutoff tolerance")
          variablemask[removei] = 0 # mask variables with low contribution

      meanpercentrange = numpy.mean(percentrange)
      stdpercentrange = numpy.std(percentrange)

      print("mean std percentrange = ", meanpercentrange, stdpercentrange)

      print("mean - std = ", meanpercentrange - stdpercentrange)

      # remove according to range statistics

      if dyncutoff == 1:

        print("adjusting removals dyanmically, mean + stdev is cutoff")
  
        variablemask = [1]*startnvars # assume we start with all variables
 
        for removei in range (0, startnvars):
          percentrange[removei] = (deltafitwithvarremoved[removei]-rangemin)/totalrange
          print("percentrange for variable ", removei, " = ", percentrange[removei])
          if percentrange[removei] < meanpercentrange - 1*stdpercentrange :
            print("removing above variable, below dynamic cutoff tolerance of ",  meanpercentrange -1*stdpercentrange)
            variablemask[removei] = 0 # mask variables with low contribution 
 
    print("done with variable sensitivity check")

    # now we refit without the masked variables

    subtablecheck = copy.deepcopy(basesubtable)
 
    for removei in range(0, startnvars):
      subtablecheck = copy.deepcopy(basesubtable)
      for rowiter in range(0, len(subtablecheck)):
        if (variablemask[removei] == 0):
          subtablecheck[rowiter][removei] = 0

    # hmm not a true refit just zeroing

    #clf.fit(subtablecheck, ybinary) ## TEST AUG 9 2022

    yhat = clf.predict(subtablecheck) # refit with subset of variables
    correct = 0
    for k in range(0, len(ybinary)):
      if (ybinary[k] == yhat[k]):
        correct += 1
    print("for subset of ", sum(variablemask), ", fit correct logit % = ", float(correct) / len(yhat))

    nonZeroCoeffsPerBacktestM3.append(sum(variablemask))

    if False:
      # checking regression
      clfKNR.fit(subtablecheck,y)
      yhatKNR = clfKNR.predict(subtablecheck)
 
      knrR2KNRsubvar = clfKNR.score(subtablecheck,y)

      knrAdjR2subvar = adjR2(knrR2KNRsubvar, len(y), sum(variablemask))

      print("knrAdjR2subvar = ", knrAdjR2subvar, " full var AdjR2 =", knrR2KNRsubvar)
 
  #print "Neighbors const & coeffs"
  #print(clf.intercept_)
  #print(clf.coef_)

  # wow this is wrong:   predrow = subtable[forecastrow]

  if (forecastrow >= 0):
    predrow = [1] + regtable[forecastrow][3:]
  else:
    print("no backtest, using futureXfull row") 
    #predrow = [1] + regtable[0][3:]  # first row by default
    predrow = [1] + futureXfull[3:]  # first row by default

  #print "lars predrow = ", predrow

  print("model 3:")
  print(clf)

  # need to now mask variables on the predicted row

  print("ready to remove variables on predicted row")

  # if we had run as LarsIC, startnvars is not defined so ignore the variable removal process...LARS does it by itself 

  if ('startnvars' in vars() or 'startnvars' in globals()):
    print("removing low contributor variables")
    for removei in range(0, startnvars):
        if (variablemask[removei] == 0):
          predrow[removei] = 0

  if 'didnormalize' in vars(): 
    if didnormalize == 1:
      # normalize the predicted row too
      print("normalizing the predicted row")
      for kcol in range(0, startnvars): # across the cols        
         predrow[kcol] = predrow[kcol] / normfactor_subtable[kcol]

  larspred = clf.predict(np.array(predrow).reshape(1,-1))
  # still broken in 0.11 scikit-learn
  # fixed in 0.12.1 
  if callable(getattr(clf,'predict_proba', None)):
    larspred_proba = clf.predict_proba(np.array(predrow).reshape(1,-1))

    # need to do this if this is a list or array

    print("type of larspred_proba = ", type(larspred_proba))

    print("really knn larspred_proba = ", larspred_proba)
  else:
    print("skipping predict_proba, does not exist")
    larspred_proba = None

  # do i need to check len?
  if larspred_proba is not None:
    larspred_prob1_scalar = larspred_proba[0][1]
    # map 0 to 1 ... to -1 to 1, zero centered
    larspred_prob1_scalar = larspred_prob1_scalar * 2 - 1
  else:
    larspred_prob1_scalar = 0 # change to zero centered

  print("larspred_prob1_scalar", larspred_prob1_scalar)

  # back to scalar

  if isinstance(larspred, numpy.ndarray):
    larspred = larspred[0]

  #if larspred_prob1_scalar is not None and larspred_prob1_scalar is not 0:
  if larspred_prob1_scalar is not None and larspred_prob1_scalar != 0:
    larspred = larspred_prob1_scalar

#  if isinstance(larspred_proba, numpy.ndarray):
#    larspred_proba = larspred_proba[0]

#  print "larspred_proba", larspred_proba

  # map 0 back to negative for logistic

  if uselogit == 1:
    if (larspred <= 0 and (larspred_prob1_scalar is None)):
      larspred = -1
 
  if (forecastrow >= 0): 
    print("custom ap = ", regtable[forecastrow][SPYCO], " ", larspred) 
  else:
    print("custom ap = ", futureXfull[SPYCO], " ", larspred)

  backtestActual.append(regtable[forecastrow][SPYCO])
  backtestForecastM3.append(larspred)

  jout["forecast_ahead3"] = larspred
 
  savefcr = forecastrow

  if (forecastrow<0):
    forecastrow = 0
 
  if ((larspred > 0) and (regtable[forecastrow][SPYCO] > 0)) or ((larspred < 0) and (regtable[forecastrow][SPYCO] < 0)):
      larscorrect = 1
      #print "GOOD larspred", "forecastrow", forecastrow
  else:
      larscorrect = 0
      #print "BAD larspred", "forecastrow", forecastrow

  forecastrow = savefcr

  print("annealing or min abs solved:") 
  print(theresult)

  if (forecastrow == -1):
    tmplist = []
    for xi in theresult.x:
      tmplist.append(xi) 

    jout["annealing_model"] = tmplist 

  # note:  only 1 of these is relevant (check which function was used theobj or theobjAbs)

  # this is the fit metric not individual points evaluated

  # not doing anything yet w/ these

  out = -theobj(theresult.x)
  print("final re-evaluate theobj = ", out) 

  if model1minabs == 1:
    out = theobjAbs(theresult.x)
    print("final re-evaluate theobjAbs = ", out) 

  outFullVar = out

  # sensitivity check for each variable for model 1
  # orig way, set each variable to 0 in turn and
  # see how much the result changes 

  if knnvarcutoff > 0:  
    theResultXSave = copy.deepcopy(theresult.x) # may not need

    outForThisVarList = []
  
    for varnum in range(0, len(theResultXSave)):
      theResultXTest = copy.deepcopy(theresult.x) 
      theResultXTest[varnum] = 0
    
      if model1minabs == 0:
        outForThisVar = -theobj(theResultXTest)
        print("per var re-evaluate theobj = ", outForThisVar)

      if model1minabs == 1:
        outForThisVar = theobjAbs(theResultXTest)
        print("per var re-evaluate theobjAbs = ", outForThisVar)

      outForThisVarList.append(float(outForThisVar))
  
    # the variables when zeroed that cause the most change in the metric are the most important (hypothesis)
    
    print("outForThisVarList", outForThisVarList)

    maxOfList = max(outForThisVarList)
    minOfList = min(outForThisVarList)

    if outFullVar < minOfList:
      minOfList = outFullVar
    
    if outFullVar > maxOfList:
      maxOfList = outFullVar

    rangeOfList = maxOfList - minOfList

    print("rangeOfList including no vars removed", rangeOfList)

    varMask = []

    for varnum in range(0, len(outForThisVarList)):
      print("var ", varnum, "numerator", (outForThisVarList[varnum] - minOfList))
      percentChange = (outForThisVarList[varnum] - minOfList)/rangeOfList
      print("model 1 var", varnum, " percentChange", percentChange)
      localCutoff = knnvarcutoff / 1000.0
      print("localCutoff", localCutoff) 
      # if all variables are less than localcutoff; bug (div zero in Anneal)
      if percentChange < localCutoff:
        varMask.append(0)
      else:
        varMask.append(1)    

    # varMask is same order as variables in A and theresult.x

    # now re-solve model w/ only the most significant variables

    #def f(a,b,c):
    #    print a,b,c
    #bound_f = partial(f,1)
    #bound_f(2,3)

    varCount = numpy.count_nonzero(varMask)    

    # no variables saved yikes put them all at 1 for now
    if varCount == 0:
      print("caution, varCount is 0, go back to all on for now")
      for varnum in range(0, len(outForThisVarList)): 
        varMask[varnum] = 1
    
    nonZeroCoeffsPerBacktestM1.append(varCount)

    boundTheObjAbs = partial(theobjAbsPreMask, varMask)
    boundTheObj = partial(theobjPreMask, varMask)

    # only coded for min abs to start

    if model1minabs == 1:
      theresult = scipy.optimize.minimize(boundTheObjAbs, B, method='BFGS', options=d)
    else:
      optMethod = annealToUse
      if coolrate == 3:
        optMethod = 'BFGS'
      theresult = scipy.optimize.minimize(boundTheObj, B, method=optMethod, options=d) # BFGS

    theresultXMasked = numpy.multiply(theresult.x, varMask)

    theresult.x = theresultXMasked

    # end first trial easy way of var sensitivity
 
  if (forecastrow >= 0):
    corr, fittedPred = forecast1(theresult.x, forecastrow) # was theresult.x
  else:
    print("no backtest:  forecast with futureXfull row")
    corr, fittedPred = forecast1(theresult.x,-1)          # was theresult.x
    jout["forecast_ahead1"] = fittedPred

  backtestForecastM1.append(fittedPred)

  forecastcorrect = forecastcorrect + corr
  forecastcorrect2 = forecastcorrect2 + corr2
  forecastcorrect3 = forecastcorrect3 + larscorrect
  print("corr1 corr2 corr3 ", corr, corr2, larscorrect) 

  # collect the bad run sequences
  # need to do one at the end of this after backtest loop

  if corr == 0:
    badrun += 1
    #badrunstair.append(badrun)
  else:  # a good run...the bad sequence is broken, so save it and reset the count
    badrunsequence.append(badrun)
    #badrunstair.append(badrun)
    badrun = 0

  if corr2 == 0:
    badrun2 += 1
    #badrunstair2.append(badrun2)
  else:
    badrunsequence2.append(badrun2)
    #badrunstair2.append(badrun2)
    badrun2 = 0

  if larscorrect == 0: # corr3 essentially
    badrun3 += 1
    #badrunstair3.append(badrun3)
  else:
    badrunsequence3.append(badrun3)
    #badrunstair3.append(badrun3)
    badrun3 = 0

  print("ready to append badrun = ", badrun, "to stair")
  print("ready to append badrun2 = ", badrun2, "to stair")
  print("ready to append badrun3 = ", badrun3, "to stair")

  badrunstair.append(badrun)
  badrunstair2.append(badrun2)
  badrunstair3.append(badrun3)

  # init for print purposes
  brsNoZero = [0];
  brsNoZero2 = [0];
  brsNoZero3 = [0]; 

  # final countup
  if forecastrow == ntrials:
    print("final countup of bad runs")
    badrunsequence.append(badrun) 
    badrunsequence2.append(badrun2)
    badrunsequence3.append(badrun3)
    print("bad run sequences:")
    print(badrunsequence) 
    print(badrunsequence2)
    print(badrunsequence3)

    brsNoZero = copy.copy(badrunsequence)
    brsNoZero2 = copy.copy(badrunsequence2)
    brsNoZero3 = copy.copy(badrunsequence3)

    while brsNoZero.count(0) > 0:
      brsNoZero.remove(0)
 
    while brsNoZero2.count(0) > 0:
      brsNoZero2.remove(0)

    while brsNoZero3.count(0) > 0:
      brsNoZero3.remove(0)
   
    print("bad run stats: avg max avgNoZero")
    print(numpy.mean(badrunsequence), " ", numpy.max(badrunsequence), " ", numpy.mean(brsNoZero))
    print(numpy.mean(badrunsequence2), " ", numpy.max(badrunsequence2), " ", numpy.mean(brsNoZero2))
    print(numpy.mean(badrunsequence3), " ", numpy.max(badrunsequence3), " ", numpy.mean(brsNoZero3))

    # plot

    # figure the max y so we can set all Y axes equal

    m1 = numpy.max(badrunstair)
    m2 = numpy.max(badrunstair2)
    m3 = numpy.max(badrunstair3)

    ymax = numpy.max([m1, m2, m3])

    fig = plt.figure()

    #plot(d)
    #bar(range(0,len(badrunstair3)),badrunstair3, color="g")

    print("len stair array = ", len(badrunstair))
    print("len stair array 2 = ", len(badrunstair2))
    print("len stair array 3 = ", len(badrunstair3))

    print("badrunstair = ", badrunstair)
    print("badrunstair2 = ", badrunstair2)
    print("badrunstair3 = ", badrunstair3)

    plt.title("consecutive bad forecasts in backtest")

    subplot(311)

    plt.title("Y axes = cumulative bad forecast days\nduring backtest (lower is better)")

    xlim(0, len(badrunstair))
    ylim(0, ymax)

    plt.ylabel("model 1")
    step(list(range(0,len(badrunstair))),badrunstair, color="r", where="post")

    subplot(312)

    xlim(0, len(badrunstair)) # use same x as 1st
    ylim(0, ymax)

    plt.ylabel("model 2") 
    step(list(range(0,len(badrunstair2))),badrunstair2, color="b", where="post") 

    subplot(313)

    xlim(0, len(badrunstair)) # use same x as 1st
    ylim(0, ymax)

    plt.ylabel("model 3")
    step(list(range(0,len(badrunstair3))),badrunstair3, color="g", where="post")

    plt.xlabel("backtest day:  0 = yesterday, to the right = back in time") 

    # make a multi page doc also...trial

    ### pdf_pages = PdfPages('my_fancy_document.pdf')
 
    ### TEMP REMOVE FOR DEBUG py3.11 
    fig.savefig('backtest_runs.pdf')

    ### pdf_pages.savefig(fig) # save to the multipage file also

    # these plots are not shown to the user

    fig = plt.figure()
    plot(backtestActual, backtestForecastM1, "ro")

    #m1R2 = r2_score(backtestActual, backtestForecastM1)
    #print "backtest model1 R2", m1R2

    fig.savefig('m1.pdf')

    fig = plt.figure()
    plot(backtestActual, backtestForecastM2, "go")

    fig.savefig('m2.pdf')

    fig = plt.figure()
    plot(backtestActual, backtestForecastM3, "bo")

    fig.savefig('m3.pdf')

    # a is array
    def positive(a):
        b = []
        for i in range(0,len(a)):
          if a[i] > 0:
            b.append(True)
          else:
            b.append(False)
        return b

    def negative(a):
        b = []
        for i in range(0,len(a)):
          if a[i] < 0:
            b.append(True)
          else:
            b.append(False)
        return b

    # confusion matrix computations
    # note that left is actual, right is predicted in the coding 
 
    posPos1 = numpy.logical_and(positive(backtestActual), positive(backtestForecastM1))
    negNeg1 = numpy.logical_and(negative(backtestActual), negative(backtestForecastM1))

    posNeg1 =  numpy.logical_and(positive(backtestActual), negative(backtestForecastM1))
    negPos1 =  numpy.logical_and(negative(backtestActual), positive(backtestForecastM1))

    if False:
      print("confusion raw data")
      print("posos1", posPos1)
      print("negNeg1", negNeg1)
      print("posNeg1", posNeg1)
      print("negPos1", negPos1)

    # these outputs are sent to the json output later in the run jout

    print("confusion matrix m1")
    print("posPos1 = ", numpy.count_nonzero(posPos1))
    print("negNeg1 = ", numpy.count_nonzero(negNeg1)) 
    print("posNeg1 = ", numpy.count_nonzero(posNeg1))
    print("negPos1 = ", numpy.count_nonzero(negPos1))

    posPos2 = numpy.logical_and(positive(backtestActual), positive(backtestForecastM2))
    negNeg2 = numpy.logical_and(negative(backtestActual), negative(backtestForecastM2))

    posNeg2 =  numpy.logical_and(positive(backtestActual), negative(backtestForecastM2))
    negPos2 =  numpy.logical_and(negative(backtestActual), positive(backtestForecastM2))

    print("confusion matrix m2")
    print("posPos2 = ", numpy.count_nonzero(posPos2))
    print("negNeg2 = ", numpy.count_nonzero(negNeg2)) 
    print("posNeg2 = ", numpy.count_nonzero(posNeg2))
    print("negPos2 = ", numpy.count_nonzero(negPos2))

    posPos3 = numpy.logical_and(positive(backtestActual), positive(backtestForecastM3))
    negNeg3 = numpy.logical_and(negative(backtestActual), negative(backtestForecastM3))

    posNeg3 =  numpy.logical_and(positive(backtestActual), negative(backtestForecastM3))
    negPos3 =  numpy.logical_and(negative(backtestActual), positive(backtestForecastM3))

    print("confusion matrix m3")
    print("posPos3 = ", numpy.count_nonzero(posPos3))
    print("negNeg3 = ", numpy.count_nonzero(negNeg3))
    print("posNeg3 = ", numpy.count_nonzero(posNeg3))
    print("negPos3 = ", numpy.count_nonzero(negPos3))

    # loop over zval tolerances

    bestM1pval = 1
    bestM2pval = 1
    bestM3pval = 1

    bestM1PercentTol = 0
    bestM2PercentTol = 0
    bestM3PercentTol = 0

    # these are not actually the best, these are the current
    currentCorrectNumeratorM1 = 0
    currentCorrectDenominatorM1 = 0

    currentCorrectNumeratorM2 = 0
    currentCorrectDenominatorM2 = 0

    currentCorrectNumeratorM3 = 0
    currentCorrectDenominatorM3 = 0

    # these are the best
    bestCorrectNumeratorM1 = 0
    bestCorrectDenominatorM1 = 0

    bestCorrectNumeratorM2 = 0
    bestCorrectDenominatorM2 = 0

    bestCorrectNumeratorM3 = 0
    bestCorrectDenominatorM3 = 0

    # arrays
    BESTbacktestForecastM1outsideOfZTol = []
    BESTbacktestForecastM2outsideOfZTol = [] 
    BESTbacktestForecastM3outsideOfZTol = [] 
 
    forecastRangeM1 = numpy.max(backtestForecastM1) - numpy.min(backtestForecastM1)
    forecastRangeM2 = numpy.max(backtestForecastM2) - numpy.min(backtestForecastM2)
    forecastRangeM3 = numpy.max(backtestForecastM3) - numpy.min(backtestForecastM3)

    print("forecastRangeM1 2 3 = ", forecastRangeM1, forecastRangeM2, forecastRangeM3)

    # dont do this for a forecast

    if ntrials > -1: # backtest
     # note only 1 indent
     for percentTol in range(0,60):

      print("--- new percentTol trial: ---")

      # loc 1 first def of m1ZTol for computing it
      m1ZTol = forecastRangeM1*float(percentTol)*tolIndexToTol
      m2ZTol = forecastRangeM2*float(percentTol)*tolIndexToTol
      m3ZTol = forecastRangeM3*float(percentTol)*tolIndexToTol

      print("forecastRangeM1 2 3 = ", forecastRangeM1, forecastRangeM2, forecastRangeM3)
    
      backtestForecastM1outsideOfZTol = numpy.abs(backtestForecastM1) > m1ZTol
      backtestForecastM2outsideOfZTol = numpy.abs(backtestForecastM2) > m2ZTol
      backtestForecastM3outsideOfZTol = numpy.abs(backtestForecastM3) > m3ZTol

      print("backtestForecastM1outsideOfZTol", backtestForecastM1outsideOfZTol) 
      print("backtestForecastM2outsideOfZTol", backtestForecastM2outsideOfZTol)
      print("backtestForecastM3outsideOfZTol", backtestForecastM3outsideOfZTol)

      print("backtestForecastM1 = ", backtestForecastM1)
      print("backtestForecastM2 = ", backtestForecastM2)
      print("backtestForecastM3 = ", backtestForecastM3)

      print("backtestActual = ", backtestActual)

      sameSignM1 = numpy.sign(backtestActual) == numpy.sign(backtestForecastM1)
      sameSignM2 = numpy.sign(backtestActual) == numpy.sign(backtestForecastM2)
      sameSignM3 = numpy.sign(backtestActual) == numpy.sign(backtestForecastM3)

      print("sameSignM1" , sameSignM1)
      print("sameSignM2" , sameSignM2)
      print("sameSignM3" , sameSignM3)
 
      sameSignAndOutsideM1 = numpy.logical_and(sameSignM1, backtestForecastM1outsideOfZTol)
      sameSignAndOutsideM2 = numpy.logical_and(sameSignM2, backtestForecastM2outsideOfZTol)
      sameSignAndOutsideM3 = numpy.logical_and(sameSignM3, backtestForecastM3outsideOfZTol)

      print("sameSignAndOutsideM1", sameSignAndOutsideM1)
      print("sameSignAndOutsideM2", sameSignAndOutsideM2)
      print("sameSignAndOutsideM3", sameSignAndOutsideM3)

      countSuccessM1 = numpy.count_nonzero(sameSignAndOutsideM1)
      countSuccessM2 = numpy.count_nonzero(sameSignAndOutsideM2)
      countSuccessM3 = numpy.count_nonzero(sameSignAndOutsideM3)

      countOutsideM1 = numpy.count_nonzero(backtestForecastM1outsideOfZTol)
      countOutsideM2 = numpy.count_nonzero(backtestForecastM2outsideOfZTol)
      countOutsideM3 = numpy.count_nonzero(backtestForecastM3outsideOfZTol) 

      print("countSuccessM1", countSuccessM1, "out of", countOutsideM1)
      print("countSuccessM2", countSuccessM2, "out of", countOutsideM2)
      print("countSuccessM3", countSuccessM3, "out of", countOutsideM3)

      # maybe assign the current* ones even if zero

      if countOutsideM1 > 0:
        percCorrectOutsideZM1 = float(countSuccessM1)/countOutsideM1
        currentCorrectNumeratorM1 = countSuccessM1
        currentCorrectDenominatorM1 = countOutsideM1
      else:
        percCorrectOutsideZM1 = 0 # just init not sure what to do
    
      if countOutsideM2 > 0:
        percCorrectOutsideZM2 = float(countSuccessM2)/countOutsideM2
        currentCorrectNumeratorM2 = countSuccessM2
        currentCorrectDenominatorM2 = countOutsideM2
      else:
        percCorrectOutsideZM2 = 0

      if countOutsideM3 > 0:
        percCorrectOutsideZM3 = float(countSuccessM3)/countOutsideM3
        currentCorrectNumeratorM3 = countSuccessM3
        currentCorrectDenominatorM3 = countOutsideM3
      else:
        percCorrectOutsideZM3 = 0

      print("percent correct outside Z tol m1 m2 m3 = ", percCorrectOutsideZM1, percCorrectOutsideZM2, percCorrectOutsideZM3, "at m[123]ZTol", m1ZTol, m2ZTol, m3ZTol)

      prob1tol = 1- prob(countSuccessM1-1, 0.5, countOutsideM1)
      prob2tol = 1- prob(countSuccessM2-1, 0.5, countOutsideM2)
      prob3tol = 1- prob(countSuccessM3-1, 0.5, countOutsideM3)

      print("tolerances ", m1ZTol, m2ZTol, m3ZTol)

      print("pval with Z tol m1 m2 m3 = ", prob1tol, prob2tol, prob3tol, "at m[123]ZTol", m1ZTol, m2ZTol, m3ZTol)

      # best pval orig trial
      if True:
        if (prob1tol < bestM1pval):
          bestM1PercentTol = percentTol
          bestM1pval = prob1tol
          bestCorrectNumeratorM1 = currentCorrectNumeratorM1
          bestCorrectDenominatorM1 = currentCorrectDenominatorM1
          BESTbacktestForecastM1outsideOfZTol = backtestForecastM1outsideOfZTol
        if (prob2tol < bestM2pval):
          bestM2PercentTol = percentTol 
          bestM2pval = prob2tol
          bestCorrectNumeratorM2 = currentCorrectNumeratorM2
          bestCorrectDenominatorM2 = currentCorrectDenominatorM2
          BESTbacktestForecastM2outsideOfZTol = backtestForecastM2outsideOfZTol
        if (prob3tol < bestM3pval):
          bestM3PercentTol = percentTol
          bestM3pval = prob3tol
          bestCorrectNumeratorM3 = currentCorrectNumeratorM3
          bestCorrectDenominatorM3 = currentCorrectDenominatorM3
          BESTbacktestForecastM3outsideOfZTol = backtestForecastM3outsideOfZTol
      else:
          doi = 1

    # end of ztol determination loop

    # this is after the ztol determination loop

    if ntrials > -1:
      # recalc from loc 1 (make function?)
      m1ZTol = forecastRangeM1*float(bestM1PercentTol)*tolIndexToTol
      m2ZTol = forecastRangeM2*float(bestM2PercentTol)*tolIndexToTol
      m3ZTol = forecastRangeM3*float(bestM3PercentTol)*tolIndexToTol
 
      print("bestM1pval", bestM1pval, "at tol index", bestM1PercentTol, "m1ZTol = ", m1ZTol, "numerator", bestCorrectNumeratorM1, "denom", bestCorrectDenominatorM1)
      print("bestM2pval", bestM2pval, "at tol index", bestM2PercentTol, "m2ZTol = ", m2ZTol, "numerator", bestCorrectNumeratorM2, "denom", bestCorrectDenominatorM2)
      print("bestM3pval", bestM3pval, "at tol index", bestM3PercentTol, "m3ZTol = ", m3ZTol, "numerator", bestCorrectNumeratorM3, "denom", bestCorrectDenominatorM3)

      # --- here we can compute the stairstep bad run plots based on the found ztols

      print("recompute stairstep bad run plots for outside ztol")

      # to start, use same code as in the ztol determination loop for the best ztols

      print("forecastRangeM1 2 3 = ", forecastRangeM1, forecastRangeM2, forecastRangeM3)

      backtestForecastM1outsideOfZTol = numpy.abs(backtestForecastM1) > m1ZTol
      backtestForecastM2outsideOfZTol = numpy.abs(backtestForecastM2) > m2ZTol
      backtestForecastM3outsideOfZTol = numpy.abs(backtestForecastM3) > m3ZTol

      print("backtestForecastM1outsideOfZTol", backtestForecastM1outsideOfZTol)
      print("backtestForecastM2outsideOfZTol", backtestForecastM2outsideOfZTol)
      print("backtestForecastM3outsideOfZTol", backtestForecastM3outsideOfZTol)

      print("backtestForecastM1 = ", backtestForecastM1)
      print("backtestForecastM2 = ", backtestForecastM2)
      print("backtestForecastM3 = ", backtestForecastM3)

      print("backtestActual = ", backtestActual)

      sameSignM1 = numpy.sign(backtestActual) == numpy.sign(backtestForecastM1)
      sameSignM2 = numpy.sign(backtestActual) == numpy.sign(backtestForecastM2)
      sameSignM3 = numpy.sign(backtestActual) == numpy.sign(backtestForecastM3)

      print("sameSignM1" , sameSignM1)
      print("sameSignM2" , sameSignM2)
      print("sameSignM3" , sameSignM3)

      sameSignAndOutsideM1 = numpy.logical_and(sameSignM1, backtestForecastM1outsideOfZTol)
      sameSignAndOutsideM2 = numpy.logical_and(sameSignM2, backtestForecastM2outsideOfZTol)
      sameSignAndOutsideM3 = numpy.logical_and(sameSignM3, backtestForecastM3outsideOfZTol)

      print("sameSignAndOutsideM1", sameSignAndOutsideM1)
      print("sameSignAndOutsideM2", sameSignAndOutsideM2)
      print("sameSignAndOutsideM3", sameSignAndOutsideM3)

      # trade system experiment
      # compute returns (start w/ gains / losses)

      # assume 1 share held
      # then gains are in dollars

      buyAndHoldGain = 0
      allForecastsGain1 = 0
      allForecastsGain2 = 0
      allForecastsGain3 = 0
      ztolGain1 = 0
      ztolGain2 = 0
      ztolGain3 = 0 
     
      print("costPerTrade", costPerTrade)

      # also tack on forecasts to regtable rows

      # use regtable with levels (this has the header row)

      regtableWithBacktests = copy.deepcopy(regtableL)
     
      # header row
      # hmm there is no header row on this data table
      headerRow = regtableWithBacktests[0] # pointer to list
      headerRow.append("m1forecast")
      headerRow.append("m1forecastZTol")
      headerRow.append("m2forecast")
      headerRow.append("m2forecastZTol")
      headerRow.append("m3forecast")
      headerRow.append("m3forecastZTol")

      oneIndexOlderThanBacktest = len(backtestActual) + 1 # also account for header row

      # to pull the open1 (1 lag)
      oneIndexNewerThanBacktest = len(backtestActual) - 1 

      # this is the target price (level)

      # may fail for 1 backtest day?
 
      print("close before backtest", regtableWithBacktests[oneIndexOlderThanBacktest][0], regtableWithBacktests[oneIndexOlderThanBacktest][2])
      print("close most recent    ", regtableWithBacktests[1][0], regtableWithBacktests[1][2])

      print("open on backtest start day", regtableWithBacktests[oneIndexNewerThanBacktest][0], regtableWithBacktests[oneIndexNewerThanBacktest][5]) 

      backtestStartDay = "one trading day before " + str(regtableWithBacktests[oneIndexNewerThanBacktest][0])

      openOnBacktestStartDay = regtableWithBacktests[oneIndexNewerThanBacktest][5] # this pulls open1 = open lag 1

      initialInvestment = shareCount*openOnBacktestStartDay + costPerTrade # this has to be plus since you have that penalty

      print("initial investment plus one cost per trade", initialInvestment)

      closeBeforeBacktest = regtableWithBacktests[oneIndexOlderThanBacktest][2]
      closeMostRecent = regtableWithBacktests[1][2]

      buyAndHoldActual = shareCount*(closeMostRecent - openOnBacktestStartDay) - 2*costPerTrade

      print("buyAndHoldActual gain", buyAndHoldActual)

      buyAndHoldGainPlot = [[], []]

      ztolGain1Plot = [[], []]
      ztolGain2Plot = [[], []]
      ztolGain3Plot = [[], []]

      # backtests go backward in time (recent to oldest) 
      for reverseIdx in range(0, len(backtestActual)):

        idx = len(backtestActual) - 1 - reverseIdx # for plotting and computing in proper order forward

        # make a matrix from the regtable levels matrix that has
        # extra columns for all 6 forecasts (3 models + 3 ztol versions)

        # start with 1 forecast from M1 no ztol
        # dont need to access data row each time, it should be the same
        dataRow = regtableWithBacktests[idx+1] # account for header row
        dataRow.append(backtestForecastM1[idx])

        if (backtestForecastM1outsideOfZTol[idx]):
          dataRow.append(backtestForecastM1[idx])
        else:
          dataRow.append(0) 

        dataRow = regtableWithBacktests[idx+1] # account for header row
        dataRow.append(backtestForecastM2[idx])
        
        if (backtestForecastM2outsideOfZTol[idx]):
          dataRow.append(backtestForecastM2[idx])
        else:
          dataRow.append(0) 

        dataRow = regtableWithBacktests[idx+1] # account for header row
        dataRow.append(backtestForecastM3[idx])

        if (backtestForecastM3outsideOfZTol[idx]):
          dataRow.append(backtestForecastM3[idx])
        else:
          dataRow.append(0)

        # we go back in time so the last one is the orig price
        # since this close is within the backtest, go one beyond for reference price?
        print("apparent target close", regtableWithBacktests[idx+1][0], regtableWithBacktests[idx+1][2])

        # end creation of output backtest table
 
        # here is where trade computation starts

        if (idx == 0 or idx == (len(backtestActual)-1) ):
          buyAndHoldCPT = costPerTrade # 1 trade at start 1 trade at end
        else:
          buyAndHoldCPT = 0 # no trades in between

        buyAndHoldGain += (shareCount*backtestActual[idx] - buyAndHoldCPT)                    

        buyAndHoldGainPlot[0].append(reverseIdx)
        buyAndHoldGainPlot[1].append(buyAndHoldGain)

        # allow shorting use abs 

        # if we do not allow shorting, we only trade on days when forecast is +
        # allowShorting = 1 # do from arg now

        # we should have close-out cost

        # do 2 trades per day 

        # model 1
        if (sameSignM1[idx]):
          model1Gain = shareCount*abs(backtestActual[idx]) - costPerTrade*2
        else:
          model1Gain = -shareCount*abs(backtestActual[idx]) - costPerTrade*2 # loss

        if allowShorting == 1: # always sum up if we allow shorting
          allForecastsGain1 += model1Gain
        elif backtestForecastM1[idx] > 0: # otherwise only sum up if forecast was postive
          allForecastsGain1 += model1Gain
           
        if (backtestForecastM1outsideOfZTol[idx]): # only make a trade if we are outside ztol
          if allowShorting == 1: # always sum up if we allow shorting
            ztolGain1 += model1Gain
          elif backtestForecastM1[idx] > 0:
            ztolGain1 += model1Gain   

        ztolGain1Plot[0].append(reverseIdx)
        ztolGain1Plot[1].append(ztolGain1)

        # model 2
        if (sameSignM2[idx]):
          model2Gain = shareCount*abs(backtestActual[idx]) - costPerTrade*2
        else:
          model2Gain = -shareCount*abs(backtestActual[idx]) - costPerTrade*2 # loss

        if allowShorting == 1:
          allForecastsGain2 += model2Gain
        elif backtestForecastM2[idx] > 0:
          allForecastsGain2 += model2Gain
          
        if (backtestForecastM2outsideOfZTol[idx]): # only make a trade if we are outside ztol
          if allowShorting == 1: # always sum up if we allow shorting
            ztolGain2 += model2Gain
          elif backtestForecastM2[idx] > 0:
            ztolGain2 += model2Gain

        ztolGain2Plot[0].append(reverseIdx)
        ztolGain2Plot[1].append(ztolGain2)

        # model 3
        if (sameSignM3[idx]):     
          model3Gain = shareCount*abs(backtestActual[idx]) - costPerTrade*2
        else:
          model3Gain = -shareCount*abs(backtestActual[idx]) - costPerTrade*2 # loss

        if allowShorting == 1:    
          allForecastsGain3 += model3Gain
        elif backtestForecastM3[idx] > 0:
          allForecastsGain3 += model3Gain
  
        if (backtestForecastM3outsideOfZTol[idx]): # only make a trade if we are outside ztol
          if allowShorting == 1: # always sum up if we allow shorting     
            ztolGain3 += model3Gain
          elif backtestForecastM3[idx] > 0:
            ztolGain3 += model3Gain

        ztolGain3Plot[0].append(reverseIdx)
        ztolGain3Plot[1].append(ztolGain3)
            
      print("buy n hold gain dollars", buyAndHoldGain)

      print("allForecastsGain1      ", allForecastsGain1)
      print("ztolGain1              ", ztolGain1)

      print("allForecastsGain2      ", allForecastsGain2)
      print("ztolGain2              ", ztolGain2)

      print("allForecastsGain3      ", allForecastsGain3)
      print("ztolGain3              ", ztolGain3)

      # initial investment accounts for one cost of trade to open the position

      print("buy n hold gain   rtn  ", buyAndHoldGain/initialInvestment)

      print("allForecastsGain1 rtn  ", allForecastsGain1/initialInvestment)
      print("ztolGain1         rtn  ", ztolGain1/initialInvestment)

      print("allForecastsGain2 rtn  ", allForecastsGain2/initialInvestment)
      print("ztolGain2         rtn  ", ztolGain2/initialInvestment)

      print("allForecastsGain3 rtn  ", allForecastsGain3/initialInvestment)
      print("ztolGain3         rtn  ", ztolGain3/initialInvestment)

      # for checking

      jout["backtestStartDay"] = backtestStartDay
      jout["priceAtOpenBacktest"] = openOnBacktestStartDay
 
      jout["initialInvestmentAtOpenPlusOneTradeCost"] = initialInvestment

      # --- 
      jout["buyAndHoldGain"] = buyAndHoldGain

      jout["allForecastsGain1"] = allForecastsGain1
      jout["ztolGain1"] = ztolGain1

      jout["allForecastsGain2"] = allForecastsGain2
      jout["ztolGain2"] = ztolGain2

      jout["allForecastsGain3"] = allForecastsGain3
      jout["ztolGain3"] = ztolGain3

      # ---
      jout["buyAndHoldGainRtn"] = buyAndHoldGain/initialInvestment

      jout["allForecastsGain1Rtn"] = allForecastsGain1/initialInvestment
      jout["ztolGain1Rtn"] = ztolGain1/initialInvestment

      jout["allForecastsGain2Rtn"] = allForecastsGain2/initialInvestment
      jout["ztolGain2Rtn"] = ztolGain2/initialInvestment

      jout["allForecastsGain3Rtn"] = allForecastsGain3/initialInvestment
      jout["ztolGain3Rtn"] = ztolGain3/initialInvestment

      # plot gains over time

      # show cumulative instead
      
      fig = plt.figure()
      plot(ztolGain3Plot[1])
      fig.savefig('gainsOverTimeM1.pdf')

      fig = plt.figure()
      plot(ztolGain2Plot[1])
      fig.savefig('gainsOverTimeM2.pdf')

      fig = plt.figure()
      plot(ztolGain3Plot[1]) 
      fig.savefig('gainsOverTimeM3.pdf')

      buyAndHoldCumulativePlot = []

      ztolCumulative1Plot = []
      ztolCumulative2Plot = []
      ztolCumulative3Plot = []

      buyAndHoldCumulativePlot.append(initialInvestment) # the zero value

      ztolCumulative1Plot.append(initialInvestment) # the zero value
      ztolCumulative2Plot.append(initialInvestment) # the zero value
      ztolCumulative3Plot.append(initialInvestment) # the zero value

      for xd in range(0, len(ztolGain3Plot[1])):
        ztolCumulative1Plot.append(initialInvestment + ztolGain1Plot[1][xd])
        ztolCumulative2Plot.append(initialInvestment + ztolGain2Plot[1][xd])
        ztolCumulative3Plot.append(initialInvestment + ztolGain3Plot[1][xd])
        buyAndHoldCumulativePlot.append(initialInvestment + buyAndHoldGainPlot[1][xd])

      fig = plt.figure()

      # need buy n hold (?)
      # x axis as dates?

      plot(buyAndHoldCumulativePlot, 'k', label='buy & hold approx')
      plot(ztolCumulative1Plot, 'r', label='model 1')
      plot(ztolCumulative2Plot, 'g', label='model 2') 
      plot(ztolCumulative3Plot, 'b', label='model 3')

      plt.title("backtest of a simple trading system based on the 3 models")

      xlabel("forward in time to the right (days)")
      ylabel("dollars")

      legend(loc='upper left', borderaxespad=0, prop={'size': 6})

      fig.savefig('gainsOverTimeCumulative.pdf')

      print("ztolCumulative3Plot", ztolCumulative3Plot)

      percentDiffArray1 = []
      percentDiffArray2 = [] 
      percentDiffArray3 = []

      for xd in range(1, len(ztolCumulative3Plot)):

        diff1 = ztolCumulative1Plot[xd] - ztolCumulative1Plot[xd-1]
        percentDiff1 = diff1/ztolCumulative1Plot[xd-1]
        percentDiffArray1.append(percentDiff1)
 
        diff2 = ztolCumulative2Plot[xd] - ztolCumulative2Plot[xd-1]
        percentDiff2 = diff2/ztolCumulative2Plot[xd-1]
        percentDiffArray2.append(percentDiff2)

        diff3 = ztolCumulative3Plot[xd] - ztolCumulative3Plot[xd-1]
        percentDiff3 = diff3/ztolCumulative3Plot[xd-1]
        percentDiffArray3.append(percentDiff3)

      percentDiffArray1Neg = []
      percentDiffArray2Neg = []
      percentDiffArray3Neg = []

      percentDiffArray1Pos = []
      percentDiffArray2Pos = []
      percentDiffArray3Pos = []

      for xn in range(0, len(percentDiffArray1)):
        if percentDiffArray1[xn] < 0:
          percentDiffArray1Neg.append(percentDiffArray1[xn])
        else:  
          percentDiffArray1Neg.append(0)

        if percentDiffArray2[xn] < 0:
          percentDiffArray2Neg.append(percentDiffArray2[xn])
        else:
          percentDiffArray2Neg.append(0)

        if percentDiffArray3[xn] < 0:
          percentDiffArray3Neg.append(percentDiffArray3[xn])
        else:
          percentDiffArray3Neg.append(0)

        # now pos

        if percentDiffArray1[xn] >= 0:
          percentDiffArray1Pos.append(percentDiffArray1[xn])
        else:
          percentDiffArray1Pos.append(0)

        if percentDiffArray2[xn] >= 0:
          percentDiffArray2Pos.append(percentDiffArray2[xn])
        else:
          percentDiffArray2Pos.append(0)

        if percentDiffArray3[xn] >= 0:
          percentDiffArray3Pos.append(percentDiffArray3[xn])
        else:
          percentDiffArray3Pos.append(0)

      volatPercentDiff1Neg = numpy.std(percentDiffArray1Neg)
      volatPercentDiff2Neg = numpy.std(percentDiffArray2Neg)
      volatPercentDiff3Neg = numpy.std(percentDiffArray3Neg)

      volatPercentDiff1Pos = numpy.std(percentDiffArray1Pos)
      volatPercentDiff2Pos = numpy.std(percentDiffArray2Pos)
      volatPercentDiff3Pos = numpy.std(percentDiffArray3Pos)

      volatPercentDiff1 = numpy.std(percentDiffArray1)
      percentDiff1 = (ztolCumulative1Plot[len(ztolCumulative1Plot)-1]-ztolCumulative1Plot[0])/ztolCumulative1Plot[0]

      volatPercentDiff2 = numpy.std(percentDiffArray2)
      percentDiff2 = (ztolCumulative2Plot[len(ztolCumulative2Plot)-1]-ztolCumulative2Plot[0])/ztolCumulative2Plot[0]
   
      volatPercentDiff3 = numpy.std(percentDiffArray3)
      percentDiff3 = (ztolCumulative3Plot[len(ztolCumulative3Plot)-1]-ztolCumulative3Plot[0])/ztolCumulative3Plot[0]

      print("percentDiff1", percentDiff1)
      print("volatPercentDiff1", volatPercentDiff1)
      print("volatPercentDiff1Neg", volatPercentDiff1Neg)
      sharpe1  = ((percentDiff1/(ntrials+1)) - 0.01*riskFreeRate/252.0)/volatPercentDiff1
      sortino1 = ((percentDiff1/(ntrials+1)) - 0.01*riskFreeRate/252.0)/volatPercentDiff1Neg  
      sortino1p = ((percentDiff1/(ntrials+1)) - 0.01*riskFreeRate/252.0)/volatPercentDiff1Pos

      if isinf(sharpe1):
        sharpe1 = 0

      if isinf(sortino1):
        sortino1 = 0

      if isinf(sortino1p):
        sortino1p = 0
 
      print("sharpe1", sharpe1) 
      print("sortino1", sortino1)
      print("sortino1p", sortino1p)

      print("percentDiff2", percentDiff2)
      print("volatPercentDiff2", volatPercentDiff2)
      sharpe2 = ((percentDiff2/(ntrials+1)) - 0.01*riskFreeRate/252.0)/volatPercentDiff2
      sortino2 = ((percentDiff2/(ntrials+1)) - 0.01*riskFreeRate/252.0)/volatPercentDiff2Neg
      sortino2p = ((percentDiff2/(ntrials+1)) - 0.01*riskFreeRate/252.0)/volatPercentDiff2Pos

      if isinf(sharpe2):
        sharpe2 = 0

      if isinf(sortino2):
        sortino2 = 0

      if isinf(sortino2p):
        sortino2p = 0

      print("sharpe2", sharpe2) 
      print("sortino2",sortino2)
      print("sortino2p", sortino2p)

      print("percentDiff3", percentDiff3)
      print("volatPercentDiff3", volatPercentDiff3)
      sharpe3 = ((percentDiff3/(ntrials+1)) - 0.01*riskFreeRate/252.0)/volatPercentDiff3  
      sortino3 = ((percentDiff3/(ntrials+1)) - 0.01*riskFreeRate/252.0)/volatPercentDiff3Neg  
      sortino3p = ((percentDiff3/(ntrials+1)) - 0.01*riskFreeRate/252.0)/volatPercentDiff3Pos

      if isinf(sharpe3):
        sharpe3 = 0

      if isinf(sortino3):
        sortino3 = 0

      if isinf(sortino3p):
        sortino3p = 0

      print("sharpe3", sharpe3)
      print("sortino3",sortino3)
      print("sortino3p", sortino3p)
 
      # here is the new stuff, we want the bad forecasts

      diffSignM1 = numpy.sign(backtestActual) != numpy.sign(backtestForecastM1)
      diffSignM2 = numpy.sign(backtestActual) != numpy.sign(backtestForecastM2)
      diffSignM3 = numpy.sign(backtestActual) != numpy.sign(backtestForecastM3)

      diffSignAndOutsideM1 = numpy.logical_and(diffSignM1, backtestForecastM1outsideOfZTol)
      diffSignAndOutsideM2 = numpy.logical_and(diffSignM2, backtestForecastM2outsideOfZTol)
      diffSignAndOutsideM3 = numpy.logical_and(diffSignM3, backtestForecastM3outsideOfZTol)

      print("diffSignAndOutsideM1 =", diffSignAndOutsideM1)
      print("diffSignAndOutsideM2 =", diffSignAndOutsideM2)
      print("diffSignAndOutsideM3 =", diffSignAndOutsideM3)

      # now here is where we can compute the stairstep fail plots 

      # all stair step plots should have the same len, check it?

      for istair in range(0, len(diffSignAndOutsideM2)): # use safe linear as our mantra for len
        if diffSignAndOutsideM1[istair]:
          badrunZTol += 1
        else: # a good run...the bad sequence is broken, so save it and reset the count
          badrunsequenceZTol.append(badrunZTol)
          badrunZTol = 0

        # same for model 2
        if diffSignAndOutsideM2[istair]:
          badrun2ZTol += 1
        else: # a good run...the bad sequence is broken, so save it and reset the count
          badrunsequence2ZTol.append(badrun2ZTol)
          badrun2ZTol = 0

        # same for model 3 
        if diffSignAndOutsideM3[istair]:
          badrun3ZTol += 1
        else: # a good run...the bad sequence is broken, so save it and reset the count
          badrunsequence3ZTol.append(badrun3ZTol)
          badrun3ZTol = 0

        badrunstairZTol.append(badrunZTol)
        badrunstair2ZTol.append(badrun2ZTol)
        badrunstair3ZTol.append(badrun3ZTol)

      print("badrunstairZTol", badrunstairZTol)
      print("badrunstair2ZTol", badrunstair2ZTol)
      print("badrunstair3ZTol", badrunstair3ZTol)

      # note:  there is a difference between badrunstair and badrunsequence
      # bad run sequence only includes the top stairstep not the steps up to it
      # SEQUENCE is what the metrics are computed from

      # init for print purposes
      brsNoZeroZTol = [0]
      brsNoZero2ZTol = [0]
      brsNoZero3ZTol = [0]

      print("badrunsequenceZTol", badrunsequenceZTol)
      print("badrunsequence2ZTol", badrunsequence2ZTol)
      print("badrunsequence3ZTol", badrunsequence3ZTol)

      brsNoZeroZTol = copy.copy(badrunsequenceZTol)
      brsNoZero2ZTol = copy.copy(badrunsequence2ZTol)
      brsNoZero3ZTol = copy.copy(badrunsequence3ZTol)

      def removeZeros(array):
        print("before array", array) 
        arrayClean = []
        for ival in range(0, len(array)):
          if array[ival] != 0:
            arrayClean.append(float(array[ival]))
        print("after array", arrayClean)
        return arrayClean 

      # orig way may be right
      if False:
        while brsNoZeroZTol.count(0) > 0:
          brsNoZeroZTol.remove(0)
          print("after one 0 remove: ", brsNoZeroZTol)

        while brsNoZero2ZTol.count(0) > 0:
          brsNoZero2ZTol.remove(0)

        while brsNoZero3ZTol.count(0) > 0:
          brsNoZero3ZTol.remove(0)
      else:
        brsNoZeroZTol = removeZeros(brsNoZeroZTol)
        brsNoZero2ZTol = removeZeros(brsNoZero2ZTol)
        brsNoZero3ZTol = removeZeros(brsNoZero3ZTol)

      if len(brsNoZeroZTol) == 0:
        brsNoZeroZTol = [0]
      if len(brsNoZero2ZTol) == 0:
        brsNoZero2ZTol = [0]
      if len(brsNoZero3ZTol) == 0:
        brsNoZero3ZTol = [0]
 
      print("bad run stats after zero tol: avg max avgNoZero")
      print(numpy.mean(badrunsequenceZTol), " ", numpy.max(badrunsequenceZTol), " ", numpy.mean(brsNoZeroZTol))
      print(numpy.mean(badrunsequence2ZTol), " ", numpy.max(badrunsequence2ZTol), " ", numpy.mean(brsNoZero2ZTol))
      print(numpy.mean(badrunsequence3ZTol), " ", numpy.max(badrunsequence3ZTol), " ", numpy.mean(brsNoZero3ZTol))

      # now redo these at the end with the ztol'd versions

      print("brsNoZeroZTol", brsNoZeroZTol)
      print("brsNoZero2ZTol", brsNoZero2ZTol)
      print("brsNoZero3ZTol", brsNoZero3ZTol)

      meanbad = str(numpy.mean(brsNoZeroZTol))
      meanbad2 = str(numpy.mean(brsNoZero2ZTol))
      meanbad3 = str(numpy.mean(brsNoZero3ZTol))

      if (len(badrunsequenceZTol) > 0):
        L8 =  "model 1 bad predict sequence ztol max = " + str(numpy.max(badrunsequenceZTol))  + " mean = " + meanbad[:5]
      else:
        L8 = ""

      if (len(badrunsequence2ZTol) > 0):
        L9 =  "model 2 bad predict sequence ztol max = " + str(numpy.max(badrunsequence2ZTol)) + " mean = " + meanbad2[:5]
      else:
        L9 = ""

      if (len(badrunsequence3ZTol) > 0):
        L10 = "model 3 bad predict sequence ztol max = " + str(numpy.max(badrunsequence3ZTol)) + " mean = " + meanbad3[:5]
      else:
        L10 = ""

      # overwrites orig in-loop computation, caution with L1 ... L7
      # wait, this is in loop doi

      Larray = [L1, L2, L3, L4, L5, L6, L7, " ",  L8, L9, L10];

      jout["running_tallies"] = Larray;

      # plot the bad run stair plots for the ZTol cases (overwrite the orig for now)
      # note:  we are using ymax previously computed for the non zero tol'd plots
      # hence we should maybe not get rid of the non zero tol'd plots code

      fig = plt.figure()

      print("len stair array = ", len(badrunstairZTol))
      print("len stair array 2 = ", len(badrunstair2ZTol))
      print("len stair array 3 = ", len(badrunstair3ZTol))

      ### this is causing the overlap of graphics bug
      ### DEBUG py3.11 plt.title("consecutive bad forecasts in backtest considering ZTOL")

      subplot(311)

      plt.title("Y axes = cumulative bad forecast days, consecutive bad forecasts in backtest considering ZTOL\nduring backtest (lower is better, grey = without ztol)\ndots = trade was recommended", fontsize=8)
 
      xlim(0, len(badrunstairZTol))
      ylim(0, ymax)

      graycolor = "0.85"

      wherestep = "mid"

      plt.ylabel("model 1")
      step(list(range(0,len(badrunstair))),     badrunstair,     color=graycolor, where=wherestep)
      step(list(range(0,len(badrunstairZTol))), badrunstairZTol, color="r", where=wherestep)

      xdot, ydot = makeTradeRecommenedArrays(backtestForecastM1outsideOfZTol)

      plot(xdot, ydot, 'k.')

      subplot(312)

      xlim(0, len(badrunstairZTol)) # use same x as 1st
      ylim(0, ymax)

      plt.ylabel("model 2")
      step(list(range(0,len(badrunstair2))),    badrunstair2,     color=graycolor, where=wherestep)
      step(list(range(0,len(badrunstair2ZTol))),badrunstair2ZTol, color="b", where=wherestep)

      xdot, ydot = makeTradeRecommenedArrays(backtestForecastM2outsideOfZTol)
      plot(xdot, ydot, 'k.')

      subplot(313)

      xlim(0, len(badrunstairZTol)) # use same x as 1st
      ylim(0, ymax)

      plt.ylabel("model 3")
      step(list(range(0,len(badrunstair3))),    badrunstair3,     color=graycolor, where=wherestep) 
      step(list(range(0,len(badrunstair3ZTol))),badrunstair3ZTol, color="g", where=wherestep)

      xdot, ydot = makeTradeRecommenedArrays(backtestForecastM3outsideOfZTol)
      plot(xdot, ydot, 'k.')

      plt.xlabel("backtest day:  0 = yesterday, to the right = back in time")

      # not working so great w/ the matplotlib that is current now in py3.11 so combine this text w/ top plot title
      # plt.title("consecutive bad forecasts in backtest considering ZTOL", fontsize=10)

      fig.savefig('backtest_runs.pdf')

      # end new stuff

      countSuccessM1 = numpy.count_nonzero(sameSignAndOutsideM1)
      countSuccessM2 = numpy.count_nonzero(sameSignAndOutsideM2)
      countSuccessM3 = numpy.count_nonzero(sameSignAndOutsideM3)

      countOutsideM1 = numpy.count_nonzero(backtestForecastM1outsideOfZTol)
      countOutsideM2 = numpy.count_nonzero(backtestForecastM2outsideOfZTol)
      countOutsideM3 = numpy.count_nonzero(backtestForecastM3outsideOfZTol)

      print("countSuccessM1", countSuccessM1, "out of", countOutsideM1)
      print("countSuccessM2", countSuccessM2, "out of", countOutsideM2)
      print("countSuccessM3", countSuccessM3, "out of", countOutsideM3)

      # new stuff count fails for checking

      countFailM1 = numpy.count_nonzero(diffSignAndOutsideM1)
      countFailM2 = numpy.count_nonzero(diffSignAndOutsideM2)
      countFailM3 = numpy.count_nonzero(diffSignAndOutsideM3)

      print("countFailM1", countFailM1, "out of", countOutsideM1)
      print("countFailM2", countFailM2, "out of", countOutsideM2)
      print("countFailM3", countFailM3, "out of", countOutsideM3)
 
      # end new stuff

      # --- end badrun w/ ztol & plot
      
      # -----
      # confusion matrix computations for zero toleranced cases
      # note that left is actual, right is predicted in the coding 
      # initially we start w/ prior code

      # ref backtestForecastM1outsideOfZTol

      # this is repeating the whole of confusion matrix calcs, maybe only need the AND with BEST

      posPos1z = numpy.logical_and(positive(backtestActual), positive(backtestForecastM1))
      negNeg1z = numpy.logical_and(negative(backtestActual), negative(backtestForecastM1))

      posNeg1z =  numpy.logical_and(positive(backtestActual), negative(backtestForecastM1))
      negPos1z =  numpy.logical_and(negative(backtestActual), positive(backtestForecastM1))

      # here we apply the outside of z tol mask

      print("posPos1z", posPos1z)
      print("BESTbacktestForecastM1outsideOfZTol", BESTbacktestForecastM1outsideOfZTol)

      posPos1z = numpy.logical_and(posPos1z, BESTbacktestForecastM1outsideOfZTol)
      negNeg1z = numpy.logical_and(negNeg1z, BESTbacktestForecastM1outsideOfZTol)

      posNeg1z = numpy.logical_and(posNeg1z, BESTbacktestForecastM1outsideOfZTol)
      negPos1z = numpy.logical_and(negPos1z, BESTbacktestForecastM1outsideOfZTol)
 
      if False:
        print("confusion raw data")
        print("posPos1z", posPos1z)
        print("negNeg1z", negNeg1z)
        print("posNeg1z", posNeg1z)
        print("negPos1z", negPos1z)

      # these outputs are sent to the json output later in the run jout

      print("confusion matrix m1 for ztol")
      print("posPos1z = ", numpy.count_nonzero(posPos1z))
      print("negNeg1z = ", numpy.count_nonzero(negNeg1z))
      print("posNeg1z = ", numpy.count_nonzero(posNeg1z))
      print("negPos1z = ", numpy.count_nonzero(negPos1z))

      posPos2z = numpy.logical_and(positive(backtestActual), positive(backtestForecastM2))
      negNeg2z = numpy.logical_and(negative(backtestActual), negative(backtestForecastM2))

      posNeg2z =  numpy.logical_and(positive(backtestActual), negative(backtestForecastM2))
      negPos2z =  numpy.logical_and(negative(backtestActual), positive(backtestForecastM2))

      # here we apply the outside of z tol mask

      print("posPos2z", posPos2z)
      print("BESTbacktestForecastM2outsideOfZTol", BESTbacktestForecastM2outsideOfZTol)

      posPos2z = numpy.logical_and(posPos2z, BESTbacktestForecastM2outsideOfZTol)
      negNeg2z = numpy.logical_and(negNeg2z, BESTbacktestForecastM2outsideOfZTol)

      posNeg2z = numpy.logical_and(posNeg2z, BESTbacktestForecastM2outsideOfZTol)
      negPos2z = numpy.logical_and(negPos2z, BESTbacktestForecastM2outsideOfZTol)

      print("confusion matrix m2 for ztol")
      print("posPos2z = ", numpy.count_nonzero(posPos2z))
      print("negNeg2z = ", numpy.count_nonzero(negNeg2z))
      print("posNeg2z = ", numpy.count_nonzero(posNeg2z))
      print("negPos2z = ", numpy.count_nonzero(negPos2z))

      posPos3z = numpy.logical_and(positive(backtestActual), positive(backtestForecastM3))
      negNeg3z = numpy.logical_and(negative(backtestActual), negative(backtestForecastM3))

      posNeg3z =  numpy.logical_and(positive(backtestActual), negative(backtestForecastM3))
      negPos3z =  numpy.logical_and(negative(backtestActual), positive(backtestForecastM3))
      
      # here we apply the outside of z tol mask
 
      print("posPos3z", posPos3z)
      print("BESTbacktestForecastM3outsideOfZTol", BESTbacktestForecastM3outsideOfZTol)

      posPos3z = numpy.logical_and(posPos3z, BESTbacktestForecastM3outsideOfZTol)
      negNeg3z = numpy.logical_and(negNeg3z, BESTbacktestForecastM3outsideOfZTol)

      posNeg3z = numpy.logical_and(posNeg3z, BESTbacktestForecastM3outsideOfZTol)
      negPos3z = numpy.logical_and(negPos3z, BESTbacktestForecastM3outsideOfZTol)

      print("confusion matrix m3 for ztol")
      print("posPos3z = ", numpy.count_nonzero(posPos3z))
      print("negNeg3z = ", numpy.count_nonzero(negNeg3z))
      print("posNeg3z = ", numpy.count_nonzero(posNeg3z))
      print("negPos3z = ", numpy.count_nonzero(negPos3z))

    else:
      print("zero tolerances not computed for forecast")
 
    # print 'The scikit-learn version is ', sklearn.__version__
 
    ### pdf_pages.savefig(fig) # save to the multipage file also

    havepercentrange = True # assume yes

    if not ('percentrange' in vars()):
      # this will arise if model 3 is not KNN classifier 
      # in this case fill in with Lars coeffs  
      print((clf.intercept_))
      print((clf.coef_))

      tmparray = copy.deepcopy(clf.coef_)
      #del tmparray[0] # get rid of bogus constant
      tmparray[0] = clf.intercept_

      percentrange =  tmparray
     
      havepercentrange = False # the data is now coefficients from LARS
    
    if True:
      fig = plt.figure()

      thetitle = "tgt="+symbols[0]+"c-o, "

      ## plt.title(thetitle + "estimated relative unsigned importance of lagged vars in most recent time window, KNN classifier", fontsize=6)

      subplots_adjust(bottom=0.15, top = 0.8)

      axes = subplot(211)

      if havepercentrange:
        axes.set_title(thetitle + "estimated relative unsigned importance of lagged vars in most recent time window, KNN classifier", fontsize=6)

        axes.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
      else:
        axes.set_title(thetitle + "coefficients of partial linear LARS model, insignificant coeffs set to 0", fontsize=6)
 
      # busted in latest plot library
      #grid(b=True, which='minor', color='0.4', linestyle='-')
      #grid(b=True, which='major', color='0.8', linestyle='-')

      plot(percentrange, "go")

      aticklist = axes.get_xticks().tolist()

      modheader = copy.deepcopy(headerline)

      del modheader[0]
      del modheader[0]
      del modheader[0] 

      modheader.insert(0, "const")

      print("len aticklist = ", len(aticklist))
      print("len modheader = ", len(modheader))

      axes.tick_params(axis='both', which='major', labelsize=6)
 
      #axes.set_xticklabels(modheader)

      # put on bottom graph    xticks(range(0, len(modheader)), modheader, rotation=90)

      # just plain unmarked ticks on top graph

      emptyticks = [""]*len(modheader)

      xticks(list(range(0, len(modheader))), emptyticks)

      # put on bottom graph    xlabel("1 or 2 indicates # of days lagged -- vol = recent volatility -- all variables are daily differenced", fontsize=8) 

      #plot(percentrange, "go")

      # other plot 

      axes = subplot(212)
  
      axes.set_title(thetitle + "full linear model coefficients, not all are significant, all variables in daily differenced dollar units", fontsize=6)
 
      # busted in latest plot library
      #grid(b=True, which='minor', color='0.4', linestyle='-')
      #grid(b=True, which='major', color='0.8', linestyle='-')
 
      plot(solved[0], "bo") # full linear
 
      axes.tick_params(axis='both', which='major', labelsize=6)
      xticks(list(range(0, len(modheader))), modheader, rotation=90)
 
      xlabel("1 or 2 indicates # of days lagged -- vol = recent volatility -- all variables are daily differenced", fontsize=8) 
      #ylabel("full linear cofficients", fontsize=6)
 
      fig.savefig('sens.pdf')

      ### pdf_pages.savefig(fig) # save also to the multi page file
 
  bestarray = ["", "", ""];

  if (forecastcorrect > forecastcorrect2 and forecastcorrect > forecastcorrect3):
    bestarray[0] = "<-- best"
  elif (forecastcorrect2 > forecastcorrect and forecastcorrect2 > forecastcorrect3):
    bestarray[1] = "<-- best"
  elif (forecastcorrect3 > forecastcorrect and forecastcorrect3 > forecastcorrect2):
    bestarray[2] = "<-- best"

  model1desc = " anneal "

  methodSuffix = ""
  if annealToUse == 'BFGS':
    methodSuffix = "(BFGS) " 

  if model1minabs == 1:
    model1desc = " minabs^" + str(residExp) + " "
  else:
    if coolrate == 0:
      model1desc = " anneal fast " + methodSuffix
    elif coolrate == 1:
      model1desc = " anneal slow " + methodSuffix
    elif coolrate == 2:
      model1desc = " BFGS opt "
 
  model3desc = " custom model "

  if uselogit == 1 and uselars == False :
    model3desc = " knc "

  if uselogit == 0 and uselars == True :
    if lassolarsbic == 1:
      model3desc = " LLBIC "
    else:
      model3desc = " LLAIC "
 
  if (forecastrow >= 0):
    L1 = "running % correct model 1 " + str(forecastcorrect/float(forecastrow+1))[:6] + " " + str(forecastcorrect) + " of " + str(forecastrow+1) + model1desc + bestarray[0]
    L2 = "running % correct model 2 " + str(forecastcorrect2/float(forecastrow+1))[:6] + " " + str(forecastcorrect2) + " of " + str(forecastrow+1) + " lstsq " + bestarray[1]
    L3 = "running % correct model 3 " + str(forecastcorrect3/float(forecastrow+1))[:6] + " " + str(forecastcorrect3) + " of " + str(forecastrow+1) + model3desc + bestarray[2]

    L4 = "probability of achieving above results or better randomly by coin flips [lower is better]:\n"

    prob1 = 1- prob(forecastcorrect-1, 0.5, forecastrow+1);
    prob2 = 1- prob(forecastcorrect2-1, 0.5, forecastrow+1);
    prob3 = 1- prob(forecastcorrect3-1, 0.5, forecastrow+1);
    
    L5 = "coinflips to achieve backtest model 1 " + str(prob1)[:9] + " : " + probmsg(prob1)
    L6 = "coinflips to achieve backtest model 2 " + str(prob2)[:9] + " : " + probmsg(prob2)
    L7 = "coinflips to achieve backtest model 3 " + str(prob3)[:9] + " : " + probmsg(prob3)

    # redo these at the end with the ztol'd versions

    meanbad = str(numpy.mean(brsNoZero))
    meanbad2 = str(numpy.mean(brsNoZero2))
    meanbad3 = str(numpy.mean(brsNoZero3))

    if (len(badrunsequence) > 0):  
      L8 =  "model 1 bad predict sequence max = " + str(numpy.max(badrunsequence))  + " mean = " + meanbad[:5] 
    else:
      L8 = "" 

    if (len(badrunsequence2) > 0): 
      L9 =  "model 2 bad predict sequence max = " + str(numpy.max(badrunsequence2)) + " mean = " + meanbad2[:5]
    else:
      L9 = ""

    if (len(badrunsequence3) > 0):
      L10 = "model 3 bad predict sequence max = " + str(numpy.max(badrunsequence3)) + " mean = " + meanbad3[:5]
    else:
      L10 = ""

    # useful for optimizers 

    if (len(badrunsequence) > 0 and len(badrunsequence2) > 0 and len(badrunsequence3) > 0):
      optoutput = str(forecastcorrect) + " " + str(forecastcorrect2) + " " + str(forecastcorrect3) + " " + str(prob1)[:6] + " " + str(prob2)[:6] + " " + str(prob3)[:6] + " " + str(numpy.max(badrunsequence)) + " " + str(numpy.max(badrunsequence2)) + " " + str(numpy.max(badrunsequence3)) + " " + meanbad + " " + meanbad2 + " " + meanbad3 # no \n on end

      with open("optpoint", 'w') as optfile:
        optfile.write(optoutput)

    print(L1)
    print(L2)
    print(L3)
    print(L4)
    print(L5)
    print(L6)
    print(L7)

    print(L8)
    print(L9)
    print(L10)

    Larray = [L1, L2, L3, L4, L5, L6, L7, " ",  L8, L9, L10];

    # remove ... this is now assigned when the ztol versions of bad run metrics are computed, above 
    # now we want only the intermediate "while backtesting" runs to show this since we dont have ztol version yet
    if forecastrow != ntrials:
      jout["running_tallies"] = Larray; 

    # metrics for linear & lars if any

    meanLinearAdjR2 = numpy.mean(linearAdjR2List)
    stdLinearAdjR2 = numpy.std(linearAdjR2List)

    print("running meanLinearAdjR2 =", meanLinearAdjR2)
    print("running stdLinearAdjR2  =", stdLinearAdjR2)

    meanLarsAdjR2 = numpy.mean(larsAdjR2List)
    stdLarsAdjR2 = numpy.std(larsAdjR2List)
    
    print("running meanLarsAdjR2 =", meanLarsAdjR2)
    print("running stdLarsAdjR2  =", stdLarsAdjR2)

    # see how pvalue is computed above
    print("OUTPUT meanLarsAdjR2 =", meanLarsAdjR2, " p-value =", prob3, " percentCorrect = ", forecastcorrect3/float(forecastrow+1))

  else:
    print("skipping backtest calc for first point")

  jout["status"] = str(forecastrow+1) + " backtest trials completed in this run"

  if (forecastrow+1) == 0:
    jout["status"] = jout["status"] + "...forward forecast only"

  if skipio == False:
    with open("status", "w") as outfile:
      json.dump(jout, outfile)

  endTimeOneRow = 0 # time.clock()

  oneRowTime = (endTimeOneRow-startTimeOneRow)

  sumOfRowTimes += oneRowTime 
  sumOfRowTimesCount += 1

  print("one backtest row exe time =", str(oneRowTime))

print("all row times = ", sumOfRowTimes)
print("avg row time = ", sumOfRowTimes/sumOfRowTimesCount)

meanLinearAdjR2 = numpy.mean(linearAdjR2List)
stdLinearAdjR2 = numpy.std(linearAdjR2List)

print("final meanLinearAdjR2 =", meanLinearAdjR2)
print("final stdLinearAdjR2  =", stdLinearAdjR2)

meanLarsAdjR2 = numpy.mean(larsAdjR2List)
stdLarsAdjR2 = numpy.std(larsAdjR2List)

print("final meanLarsAdjR2 =", meanLarsAdjR2)
print("final stdLarsAdjR2  =", stdLarsAdjR2)

# end backtest (?)

if ntrials > -1: 
  print("forecast correct1 = ", forecastcorrect, " of ", ntrials+1)
  print("forecast correct2 = ", forecastcorrect2, " of ", ntrials+1)
  print("forecast correct3 = ", forecastcorrect3, " of ", ntrials+1)
 
  print("percent model 1 = ", forecastcorrect/float(ntrials+1))
  print("percent model 2 = ", forecastcorrect2/float(ntrials+1))
  print("percent model 3 = ", forecastcorrect3/float(ntrials+1))
else:
  print("current day model only; no backtests done")

# the left is actual, right is predicted (e.g. posNeg...  actual is pos)
# these are the backtest confusion matrices with no zero tolerance

jout["confusion_posPos1"] = numpy.count_nonzero(posPos1)
jout["confusion_negNeg1"] = numpy.count_nonzero(negNeg1)
jout["confusion_posNeg1"] = numpy.count_nonzero(posNeg1)
jout["confusion_negPos1"] = numpy.count_nonzero(negPos1)

jout["confusion_posPos2"] = numpy.count_nonzero(posPos2)
jout["confusion_negNeg2"] = numpy.count_nonzero(negNeg2)
jout["confusion_posNeg2"] = numpy.count_nonzero(posNeg2)
jout["confusion_negPos2"] = numpy.count_nonzero(negPos2)

jout["confusion_posPos3"] = numpy.count_nonzero(posPos3)
jout["confusion_negNeg3"] = numpy.count_nonzero(negNeg3)
jout["confusion_posNeg3"] = numpy.count_nonzero(posNeg3)
jout["confusion_negPos3"] = numpy.count_nonzero(negPos3)

# confusion matrices considering zero tolerances

# fix error on forecast
try: posPos1z 
except NameError: posPos1z = None

if posPos1z is not None:
  jout["confusion_posPos1z"] = numpy.count_nonzero(posPos1z)
  jout["confusion_negNeg1z"] = numpy.count_nonzero(negNeg1z)
  jout["confusion_posNeg1z"] = numpy.count_nonzero(posNeg1z)
  jout["confusion_negPos1z"] = numpy.count_nonzero(negPos1z)

  jout["confusion_posPos2z"] = numpy.count_nonzero(posPos2z)
  jout["confusion_negNeg2z"] = numpy.count_nonzero(negNeg2z)
  jout["confusion_posNeg2z"] = numpy.count_nonzero(posNeg2z)
  jout["confusion_negPos2z"] = numpy.count_nonzero(negPos2z)

  jout["confusion_posPos3z"] = numpy.count_nonzero(posPos3z)
  jout["confusion_negNeg3z"] = numpy.count_nonzero(negNeg3z)
  jout["confusion_posNeg3z"] = numpy.count_nonzero(posNeg3z)
  jout["confusion_negPos3z"] = numpy.count_nonzero(negPos3z)

# best pvals from quick scan of zero tolerances

jout["bestM1pval"] = bestM1pval
jout["m1ZTol"] = m1ZTol
jout["bestCorrectNumeratorM1"] = bestCorrectNumeratorM1
jout["bestCorrectDenominatorM1"] = bestCorrectDenominatorM1

jout["bestM2pval"] = bestM2pval
jout["m2ZTol"] = m2ZTol
jout["bestCorrectNumeratorM2"] = bestCorrectNumeratorM2
jout["bestCorrectDenominatorM2"] = bestCorrectDenominatorM2

jout["bestM3pval"] = bestM3pval
jout["m3ZTol"] = m3ZTol
jout["bestCorrectNumeratorM3"] = bestCorrectNumeratorM3
jout["bestCorrectDenominatorM3"] = bestCorrectDenominatorM3

# report out model parameters 

jout["windowsize"] = windowsize
jout["neighbors"] = neighbors
jout["varcutoff"] = knnvarcutoff / 10.0 # back to percent
jout["volen"] = volen
jout["epsilon"] = epsilon # not x100
jout["exponent"] = exponent
jout["symbols"] = symbols # array 

# added oct 12 2022 more outputs

jout["normalize"] = normalize
jout["daysWithheld"] = daysWithheld
jout["allowShorting"] = allowShorting

jout["riskFreeRate"] = riskFreeRate

jout["nvar"] = len(Bfeed) # including constant

print("nonZeroCoeffsPerBacktestM1", nonZeroCoeffsPerBacktestM1)
print("nonZeroCoeffsPerBacktestM3", nonZeroCoeffsPerBacktestM3)

if len(nonZeroCoeffsPerBacktestM1) > 0:
  jout["m1_50thCoeffCount"] = numpy.percentile(nonZeroCoeffsPerBacktestM1, 50)
  jout["m1_5thCoeffCount"] =  numpy.percentile(nonZeroCoeffsPerBacktestM1, 5)
  jout["m1_95thCoeffCount"] =  numpy.percentile(nonZeroCoeffsPerBacktestM1, 95)
else:
  jout["m1_50thCoeffCount"] = len(Bfeed)
  jout["m1_5thCoeffCount"] =  len(Bfeed)
  jout["m1_95thCoeffCount"] = len(Bfeed)

if len(nonZeroCoeffsPerBacktestM3) > 0:
  jout["m3_50thCoeffCount"] = numpy.percentile(nonZeroCoeffsPerBacktestM3, 50)
  jout["m3_5thCoeffCount"] =  numpy.percentile(nonZeroCoeffsPerBacktestM3, 5)
  jout["m3_95thCoeffCount"] =  numpy.percentile(nonZeroCoeffsPerBacktestM3, 95)
else:
  jout["m3_50thCoeffCount"] = len(Bfeed)
  jout["m3_5thCoeffCount"] =  len(Bfeed)
  jout["m3_95thCoeffCount"] = len(Bfeed)

try:
  jout["sharpe1"] = sharpe1
  jout["sortino1"] = sortino1
  jout["sortino1p"] = sortino1p

  jout["sharpe2"] = sharpe2
  jout["sortino2"] = sortino2
  jout["sortino2p"] = sortino2p

  jout["sharpe3"] = sharpe3
  jout["sortino3"] = sortino3
  jout["sortino3p"] = sortino3p
except Exception as e: # for forward forecast we dont have above ratios 
  print("no ratios for forecast")
  
jout["alldone"] = "1"
jout["sys.version"] = sys.version

# busted in python3.11 due to trying to convert int64 to JSON 
#with open("status", "w") as outfile:
# json.dump(jout, outfile)

# FIX: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
with open("status", "w") as outfile:
  json.dump(jout, outfile, cls=NpEncoder, indent=2)

# write out forecasts computed during the trade loop

myfile = open("regtableWithBacktests.csv", 'w')
wr = csv.writer(myfile)

if 'regtableWithBacktests' in locals():
  for thelist in regtableWithBacktests: 
    wr.writerow(thelist)

myfile.close()

sys.exit(0)

