import numpy as np
import pandas as pd 
import math
import scipy.stats as st
import random
import matplotlib.pyplot as plt
import seaborn as sns 


def censorBySales(demand, order):
    observedDemand = []
    for i in demand:
        observedDemand.append(min(i, order))
    return observedDemand


## Models

def myopicNewsvendor(price, cost, demand, w = 10000): 
    observedDemand = censorBySales(demand[:100], 110)
    demand = demand[100:]
    profit = 0
    logs = pd.DataFrame()

    for day in range(len(demand)):    
        order = np.mean(observedDemand) + np.std(observedDemand)*st.norm.ppf((price-cost)/price)
        sales = min(order, demand[day])
        observedDemand.append(sales) 

        if len(observedDemand) > w:
            observedDemand = observedDemand[1:]

        profit += sales*price - order*cost

        logs.loc[day, 'order'] = round(order,2)
        logs.loc[day,'sales'] = round(sales,2) 
        logs.loc[day, 'demand'] = round(demand[day],2)
        logs.loc[day, 'profit'] = round(profit,2)
    return logs

def optimalNewsvendor(price, cost, demand, mean, std): 
    profit = 0
    logs = pd.DataFrame() 

    for day in range(len(demand)):    
        order = mean + std*st.norm.ppf((price-cost)/price)
        sales = min(order, demand[day])
        profit += sales*price - order*cost

        logs.loc[day, 'order'] = round(order,2)
        logs.loc[day,'sales'] = round(sales,2) 
        logs.loc[day, 'demand'] = round(demand[day],2)
        logs.loc[day, 'profit'] = round(profit,2)
    return logs


def MLE(price, cost, demand, w = 10000):
    observedDemand = censorBySales(demand[:100], 110)
    demand = demand[100:]
    profit = 0
    logs = pd.DataFrame()

    estMean = np.mean(observedDemand)
    estStd = np.std(observedDemand)
    stockoutDemand = observedDemand[-3:]
    S2_error = []
    stockouts = [0, 1, 0]

    for day in range(len(demand)): 
        if estMean > 1:   
            order = estMean + estStd*st.norm.ppf((price-cost)/price) 
        sales = min(order, demand[day])
        observedDemand.append(sales) 
        
        profit += sales*price - order*cost
        if sales == demand[day]: 
            stockouts.append(1)
            stockoutDemand.append(sales)
        else: 
            stockouts.append(0)
            stockoutDemand.append(0)

        if len(stockouts) > w:
            stockouts = stockouts[1:]
            stockoutDemand = stockoutDemand[1:]

        # MLE Parameters

        p = np.mean(stockouts)
        z = st.norm.ppf(p)
        d_bar = np.mean([i for i in stockoutDemand if i != 0])
        if sales == demand[day]:
            S2_error.append((sales - d_bar)*(sales - d_bar))
        S2 = np.sum(S2_error)/(np.sum(stockouts))  

        # MLE Update mean & std
        if day > 2:
            estStd = np.sqrt(S2/(1 - (z*st.norm.pdf(z)/p) - (st.norm.pdf(z)/p)*(st.norm.pdf(z)/p)))
            estMean = d_bar + estStd*st.norm.pdf(z)/p

        logs.loc[day, 'order'] = round(order,2)
        logs.loc[day,'sales'] = round(sales,2) 
        logs.loc[day, 'demand'] = round(demand[day],2)
        logs.loc[day, 'profit'] = round(profit,2)
        logs.loc[day, 'mean'] = round(estMean,2)
        logs.loc[day, 'std'] = round(estStd,2)
    return logs


def fastLearning(price, cost, demand):
    observedDemand = censorBySales(demand[:100], 110)
    demand = demand[100:]
    profit = 0
    logs = pd.DataFrame()   
    
    #parameter selection
    gamma = np.mean(observedDemand)
    learnThreshold = 3
    testThreshold = 3
    errorRange = 0.01*np.mean(observedDemand)
    stockoutThreshold = 0.2
    expectedStockouts = (price-cost)/price
  
    # Simulation settings

    learn = True
    learnCounter = 0
    testCounter = 0
    learntDemand = observedDemand[-3:]
    learntStockouts = []
        
    previousMean = np.mean(observedDemand)

    for day in range(len(demand)):
     
        order = np.mean(learntDemand) + np.std(learntDemand)*st.norm.ppf((price-cost)/price) + learn*gamma
        sales = min(order, demand[day])
        observedDemand.append(sales)                
        profit += sales*price - order*cost
        
        if learn:
            learntDemand.append(sales)

            # Stopping Trigger
            if np.abs(np.mean(learntDemand) - previousMean) <= errorRange:
                learnCounter += 1
            else: 
                learnCounter = 0
            if learnCounter > learnThreshold: 
                learn = False 
                learntStockouts = []
                for i in range(30):
                    learntStockouts.append(expectedStockouts)        
        else:
            learntStockouts = learntStockouts[1:]
            learntStockouts.append(int(order <= demand[day]))
            
            # Starting Trigger
            if np.abs(np.mean(learntStockouts) - expectedStockouts) > stockoutThreshold:
                testCounter += 1
                if testCounter > testThreshold:
                    learn = True
                    learntDemand = learntDemand[-10:]
                    learntStockouts = learntStockouts[-10:]
                    learnCounter = 0
            else:
                testCounter = 0
        previousMean = np.mean(learntDemand)

        logs.loc[day, 'order'] = round(order,2)
        logs.loc[day, 'sales'] = round(sales,2)
        logs.loc[day, 'demand'] = round(demand[day],2)
        logs.loc[day, 'profit'] = round(profit,2)
    return logs

def dualCensoredLearning(price, cost, demand):
    
    observedDemand = censorBySales(demand[:100], 110)
    demand = demand[100:]
    profit = 0
    logs = pd.DataFrame()   
    

    def dualCensoring(demand,stockouts):
        lowerLimit = np.sort(demand)[stockouts]
        censoredDemand = [] 
        for d in demand:
            censoredDemand.append(max(d, lowerLimit))
        return censoredDemand

    def mirroring(demand, mean):
        lowerHalf = []
        for d in demand:
            if d < mean: lowerHalf.append(mean - d)

        mirroredDemand = []
        for d in demand:
            if d <= mean:
                mirroredDemand.append(d)
            else:
                mirroredDemand.append(mean + lowerHalf[random.randint(0,len(lowerHalf)-1)])

        return mirroredDemand
    
    #parameter selection
    gamma = np.mean(observedDemand)/2 
    learnThreshold = 3
    testThreshold = 3
    errorRange = 0.01*np.mean(observedDemand)
    stockoutThreshold = 0.2
    expectedStockouts = (price-cost)/price
  
    # FastLearning settings
    learn = True
    learnCounter = 0
    testCounter = 0
    learntDemand = observedDemand[-3:]
    learntStockouts = [1]        
    previousMean = np.mean(observedDemand)

    # DualCensoring settings 
    dualCensoredDemand = observedDemand[-3:]
    mirroredDemand = observedDemand[-3:]

    for day in range(len(demand)):     
        order = np.mean(dualCensoredDemand) + np.std(mirroredDemand)*st.norm.ppf((price-cost)/price) + learn*gamma
        sales = min(order, demand[day])
        observedDemand.append(sales)                
        profit += sales*price - order*cost
        
        if learn:
            learntDemand.append(sales)
            dualCensoredDemand = dualCensoring(learntDemand, int(np.mean(learntStockouts)))
            mirroredDemand = mirroring(learntDemand, np.mean(dualCensoredDemand))

            # Stopping Trigger
            if np.abs(np.mean(learntDemand) - previousMean) <= errorRange:
                learnCounter += 1
            else: 
                learnCounter = 0
            if learnCounter > learnThreshold: 
                learn = False 
                learntStockouts = []
                for i in range(30):
                    learntStockouts.append(expectedStockouts)
        
        else:
            learntStockouts = learntStockouts[1:]
            learntStockouts.append(int(order <= demand[day]))
            
            # Starting Trigger
            if np.abs(np.mean(learntStockouts) - expectedStockouts) > stockoutThreshold:
                testCounter += 1
                if testCounter > testThreshold:
                    learn = True
                    learntDemand = learntDemand[-10:]
                    learntStockouts = learntStockouts[-10:]
                    learnCounter = 0
            else:
                testCounter = 0

        logs.loc[day, 'order'] = round(order,2)
        logs.loc[day, 'sales'] = round(sales,2)
        logs.loc[day, 'demand'] = round(demand[day],2)
        logs.loc[day, 'profit'] = round(profit,2)
        previousMean = np.mean(learntDemand)
    return logs


def adaptiveIncrement(price, cost, demand, w=3,
                incFactor=0.15, decFactor =0.15): 
    observedDemand = censorBySales(demand[:100], 110)
    demand = demand[100:]
    profit = 0
    logs = pd.DataFrame()

    overOrder = [0]
    incTrigger = 0

    for day in range(len(demand)):    
        
        inc = incTrigger*np.std(observedDemand)
        dec = np.mean(overOrder)
        order = np.mean(observedDemand) + np.std(observedDemand)*st.norm.ppf((price-cost)/price) + inc - dec
        sales = min(order, demand[day])

        observedDemand.append(sales) 
        overOrder.append(order - sales)
        profit += sales*price - order*cost

        if order == sales:
            incTrigger += incFactor
        else:
            incTrigger -= decFactor
        
        if len(overOrder) > w:
            overOrder = overOrder[1:]

        logs.loc[day, 'order'] = round(order,2)
        logs.loc[day,'sales'] = round(sales,2) 
        logs.loc[day, 'demand'] = round(demand[day],2)
        logs.loc[day, 'profit'] = round(profit,2)
    return logs


# K Fold Cross Validation 


def plotkFoldCV(price, cost, demand, fold):

    index = []
    incFactor = []
    decFactor = []
    window = []
    
    def sumError(ls1, ls2):
        error = []
        for i in np.arange(len(ls1)):
            error.append( np.square(ls1[i] - ls2[i]))
        return np.sum(error)

    K = np.arange(1,fold)

    for i in np.arange(1,10):
        index.append(i)

        incKfold = []
        decKfold = []
        windowKfold = [] 
        
        for k in K:
            kDemand = random.sample(demand, len(demand))
            optimal = optimalNewsvendor(price, cost, kDemand[100:], np.mean(kDemand), np.std(kDemand)).order
            incKfold.append(sumError(optimal, adaptiveIncrement(price, cost, kDemand, incFactor=i*0.05).order))
            decKfold.append(sumError(optimal, adaptiveIncrement(price, cost, kDemand, decFactor=i*0.05).order))
            windowKfold.append(sumError(optimal, adaptiveIncrement(price, cost, kDemand, w=i).order))
        incFactor.append(np.mean(incKfold))
        decFactor.append(np.mean(decKfold))
        window.append(np.mean(windowKfold))
    
    plt.plot(index, incFactor, label = 'alpha')
    plt.plot(index, decFactor, label = 'beta')
    plt.plot(index, window, label = 'window')
    plt.xlabel("parameter value")
    plt.ylabel("squared error")
    plt.legend()
    plt.title("10-Fold Cross Validation")
    plt.show()

# Plotting 
def plotOrders(logs):
    plt.style.use('seaborn')
    plt.plot(logs.index, logs.order, color = 'purple', label = 'order')
    plt.plot(logs.index, logs.demand, color = 'red', label = 'demand')
    plt.plot(logs.index, logs.sales, color = 'blue', label = 'sales')
    plt.legend()
    plt.ylabel('quantity')
    plt.xlabel('simulated days')
    plt.show()

def plotProfits(op, my, mle, mlew, dcl, ai):
    plt.style.use('seaborn')
    plt.plot(op.index, op.profit, color = 'blue', label = 'optimal')
    plt.plot(op.index, my.profit, color = 'red', label = 'myopic')
    plt.plot(op.index, mle.profit, color = 'green', label = 'MLE')
    plt.plot(op.index, mlew.profit, color = 'teal', label = 'MLE50')
    plt.plot(op.index, dcl.profit, color = 'cyan', label = 'DCL')
    plt.plot(op.index, ai.profit, color = 'pink', label = 'AI')

    plt.legend()
    plt.ylabel('profit')
    plt.xlabel('simulated days')
    plt.show()


def saveProfit(price, cost, demand, title, op):
    index = myopicNewsvendor(price, cost, demand).index
    d = demand[100:]
    bm = myopicNewsvendor(price, cost, demand).profit
    fl = fastLearning(price, cost, demand).profit
    dcl = dualCensoredLearning(price, cost, demand).profit
    ai = adaptiveIncrement(price, cost, demand).profit
    mle = MLE(price, cost, demand).profit
    mle50 = MLE(price, cost, demand, 50).profit

    bm_q = myopicNewsvendor(price, cost, demand).order
    fl_q = fastLearning(price, cost, demand).order
    dcl_q = dualCensoredLearning(price, cost, demand).order
    ai_q = adaptiveIncrement(price, cost, demand).order
    mle_q = MLE(price, cost, demand).order
    mle50_q = MLE(price, cost, demand, 50).order
    pd.DataFrame(list(zip(index, d, bm, fl, dcl, ai, mle, mle50, op.profit, 
                            bm_q, fl_q, dcl_q, ai_q, mle_q, mle50_q, op.order)), 
               columns =['day', 'd', 'bm', 'fl', 'dcl', 'ai', 'mle', 'mle50', 'op',
                        'bm_q', 'fl_q', 'dcl_q', 'ai_q', 'mle_q', 'mle50_q', 'op_q']).to_csv("data/" + title + ".csv", index = False)

## Simulation set up

def generateNormDemand(mean, std, n):
    l = []
    for i in range(n):
        l.append( max(int(random.gauss(mean, std)),0 ) )
    return l

def generateBetaDemand(mean, std, n):
    scale = mean + 3*std
    mean = mean/scale
    std = std/scale

    alpha = ((1-mean)/(std*std) - 1/mean) * mean*mean 
    beta = alpha*(1/mean - 1)
    l = []
    for i in range(n):
        l.append( max( int(np.random.beta(alpha, beta)*scale),0 ))
    return l


# Cummulative Profit

demandC = generateNormDemand(100, 30, 1100) + generateNormDemand(50, 30, 1000)

price = 1
cost = 0.5
demand = demandC
op = optimalNewsvendor(price, cost, demand[100:], 100, 30)
my = myopicNewsvendor(price, cost, demand)
mle = MLE(price, cost, demand)
mle50 = MLE(price, cost, demand, 50)
dcl = dualCensoredLearning(price, cost, demand)
ai = adaptiveIncrement(price, cost, demand)
plotProfits(op, my, mle,mle50, dcl, ai)


# Scenario A - Stationary Demand

demandA = generateNormDemand(100, 30, 2100)
op = optimalNewsvendor(1, 0.5, demandA[100:], 100, 30)
saveProfit(1, 0.5, demandA, 'A_0.5', op)

op = optimalNewsvendor(1, 0.3, demandA[100:], 100, 30)
saveProfit(1, 0.3, demandA, 'A_0.7', op)

op = optimalNewsvendor(1, 0.7, demandA[100:], 100, 30)
saveProfit(1, 0.7, demandA, 'A_0.3', op)


# Scenario B - Sudden Increase 

demandB = generateNormDemand(100, 30, 1100) + generateNormDemand(250, 30, 1000)

def optimalB(price, cost, demand): 
    profit = 0
    logs = pd.DataFrame() 
    mean = 100
    std = 30 
    for day in range(len(demand)): 
        if day == 1000:
            mean = 250
        order = mean + std*st.norm.ppf((price-cost)/price)
        sales = min(order, demand[day])
        profit += sales*price - order*cost

        logs.loc[day, 'order'] = round(order,2)
        logs.loc[day,'sales'] = round(sales,2) 
        logs.loc[day, 'demand'] = round(demand[day],2)
        logs.loc[day, 'profit'] = round(profit,2)    
    return logs

op = optimalB(1, 0.5, demandB[100:])
saveProfit(1, 0.5, demandB, 'B_0.5', op)

op = optimalB(1, 0.3, demandB[100:])
saveProfit(1, 0.3, demandB, 'B_0.7', op)

op = optimalB(1, 0.7, demandB[100:])
saveProfit(1, 0.7, demandB, 'B_0.3', op)


# Scenario C - Sudden Decrease

demandC = generateNormDemand(100, 30, 1100) + generateNormDemand(50, 30, 1000)

def optimalC(price, cost, demand): 
    profit = 0
    logs = pd.DataFrame() 
    mean = 100
    std = 30 
    for day in range(len(demand)):    
        if day == 1000:
            mean = 50
        order = mean + std*st.norm.ppf((price-cost)/price)
        sales = min(order, demand[day])
        profit += sales*price - order*cost

        logs.loc[day, 'order'] = round(order,2)
        logs.loc[day,'sales'] = round(sales,2) 
        logs.loc[day, 'demand'] = round(demand[day],2)
        logs.loc[day, 'profit'] = round(profit,2)    
    return logs

op = optimalC(1, 0.5, demandC[100:])
saveProfit(1, 0.5, demandC, 'C_0.5', op)

op = optimalC(1, 0.3, demandC[100:])
saveProfit(1, 0.3, demandC, 'C_0.7', op)

op = optimalC(1, 0.7, demandC[100:])
saveProfit(1, 0.7, demandC, 'C_0.3', op)


# Scenario D - Sudden Decrease with Aggressive Rebound

demandD = generateNormDemand(100, 30, 600) + generateNormDemand(20, 30, 200) + generateNormDemand(200, 30, 1300)

def optimalD(price, cost, demand): 
    profit = 0
    logs = pd.DataFrame() 
    mean = 100
    std = 30 
    for day in range(len(demand)):    
        if day == 500:
            mean = 20
        if day == 700:
            mean = 200
        order = mean + std*st.norm.ppf((price-cost)/price)
        sales = min(order, demand[day])
        profit += sales*price - order*cost

        logs.loc[day, 'order'] = round(order,2)
        logs.loc[day,'sales'] = round(sales,2) 
        logs.loc[day, 'demand'] = round(demand[day],2)
        logs.loc[day, 'profit'] = round(profit,2)    
    return logs

op = optimalD(1, 0.5, demandD[100:])
saveProfit(1, 0.5, demandD, 'D_0.5', op)

op = optimalD(1, 0.3, demandD[100:])
saveProfit(1, 0.3, demandD, 'D_0.7', op)

op = optimalD(1, 0.7, demandD[100:])
saveProfit(1, 0.7, demandD, 'D_0.3', op)


# Scenario E - Sudden Increase before Steep Decline

demandE = generateNormDemand(100, 30, 600) + generateNormDemand(200, 30, 200) + generateNormDemand(50, 30, 1300)

def optimalE(price, cost, demand): 
    profit = 0
    logs = pd.DataFrame() 
    mean = 100
    std = 30 
    for day in range(len(demand)):    
        if day == 500:
            mean = 200
        if day == 700:
            mean = 50
        
        order = mean + std*st.norm.ppf((price-cost)/price)
        sales = min(order, demand[day])
        profit += sales*price - order*cost

        logs.loc[day, 'order'] = round(order,2)
        logs.loc[day,'sales'] = round(sales,2) 
        logs.loc[day, 'demand'] = round(demand[day],2)
        logs.loc[day, 'profit'] = round(profit,2)    
    return logs

op = optimalE(1, 0.5, demandE[100:])
saveProfit(1, 0.5, demandE, 'E_0.5', op)

op = optimalE(1, 0.3, demandE[100:])
saveProfit(1, 0.3, demandE, 'E_0.7', op)

op = optimalE(1, 0.7, demandE[100:])
saveProfit(1, 0.7, demandE, 'E_0.3', op)


# Scenario F - Multiple Disruptive Events

demandF = generateNormDemand(100, 30, 600) + generateNormDemand(200, 30, 500) + generateNormDemand(30, 30, 500) + generateNormDemand(250, 30, 500) 

def optimalF(price, cost, demand): 
    profit = 0
    logs = pd.DataFrame() 
    mean = 100
    std = 30 
    for day in range(len(demand)):    
        if day == 500:
            mean = 200
        if day == 1000:
            mean = 30
        if day == 1500:
            mean = 250         
        order = mean + std*st.norm.ppf((price-cost)/price)
        sales = min(order, demand[day])
        profit += sales*price - order*cost

        logs.loc[day, 'order'] = round(order,2)
        logs.loc[day,'sales'] = round(sales,2) 
        logs.loc[day, 'demand'] = round(demand[day],2)
        logs.loc[day, 'profit'] = round(profit,2)    
    return logs

op = optimalF(1, 0.5, demandF[100:])
saveProfit(1, 0.5, demandF, 'F_0.5', op)

op = optimalF(1, 0.3, demandF[100:])
saveProfit(1, 0.3, demandF, 'F_0.7', op)

op = optimalF(1, 0.7, demandF[100:])
saveProfit(1, 0.7, demandF, 'F_0.3', op)


# MLE limited window

def saveProfitWindow(price, cost, demand, title, op):
    index = myopicNewsvendor(price, cost, demandF).index
    d = demand[100:]
    mle = MLE(price, cost, demand).profit
    mle20 = MLE(price, cost, demand,20).profit
    mle30 = MLE(price, cost, demand,30).profit
    mle50 = MLE(price, cost, demand,50).profit
    mle100 = MLE(price, cost, demand,100).profit
    mle150 = MLE(price, cost, demand,150).profit
    mle200 = MLE(price, cost, demand,200).profit
    mle300 = MLE(price, cost, demand,300).profit
    mle500 = MLE(price, cost, demand,500).profit
    
    pd.DataFrame(list(zip(index, d, op.profit, mle, mle20, mle30,
                            mle50, mle100, mle150, mle200,mle300,mle500)), 
               columns =['day', 'd', 'op', 'mle', 'mle20', 'mle30', 'mle50', 'mle100', 'mle150', 'mle200', 'mle300', 'mle500']).to_csv("data/" + title + ".csv", index = False)



op = optimalF(1, 0.5, demandF[100:])
saveProfitWindow(1, 0.5, demandF, 'G_0.5', op)

op = optimalF(1, 0.3, demandF[100:])
saveProfitWindow(1, 0.3, demandF, 'G_0.7', op)

op = optimalF(1, 0.7, demandF[100:])
saveProfitWindow(1, 0.7, demandF, 'G_0.3', op)

# Kfold 

plotkFoldCV(1, 0.5, generateNormDemand(100, 30, 110), 10)


# Sensitivity Analysis

stdList = [10, 30, 50]
for std in stdList:
    # A
    demand = generateNormDemand(100, std, 2100)
    op = optimalNewsvendor(1, 0.5, demand[100:], 100, std)
    saveProfit(1, 0.5, demand, 'A_std' + str(std), op)

    # B
    demand = generateNormDemand(100, std, 600) + generateNormDemand(250, std, 1500)
    op = optimalB(1, 0.5, demand[100:])
    saveProfit(1, 0.5, demand, 'B_std' + str(std), op)
    
    # C 
    demand = generateNormDemand(100, std, 1100) + generateNormDemand(50, std, 1000)
    op = optimalC(1, 0.5, demand[100:])
    saveProfit(1, 0.5, demand, 'C_std' + str(std), op)
    
    # D
    demand = generateNormDemand(100, std, 600) + generateNormDemand(20, std, 200) + generateNormDemand(200, std, 1300)
    op = optimalD(1, 0.5, demand[100:])
    saveProfit(1, 0.5, demand, 'D_std' + str(std), op)
    
    # E
    demand = generateNormDemand(100, std, 600) + generateNormDemand(200, std, 200) + generateNormDemand(50, std, 1300)
    op = optimalE(1, 0.5, demand[100:])
    saveProfit(1, 0.5, demand, 'E_std' + str(std), op)
    
    # F
    demand = generateNormDemand(100, std, 600) + generateNormDemand(200, std, 500) + generateNormDemand(30, std, 500) + generateNormDemand(250, std, 500) 
    op = optimalF(1, 0.5, demand[100:])
    saveProfit(1, 0.5, demand, 'F_std' + str(std), op)
    
plotkFoldCV(generateNormDemand(100, 30, 200))


# Beta Distribution

std = 30
demand = generateBetaDemand(100, std, 2100)
op = optimalNewsvendor(1, 0.5, demand[100:], 100, std)
saveProfit(1, 0.5, demand, 'A_beta', op)

# B
demand = generateBetaDemand(100, std, 600) + generateBetaDemand(250, std, 1500)
op = optimalB(1, 0.5, demand[100:])
saveProfit(1, 0.5, demand, 'B_beta', op)

# C 
demand = generateBetaDemand(100, std, 1100) + generateBetaDemand(50, std, 1000)
op = optimalC(1, 0.5, demand[100:])
saveProfit(1, 0.5, demand, 'C_beta', op)

# D
demand = generateBetaDemand(100, std, 600) + generateBetaDemand(20, std, 200) + generateBetaDemand(200, std, 1300)
op = optimalD(1, 0.5, demand[100:])
saveProfit(1, 0.5, demand, 'D_beta', op)

# E
demand = generateBetaDemand(100, std, 600) + generateBetaDemand(200, std, 200) + generateBetaDemand(50, std, 1300)
op = optimalE(1, 0.5, demand[100:])
saveProfit(1, 0.5, demand, 'E_beta', op)

# F
demand = generateBetaDemand(100, std, 600) + generateBetaDemand(200, std, 500) + generateBetaDemand(30, std, 500) + generateNormDemand(250, std, 500) 
op = optimalF(1, 0.5, demand[100:])
saveProfit(1, 0.5, demand, 'F_beta', op)
