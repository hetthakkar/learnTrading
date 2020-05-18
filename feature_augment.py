import pandas as pd
def difference(dataDf, period):
    return dataDf.sub(dataDf.shift(period), fill_value=0)

def ewm(dataDf, halflife):
    return dataDf.ewm(halflife=halflife,ignore_na=False,min_periods=0,adjust=True).mean()

def rsi(data, period):
    data_upside = data.sub(data.shift(1), fill_value=0)
    data_downside = data_upside.copy()
    data_downside[data_upside > 0] = 0
    data_upside[data_upside < 0] = 0
    avg_upside = data_upside.rolling(period).mean()
    avg_downside = - data_downside.rolling(period).mean()
    rsi = 100 - (100 * avg_downside / (avg_downside + avg_upside))
    rsi[avg_downside == 0] = 100
    rsi[(avg_downside == 0) & (avg_upside == 0)] = 0

    return rsi


def create_features(data):
    basis_X = pd.DataFrame(index = data.index, columns =  [])
    
    basis_X['mom5'] = difference(data['Close'],6)
    basis_X['mom10'] = difference(data['Close'],11)
    basis_X['mom3'] = difference(data['Close'],4)
    basis_X['mom1'] = difference(data['Close'],2)

    basis_X['rsi14'] = rsi(data['Close'],14)
    basis_X['rsi7'] = rsi(data['Close'],7)
    basis_X['rsi21'] = rsi(data['Close'],21)
    basis_X['rsi30'] = rsi(data['Close'],30)
    
  
    basis_X['emabasis5'] = ewm(data['Close'],5)
    basis_X['emabasis10'] = ewm(data['Close'],10)

    basis_X['Open'] = data['Open']
    basis_X['High'] = data['High']
    basis_X['Low'] = data['Low']
    basis_X['Close'] = data['Close']

    basis_X['200DMA'] = data['Close'].rolling(200).mean()
    basis_X['50DMA'] = data['Close'].rolling(50).mean()
    basis_X['20DMA'] = data['Close'].rolling(20).mean()

    basis_X['Trend'] = data['Close'] - data['Close'].rolling(50).mean()

    basis_X['Range'] = data['High'] - data['Low']

    basis_X['Avg True Range'] = ewm(basis_X['Range'],28)
    
    basis_X = basis_X.fillna(0)
    
    return basis_X