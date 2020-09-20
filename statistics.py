import pandas as pd

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


def graph_pt_net_value(pt):
    y = pt.holdings['net_value'].to_list()
    x = pt.holdings.index.to_list()
    plt.figure(1)
    plt.title('Net value of the portfolio from 2005 until 2019')
    plt.xlabel('Date')
    plt.ylabel('Portfolio value [$]')
    plt.legend('Net value')
    plt.plot(x, y)
    plt.show()


def graph_pt_capital(pt):
    y = pt.holdings['capital'].to_list()
    x = pt.holdings.index.to_list()
    plt.figure(2)
    plt.title('Capital of the portfolio from 2005 until 2019')
    plt.xlabel('Date')
    plt.ylabel('Portfolio value [$]')
    plt.legend('Capital')
    plt.plot(x, y)
    plt.show()


def graph_vix(vix, title='VIX', fig=1):
    x = vix['date'].to_list()
    y = vix['close'].to_list()
    plt.figure(fig)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.plot(x, y)
    plt.show()


def graph_vix1_vix2(vix1, vix2, title='VIX-VIX', fig=1):
    x = vix1['date'].to_list()
    y1 = vix1['close'].to_list()
    y2 = vix2['close'].to_list()
    y = []
    for i, item in enumerate(y1):
        y.append(float(item))
    for i, item in enumerate(y2):
        y[i] = y[i] - float(item)
    plt.figure(fig)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Deference between VIXs')
    plt.plot(x, y)
    plt.show()


def graph_custom_vix(v, title='VIX', fig=1):
    x = v.vix['Date'].to_list()
    y = v.vix['Volatility'].to_list()
    plt.figure(fig)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.plot(x, y)
    plt.show()


def graph_max_score(max_score):
    y = max_score
    x = list(range(len(y)))
    plt.figure(1)
    plt.title('Max Scores')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.plot(x, y)
    plt.show()
