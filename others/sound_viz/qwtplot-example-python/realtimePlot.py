import ui_plot
import sys
import numpy
from PyQt4 import QtCore, QtGui
import PyQt4.Qwt5 as Qwt

numPoints=1000
xs=numpy.arange(numPoints)
ys=numpy.sin(3.14159*xs*10/numPoints)

def plotSomething():
    global ys
    ys=numpy.roll(ys,-1)
    #print "PLOTTING"
    c.setData(xs, ys)
    uiplot.qwtPlot.replot()   

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    ### SET-UP WINDOWS
    
    # WINDOW plot
    win_plot = ui_plot.QtGui.QMainWindow()
    uiplot = ui_plot.Ui_win_plot()
    uiplot.setupUi(win_plot)
    uiplot.btnA.clicked.connect(plotSomething)
    uiplot.btnB.clicked.connect(lambda: uiplot.timer.setInterval(100.0))
    uiplot.btnC.clicked.connect(lambda: uiplot.timer.setInterval(10.0))
    uiplot.btnD.clicked.connect(lambda: uiplot.timer.setInterval(1.0))
    c=Qwt.QwtPlotCurve()  
    c.attach(uiplot.qwtPlot)

    uiplot.timer = QtCore.QTimer()
    uiplot.timer.start(100.0)
    
    win_plot.connect(uiplot.timer, QtCore.SIGNAL('timeout()'), plotSomething) 
    

    ### DISPLAY WINDOWS
    win_plot.show()

    #WAIT UNTIL QT RETURNS EXIT CODE
    sys.exit(app.exec_())