
import sys
from PyQt5 import QtWidgets,uic
from read_found_pairs import return_key_words
from read_found_pairs import return_scores
from read_found_pairs import return_top_gene
from read_found_pairs import return_top_lnc
global key,top
import pandas as pd
global ls_r
import cv2
from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QColorDialog, QFontDialog, QTextEdit, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem

key=''
top=''
ls_r=[]
qtCreatorFile = "mainwindow.ui" # Enter file here.
 
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
 
class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.graphicsView.scene_img = QGraphicsScene()
        self.imgShow = QPixmap()
        self.imgShow.load('1.png')
        self.imgShowItem = QGraphicsPixmapItem()
        self.imgShowItem.setPixmap(QPixmap(self.imgShow))
        k=5
        self.imgShowItem.setPixmap(QPixmap(self.imgShow).scaled(1100*k,400*k))  
        self.graphicsView.scene_img.addItem(self.imgShowItem)
        self.graphicsView.setScene(self.graphicsView.scene_img)
        self.graphicsView.fitInView(QGraphicsPixmapItem(QPixmap(self.imgShow)))

        self.setWindowTitle("LncRNA-Top-Version_1.0")
        self.pushButton.clicked.connect(self.Lookup)
        self.pushButton_2.clicked.connect(self.Find_pair)
        self.pushButton_5.clicked.connect(self.Exp)
        self.pushButton_3.clicked.connect(self.findtopk)
        self.pushButton_4.clicked.connect(self.save_as_file)
    def save_as_file(self):
        global key,top
        global ls_r
        if key=='' and top=='':
            string='''Please generate the top list by click "Find top k of specific LncRNA or Gene". '''
            self.textBrowser.setText(string)
            if len(ls_r)==0:
                string=string+'Because results list is empty.'
                self.textBrowser.setText(string)
        else:
            if len(ls_r)!=0:
                ls=pd.DataFrame(ls_r)
                ls.to_csv('Top_{}_results_of_{}.csv'.format(top,key),index=None)
                string='''Generated file: 'Top_{}_results_of_{}.csv'''.format(top,key)
                self.textBrowser.setText(string)
         
    def findtopk(self):
        global key,top
        global ls_r
        key=''
        top=''
        ls_r=[]
        if not (self.radioButton.isChecked()) and not (self.radioButton_2.isChecked()):
            string='''Please check Lnc's top K or Gene's top K button'''
            self.textBrowser.setText(string)
        if (self.radioButton.isChecked()):
            #print('topk of lnc')
            lnc=str(self.plainTextEdit.toPlainText())
            try:
              top=int(self.textEdit_2.toPlainText())
            except:
                top=0
            if len(lnc)!=0 and top!=0:
              
              list_results=return_top_lnc(lnc,top)
              ls_r=list_results
              top=len(ls_r)
              string='Results of lnc {},top{}:\n'.format(lnc,top)
              string=string+str(list_results)
              self.textBrowser.setText(string)
              key=lnc
              if len(ls_r)==0:
                  string=string+'''Lnc ensembl not found, please double-check or use look up 'key words' '''
                  self.textBrowser.setText(string)
            else:
                string='Please enter one LncRNA or Gene ensembl and topK number,K should be at least 1.'
                self.textBrowser.setText(string)
                
        if (self.radioButton_2.isChecked()):
            #print('topk of gene')
            gene=str(self.textEdit.toPlainText())
            try:
              top=int(self.textEdit_2.toPlainText())
            except:
                top=0
            if len(gene)!=0 and top!=0:
              list_results=return_top_gene(gene,top)
              ls_r=list_results
              top=len(ls_r)
              string='Results of Gene {},top{}:\n'.format(gene,top)
              string=string+str(list_results)
              self.textBrowser.setText(string)
              key=gene
              
              if len(ls_r)==0:
                  string=string+'Gene ensembl not found, please double-check'
                  self.textBrowser.setText(string)
            else:
                  string='Please enter one Gene ensemble and topK number,K should be at least 1.'
                  self.textBrowser.setText(string)
            
    def Exp(self):
        lnc='DUBR'
        gene='A1CF'
        string='Examples of DUBR and Gene:A1CF'
        top='10'
        self.textBrowser.setText(string)
        self.plainTextEdit.setPlainText(lnc)
        self.textEdit.setText(gene)
        self.textEdit_2.setText(top)
        self.radioButton.setChecked(True)
        
    def Lookup(self):
        keywords=self.plainTextEdit.toPlainText()
        if keywords!='' or len(keywords)!=0:
          list_kw=return_key_words('{}'.format(keywords))
          if len(list_kw)!=0:
            self.textBrowser.setText(str(list_kw))
          else:
            string='No results Found'
            self.textBrowser.setText(str(string))
    def Find_pair(self):
        lnc=str(self.plainTextEdit.toPlainText())
        gene=str(self.textEdit.toPlainText())
        if len(lnc)==0 or len(gene)==0:
             string='Please input lnc Name and gene name,or click example button'
             self.textBrowser.setText(str(string))
        else:
            key=return_scores(lnc,gene)
            self.textBrowser.setText(str(key))
        #print('find_pairs')
    #def CalculateTax(self):
    #    price = int(self.price_box.toPlainText())
    #    tax = (self.tax_rate.value())
    #    total_price = price+((tax / 100) * price)
    #    total_price_string = "The total price with tax is: " + str(total_price)
     #   self.testEdit.setText(total_price_string)
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
